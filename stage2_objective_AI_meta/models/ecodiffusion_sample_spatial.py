"""
=============================================================================
ecodiffusion_sample_spatial.py  —  sampler for EcoDiffusionSpatial
=============================================================================

Matches the sample_v7 signature, but threads the clean spatial conditioning
stack (obs_mask, env, obs_decay) through the U-Net at every denoising step.

Differences vs ecodiffusion_sample_v7_inpaint.py:
  1. model.unet is called as  unet(x, t, cond_emb, cond_spatial)
     (the spatial U-Net needs the extra conditioning channels).
  2. cond_spatial is built ONCE from the condition dict and held FIXED for
     the whole reverse process — it is never noised.
  3. Default eta lowered 0.5 -> 0.15. With eta=0.5 the original sampler
     injected so much unstructured noise that each sample over-predicted
     (~35 cells vs truth ~9). Lower eta keeps ensemble diversity but each
     sample is much sharper.
  4. The hard-inpainting overlay at observation cells is KEPT — it pins the
     K known cells exactly, while the spatial conditioning channels let the
     denoiser actually propagate that information to unobserved cells.

USAGE
-----
  from ecodiffusion_sample_spatial import sample_spatial
  preds = sample_spatial(model, condition, n_samples=8, eta=0.15,
                         mode='inpaint', verbose=True)
  # preds : (n_samples, B, S, Y, X) probability maps
=============================================================================
"""

import torch


@torch.no_grad()
def sample_spatial(
    model,
    condition,
    n_samples=8,
    ddim_steps=50,
    eta=0.15,
    mode='inpaint',
    inpaint_strength=1.0,
    repaint_iterations=2,
    verbose=False,
):
    """
    Inpainting-guided DDIM sampling for EcoDiffusionSpatial.

    mode:
      'inpaint'      hard inpainting at observation cells  (recommended)
      'soft_inpaint' inpaint_strength = 0.5
      'extrapolate'  no inpainting overlay (spatial channels still active)
    """
    device = next(model.parameters()).device

    B = condition['env'].shape[0]
    S = condition['env'].shape[1]
    Y, X = model.spatial_size
    T_total = model.diffusion.timesteps

    if mode == 'extrapolate':
        inpaint_strength = 0.0
    elif mode == 'soft_inpaint':
        inpaint_strength = 0.5
    else:
        inpaint_strength = 1.0

    # ── observation mask / values from the (already-sparsified) history ──
    hist = condition.get('history_P')
    if hist is not None and hist.numel() > 0:
        obs_mask = (hist[:, -1] > 0.5).float()      # (B, S, Y, X)
    else:
        obs_mask = torch.zeros(B, S, Y, X, device=device)
    obs_values = obs_mask                            # presence = 1 at obs cells

    obs_mask = obs_mask.to(device)
    obs_values = obs_values.to(device)

    # ── conditioning: per-species FiLM vector + clean spatial stack ──
    cond_emb = model.encode_condition(
        env=condition['env'],
        y_coords=condition['y_coords'],
        x_coords=condition['x_coords'],
        species_features=condition.get('species_features'),
        edge_index=condition.get('edge_index'),
        edge_weight=condition.get('edge_weight'),
        history=condition.get('history_P'),
    )                                                # (B, S, D)
    cond_spatial = model.build_spatial_cond(condition)  # (B, S, 3, Y, X)

    # ── replicate across the ensemble dimension ──
    cond_emb_ens = cond_emb.repeat(n_samples, 1, 1)
    cond_spatial_ens = cond_spatial.repeat(n_samples, 1, 1, 1, 1)
    obs_mask_ens = obs_mask.repeat(n_samples, 1, 1, 1)
    obs_values_ens = obs_values.repeat(n_samples, 1, 1, 1)

    # ── DDIM schedule ──
    skip = max(1, T_total // ddim_steps)
    timesteps = list(range(0, T_total, skip))
    alphas_cumprod = model.diffusion.alphas_cumprod.to(device)

    def get_ab(t):
        return alphas_cumprod[t]

    def forward_diffuse(x0, t_int, noise=None):
        ab = get_ab(t_int)
        if noise is None:
            noise = torch.randn_like(x0)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    # ── start from noise ──
    x = torch.randn(n_samples * B, S, Y, X, device=device)

    if verbose:
        print(f"  spatial sampler: n_samples={n_samples}, steps={ddim_steps}, "
              f"eta={eta}, mode={mode}, inpaint_strength={inpaint_strength}")
        print(f"  obs cells: {int(obs_mask.sum().item())} / {obs_mask.numel()} "
              f"({100 * obs_mask.mean().item():.4f}%)")

    # ── reverse process ──
    for i in reversed(range(len(timesteps))):
        t_val = timesteps[i]
        t_batch = torch.full((n_samples * B,), t_val, device=device,
                             dtype=torch.long)

        for r_iter in range(repaint_iterations):
            # spatial U-Net: x_t + clean conditioning channels
            noise_pred = model.unet(x, t_batch, cond_emb_ens, cond_spatial_ens)

            alpha_t = get_ab(t_val)
            alpha_prev = (get_ab(timesteps[i - 1])
                          if i > 0 else torch.tensor(1.0, device=device))

            x_0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, 0.0, 1.0)

            # hard inpainting overlay on the predicted clean map
            if inpaint_strength > 0:
                x_0_pred = ((1 - inpaint_strength * obs_mask_ens) * x_0_pred
                            + (inpaint_strength * obs_mask_ens) * obs_values_ens)

            # DDIM step
            if eta > 0 and i > 0:
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_prev) / (1 - alpha_t + 1e-8)
                ) * torch.sqrt(1 - alpha_t / (alpha_prev + 1e-8))
                sigma_t = torch.clamp(sigma_t, 0, 0.3)
                eps = torch.randn_like(x)
                variance_term = torch.clamp(1 - alpha_prev - sigma_t ** 2, 0, 1)
                x_new = (torch.sqrt(alpha_prev) * x_0_pred
                         + torch.sqrt(variance_term) * noise_pred
                         + sigma_t * eps)
            else:
                x_new = (torch.sqrt(alpha_prev) * x_0_pred
                         + torch.sqrt(1 - alpha_prev) * noise_pred)

            # overlay the forward-diffused observations at the new noise level
            if inpaint_strength > 0 and i > 0:
                t_prev = timesteps[i - 1]
                obs_at_prev = forward_diffuse(obs_values_ens, t_prev)
                x_new = ((1 - inpaint_strength * obs_mask_ens) * x_new
                         + (inpaint_strength * obs_mask_ens) * obs_at_prev)

            # RePaint: renoise and redo for consistency
            if r_iter < repaint_iterations - 1 and i > 0:
                x = forward_diffuse(x_new, t_val)
            else:
                x = x_new

        if verbose and (i % 10 == 0 or i == 0 or i == len(timesteps) - 1):
            xc = torch.clamp(x, 0, 1)
            print(f"    step {len(timesteps)-i}/{len(timesteps)}: t={t_val}, "
                  f"mean={xc.mean():.4f}, max={xc.max():.4f}")

    x = torch.clamp(x, 0, 1)
    x = x.view(n_samples, B, S, Y, X)

    if verbose:
        s0 = x[0, 0]
        n_high = int((s0 > 0.5).sum().item())
        n_obs = int(obs_mask[0].sum().item())
        n_at_obs = int(((s0 > 0.5) & (obs_mask[0] > 0.5)).sum().item())
        print(f"  sample 0: cells>0.5={n_high}, obs={n_obs}, "
              f"obs_in_recon={n_at_obs}/{n_obs}, fill_in={n_high - n_at_obs}, "
              f"mean={s0.mean():.4f}")

    return x