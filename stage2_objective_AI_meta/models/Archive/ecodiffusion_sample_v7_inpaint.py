"""
=============================================================================
ECODIFFUSION SAMPLE V7 (FIXED) — INPAINTING-GUIDED RECONSTRUCTION
=============================================================================

WHAT WAS WRONG IN THE PREVIOUS V7
---------------------------------
The previous v7 called  model(x, t_tensor, cond_ens)  which routes to
EcoDiffusionFixed.forward(). That method is the TRAINING forward — it
samples a fresh t internally and adds noise via diffusion.q_sample().
It is NOT a denoising step. That's why we got:
   TypeError: unsupported operand type(s) for *: 'Tensor' and 'dict'
   TypeError: forward() got an unexpected keyword argument 't'

THE FIX
-------
Mirror v6's calling pattern exactly:
  1. Compute  cond_emb = model.encode_condition(...)  ONCE up front
  2. In the denoising loop, call  noise_pred = model.unet(x, t_batch, cond_emb)
  3. Use model.diffusion.timesteps  and  model.diffusion.alphas_cumprod
  4. After each step, OVERWRITE observation cells with their forward-diffused
     ground truth value (this is the inpainting trick).

This matches the v6 architecture exactly, with one extra line per step
that does the inpainting overlay.
=============================================================================
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_v7_inpaint(
    model,
    condition,
    obs_mask,                  # (B, S, Y, X) — 1 where cell is observed
    obs_values=None,           # (B, S, Y, X) — true values at obs cells
    n_samples=8,
    ddim_steps=50,
    eta=0.5,
    inpaint_strength=1.0,      # 1.0 = hard inpainting (recommended)
    repaint_iterations=1,      # >1 = RePaint algorithm
    verbose=False,
):
    """
    Inpainting-guided DDIM sampling. Mirrors v6's call pattern exactly,
    plus an extra inpainting overlay step.

    Returns:
        x : (n_samples, B, S, Y, X) tensor of probability maps
    """
    device = next(model.parameters()).device

    # ── Shapes ──
    B = condition['env'].shape[0]
    Y, X = model.spatial_size
    T_total = model.diffusion.timesteps
    S = condition['env'].shape[1]

    # Default observation values: presence (=1) at observed cells
    if obs_values is None:
        obs_values = obs_mask.float()

    obs_mask = obs_mask.to(device).float()
    obs_values = obs_values.to(device).float()

    # ──────────────────────────────────────────────────────────
    # STEP 1: encode conditioning ONCE (mirrors v6 lines 122-131)
    # ──────────────────────────────────────────────────────────
    cond_emb = model.encode_condition(
        env=condition['env'],
        y_coords=condition['y_coords'],
        x_coords=condition['x_coords'],
        species_features=condition.get('species_features'),
        edge_index=condition.get('edge_index'),
        edge_weight=condition.get('edge_weight'),
        history=condition.get('history_P'),
    )  # (B, S, D) - per-species

    # ──────────────────────────────────────────────────────────
    # STEP 2: replicate everything for n_samples ensemble members
    # ──────────────────────────────────────────────────────────
    cond_emb_ens = cond_emb.repeat(n_samples, 1, 1)         # (n_samples*B, S, D)
    obs_mask_ens = obs_mask.repeat(n_samples, 1, 1, 1)
    obs_values_ens = obs_values.repeat(n_samples, 1, 1, 1)

    # ──────────────────────────────────────────────────────────
    # STEP 3: build the DDIM timestep schedule (matches v6 logic)
    # ──────────────────────────────────────────────────────────
    skip = T_total // ddim_steps
    timesteps = list(range(0, T_total, skip))

    alphas_cumprod = model.diffusion.alphas_cumprod.to(device)

    def get_ab(t):
        return alphas_cumprod[t]

    def forward_diffuse(x0, t_int, noise=None):
        """q(x_t | x_0) at integer timestep — for noising the observation values."""
        ab = get_ab(t_int)
        if noise is None:
            noise = torch.randn_like(x0)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    # ──────────────────────────────────────────────────────────
    # STEP 4: initialize from random noise
    # ──────────────────────────────────────────────────────────
    x = torch.randn(n_samples * B, S, Y, X, device=device)

    if verbose:
        print(f"  v7 INPAINT: n_samples={n_samples}, steps={ddim_steps}, "
              f"eta={eta}, inpaint_strength={inpaint_strength}, "
              f"repaint={repaint_iterations}")
        print(f"  obs_mask cells: {int(obs_mask.sum().item())} / "
              f"{obs_mask.numel()} ({100*obs_mask.mean().item():.4f}%)")

    # ──────────────────────────────────────────────────────────
    # STEP 5: DDIM denoising loop with inpainting overlay
    # ──────────────────────────────────────────────────────────
    for i in reversed(range(len(timesteps))):
        t_val = timesteps[i]
        t_batch = torch.full((n_samples * B,), t_val,
                             device=device, dtype=torch.long)

        for r_iter in range(repaint_iterations):
            # ── Predict noise (mirrors v6 line 199) ──
            noise_pred = model.unet(x, t_batch, cond_emb_ens)

            alpha_t = get_ab(t_val)
            alpha_prev = (get_ab(timesteps[i - 1])
                          if i > 0 else torch.tensor(1.0, device=device))

            # ── Predicted x_0 (mirrors v6 line 206) ──
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, 0.0, 1.0)

            # ── INPAINTING OVERLAY on x_0_pred ──
            # Force x_0_pred at observation cells to be the known truth value.
            if inpaint_strength > 0:
                x_0_pred = (
                    (1 - inpaint_strength * obs_mask_ens) * x_0_pred +
                    (inpaint_strength * obs_mask_ens) * obs_values_ens
                )

            # ── DDIM step (mirrors v6 lines 219-231) ──
            if eta > 0 and i > 0:
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_prev) / (1 - alpha_t + 1e-8)
                ) * torch.sqrt(1 - alpha_t / (alpha_prev + 1e-8))
                sigma_t = torch.clamp(sigma_t, 0, 0.3)
                eps = torch.randn_like(x)
                variance_term = torch.clamp(1 - alpha_prev - sigma_t**2, 0, 1)
                x_new = (torch.sqrt(alpha_prev) * x_0_pred
                         + torch.sqrt(variance_term) * noise_pred
                         + sigma_t * eps)
            else:
                x_new = (torch.sqrt(alpha_prev) * x_0_pred
                         + torch.sqrt(1 - alpha_prev) * noise_pred)

            # ── Final inpainting overlay on x at the new noise level ──
            # This is stronger: we directly set x at obs cells to the
            # forward-diffused ground truth, bypassing any noise the
            # DDIM step injected at those positions.
            if inpaint_strength > 0 and i > 0:
                t_prev = timesteps[i - 1]
                noise_obs = torch.randn_like(obs_values_ens)
                obs_at_prev = forward_diffuse(obs_values_ens, t_prev,
                                               noise=noise_obs)
                x_new = (
                    (1 - inpaint_strength * obs_mask_ens) * x_new +
                    (inpaint_strength * obs_mask_ens) * obs_at_prev
                )

            # RePaint: noise back to t and redo for consistency
            if r_iter < repaint_iterations - 1 and i > 0:
                noise_re = torch.randn_like(x_new)
                x = forward_diffuse(x_new, t_val, noise=noise_re)
            else:
                x = x_new

        if verbose and (i % 10 == 0 or i == 0 or i == len(timesteps) - 1):
            x_clamped = torch.clamp(x, 0, 1)
            print(f"    step {len(timesteps)-i}/{len(timesteps)}: "
                  f"t={t_val}, mean={x_clamped.mean():.4f}, "
                  f"max={x_clamped.max():.4f}")

    # Final clamp
    x = torch.clamp(x, 0, 1)

    # Reshape (n_samples*B, S, Y, X) -> (n_samples, B, S, Y, X)
    x = x.view(n_samples, B, S, Y, X)

    if verbose:
        sample_0 = x[0, 0]  # (S, Y, X)
        n_high = int((sample_0 > 0.5).sum().item())
        n_obs = int(obs_mask[0].sum().item())
        n_high_at_obs = int(((sample_0 > 0.5) & (obs_mask[0] > 0.5)).sum().item())
        n_fillin = n_high - n_high_at_obs
        print(f"  sample 0: cells>0.5={n_high}, obs={n_obs}, "
              f"obs_in_recon={n_high_at_obs}/{n_obs} "
              f"({100*n_high_at_obs/max(1,n_obs):.0f}%), "
              f"fill_in={n_fillin}, mean={sample_0.mean():.4f}")

    return x


# ─────────────────────────────────────────────────────────────────────
# Drop-in replacement matching the sample_v6 signature
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_v7(model, condition, n_samples=8, ddim_steps=50, eta=0.5,
              mode='inpaint', guidance_w=0.0, verbose=False,
              repaint_iterations=2):
    """
    Drop-in replacement for sample_v6 with inpainting capability.

    mode options:
        'inpaint'      : hard inpainting at observation cells (recommended)
        'soft_inpaint' : inpaint_strength=0.5 (compromise)
        'extrapolate'  : no inpainting (equivalent to v6, for comparison)
    """
    if 'history_P' not in condition or condition['history_P'] is None:
        from ecodiffusion_sample_v6_map_axel import sample_v6
        return sample_v6(model, condition, n_samples, ddim_steps, eta,
                          mode='extrapolate', guidance_w=guidance_w,
                          verbose=verbose)

    history = condition['history_P']  # (B, T, S, Y, X)
    obs_mask = (history[:, -1] > 0.5).float()  # (B, S, Y, X)
    obs_values = obs_mask

    if mode == 'extrapolate':
        inpaint_strength = 0.0
    elif mode == 'soft_inpaint':
        inpaint_strength = 0.5
    else:
        inpaint_strength = 1.0

    return sample_v7_inpaint(
        model, condition, obs_mask, obs_values=obs_values,
        n_samples=n_samples, ddim_steps=ddim_steps, eta=eta,
        inpaint_strength=inpaint_strength,
        repaint_iterations=repaint_iterations,
        verbose=verbose,
    )