"""
=============================================================================
ECODIFFUSION SAMPLE — v6 EXTRAPOLATION FIX
=============================================================================

WHY THIS FILE EXISTS
--------------------
The v5 sample() in your ecodiffusion.py produces predictions that mirror the
sparse input pattern instead of extrapolating. Diagnostic on your two recon
files shows fill-in cells (model > 0.5 where input was 0) = 0 — the model
literally never adds a cell beyond what was observed.

Three things in v5 cause this together:

  1.  noise_scale = clamp(x_prior * 5, 0, 1)
      → noise is ZERO at unobserved cells, so the diffusion never has any
        signal at those cells to denoise into a presence prediction.

  2.  start_timestep = 75   (out of 1000)
      → at t=75, alpha_cumprod ≈ 0.99: there is essentially no headroom
        for the model to override the initialization.

  3.  clamp(x, 0, 1) after every DDIM step
      → presses any small probability rise back down toward 0,
        especially when starting from x≈0 at unobserved cells.

WHAT v6 DOES
------------
v6 introduces three sampling MODES so the same model can produce three
different kinds of output, with no retraining:

  mode='echo'         — your old v5 behaviour (kept for backward compatibility)
  mode='extrapolate'  — start from full noise; let env+temporal+species
                        encoders drive the prediction; the sparse
                        observations enter ONLY via the temporal encoder,
                        not as a strong x_init prior.
                        THIS IS THE MODE FOR THE TRUTH | NOISY | RECON FIGURE.
  mode='guided'       — hybrid: anchor at observed cells but allow the
                        rest of the field to be reconstructed by the model.
                        Useful for cross-validation: predicted cells must
                        agree with observations (anchor) and the model
                        infers the rest.

USAGE
-----
Drop this file next to ecodiffusion.py and import its function:

    from ecodiffusion_sample_v6 import sample_v6

    preds = sample_v6(
        model       = model,                  # your loaded EcoDiffusionFixed
        condition   = condition_dict,         # same dict you build today
        n_samples   = 8,
        ddim_steps  = 50,
        eta         = 0.5,
        mode        = 'extrapolate',          # <── THIS IS THE FIX
        guidance_w  = 0.0,                    # 0 = pure model; >0 = anchor
    )

The returned tensor is shape (n_samples, B, S, Y, X) in [0,1] like before.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


@torch.no_grad()
def sample_v6(
    model,
    condition: Dict[str, torch.Tensor],
    n_samples: int = 1,
    ddim_steps: int = 50,
    eta: float = 0.5,
    mode: str = 'extrapolate',          # 'echo' | 'extrapolate' | 'guided'
    guidance_w: float = 0.0,            # only used in 'guided'
    start_timestep: Optional[int] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Sample reconstructions from the trained EcoDiffusion model.

    Parameters
    ----------
    model : EcoDiffusionFixed
        Already-loaded checkpoint, model.eval().
    condition : dict
        Must contain: env, y_coords, x_coords, species_features,
                      history_P  (B, T, S, Y, X) — last frame is the
                      sparse observation map.
        May contain: edge_index, edge_weight.
    n_samples : int
        Number of stochastic ensemble members to produce.
    ddim_steps : int
        Reverse diffusion steps. Use 50 for v5 parity.
    eta : float
        Stochastic DDIM parameter. 0=deterministic, 1=DDPM.
    mode : str
        'echo'         → reproduces v5 behaviour (no extrapolation).
        'extrapolate'  → start from full noise; let conditioning drive output.
                         This is what you want for AB14 / Axel's figure.
        'guided'       → start from full noise but pull cells where
                         x_prior=1 toward 1 with weight `guidance_w`.
                         Reasonable values: guidance_w ∈ [0.0, 0.5].
    guidance_w : float
        Strength of pull toward observations in 'guided' mode.
    start_timestep : int or None
        If None: 999 for 'extrapolate' / 'guided', 75 for 'echo'.
    verbose : bool
        Print diagnostic info.

    Returns
    -------
    samples : (n_samples, B, S, Y, X) torch.Tensor in [0, 1]
    """
    device = next(model.parameters()).device
    B = condition['env'].shape[0]
    S = condition['env'].shape[1]
    Y, X = model.spatial_size
    T = model.diffusion.timesteps

    # Encode conditioning ONCE (env, interactions, temporal, species)
    cond_emb = model.encode_condition(
        env=condition['env'],
        y_coords=condition['y_coords'],
        x_coords=condition['x_coords'],
        species_features=condition.get('species_features'),
        edge_index=condition.get('edge_index'),
        edge_weight=condition.get('edge_weight'),
        history=condition.get('history_P'),
    )  # (B, S, D)

    # Extract observation mask from history (last temporal frame)
    obs_mask = None
    if 'history_P' in condition and condition['history_P'] is not None:
        history = condition['history_P']
        x_prior = history[:, -1, :, :, :].to(device)   # (B, S, Y, X)
        # Pad/truncate species dim if needed
        if x_prior.shape[1] != S:
            if x_prior.shape[1] > S:
                x_prior = x_prior[:, :S]
            else:
                pad = torch.zeros(B, S - x_prior.shape[1], Y, X, device=device)
                x_prior = torch.cat([x_prior, pad], dim=1)
        obs_mask = (x_prior > 0.5).float()
        if verbose:
            print(f"  obs_mask: {int(obs_mask.sum().item())} cells of "
                  f"{B*S*Y*X} total ({100*obs_mask.mean().item():.4f}%)")
    else:
        x_prior = torch.zeros(B, S, Y, X, device=device)

    # Choose start timestep
    if start_timestep is None:
        if mode == 'echo':
            start_timestep = 75               # v5 default
        else:
            start_timestep = T - 1            # full denoising (key fix!)
    start_timestep = min(start_timestep, T - 1)

    # Build DDIM step schedule
    step_size = max(1, (start_timestep + 1) // ddim_steps)
    timesteps = list(range(0, start_timestep + 1, step_size))
    if verbose:
        print(f"  mode={mode}  start_t={start_timestep}  steps={len(timesteps)}  eta={eta}")

    samples = []
    for sample_idx in range(n_samples):

        # ─────────────────────────────────────────────────────────────
        # INITIALIZATION  (mode-dependent)
        # ─────────────────────────────────────────────────────────────
        noise = torch.randn(B, S, Y, X, device=device)

        if mode == 'echo':
            # v5 behaviour for backward compat / regression checks
            alpha_start = model.diffusion.alphas_cumprod[start_timestep]
            noise_scale = torch.clamp(x_prior * 5.0, 0.0, 1.0)
            sparse_noise = noise * noise_scale
            global_noise = noise * 0.05
            combined_noise = sparse_noise + global_noise
            x = (torch.sqrt(alpha_start) * x_prior
                 + torch.sqrt(1 - alpha_start) * combined_noise)
            x = torch.clamp(x, 0.0, 1.0)

        else:
            # 'extrapolate' and 'guided' both start from full Gaussian noise
            # so the model can put probability ANYWHERE based on the
            # env + temporal + species conditioning, not just where the
            # sparse observations happened to land.
            x = noise.clone()

        # ─────────────────────────────────────────────────────────────
        # DDIM REVERSE PROCESS
        # ─────────────────────────────────────────────────────────────
        for i in reversed(range(len(timesteps))):
            t_val = timesteps[i]
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)

            noise_pred = model.unet(x, t_batch, cond_emb)

            alpha_t = model.diffusion.alphas_cumprod[t_val]
            alpha_prev = (model.diffusion.alphas_cumprod[timesteps[i - 1]]
                          if i > 0 else torch.tensor(1.0, device=device))

            # Predicted x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # In modes that target a probability output we softly clamp x_0
            # to [0,1] but DON'T clamp x itself between steps
            x_0_pred = torch.clamp(x_0_pred, 0.0, 1.0)

            # Optional guidance: pull predicted x_0 toward 1 at observed cells
            if mode == 'guided' and guidance_w > 0 and obs_mask is not None:
                x_0_pred = (1.0 - guidance_w * obs_mask) * x_0_pred \
                           + guidance_w * obs_mask * 1.0

            # DDIM step (stochastic if eta>0)
            if eta > 0 and i > 0:
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_prev) / (1 - alpha_t + 1e-8)
                ) * torch.sqrt(1 - alpha_t / (alpha_prev + 1e-8))
                sigma_t = torch.clamp(sigma_t, 0, 0.3)
                eps = torch.randn_like(x)
                variance_term = torch.clamp(1 - alpha_prev - sigma_t ** 2, 0, 1)
                x = (torch.sqrt(alpha_prev) * x_0_pred
                     + torch.sqrt(variance_term) * noise_pred
                     + sigma_t * eps)
            else:
                x = (torch.sqrt(alpha_prev) * x_0_pred
                     + torch.sqrt(1 - alpha_prev) * noise_pred)

            # NOTE: NO clamp(0,1) here — that was the v5 bug.
            # We only clamp the FINAL output at end of sampling.

        # Final clamp to valid probability range
        x = torch.clamp(x, 0.0, 1.0)
        samples.append(x)

        if verbose and sample_idx == 0:
            n_cells = int((x > 0.5).sum().item())
            n_obs = int(obs_mask.sum().item()) if obs_mask is not None else 0
            n_fillin = int(((x > 0.5) & (obs_mask < 0.5)).sum().item()) \
                if obs_mask is not None else 0
            print(f"  sample 0: cells>0.5={n_cells}  obs={n_obs}  "
                  f"fill_in={n_fillin}  mean={x.mean().item():.4f}")

    return torch.stack(samples, dim=0)   # (n_samples, B, S, Y, X)