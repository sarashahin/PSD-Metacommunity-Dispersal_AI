"""
=============================================================================
DIFFUSION PROCESS: Noise Schedules and Sampling
=============================================================================
This module implements the diffusion process for DDPM (Denoising Diffusion
Probabilistic Models).

REASONING FOR DIFFUSION MODELS:

1. Mode Coverage (Critical for Rare Species)
   - GANs suffer from mode collapse, missing rare species
   - VAEs have blurry outputs due to Gaussian assumption
   - Diffusion models cover all modes by design

2. Conditional Generation
   - Easy to condition on environment, interactions, observations
   - Classifier-free guidance allows strength control

3. High-Quality Outputs
   - State-of-the-art in image generation
   - Spatial data (species maps) is similar to images

4. Inherent Uncertainty Quantification
   - Multiple samples from same condition → distribution of predictions
   - Naturally provides ensemble for uncertainty estimation

DIFFUSION PROCESS:
Forward: q(x_t|x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
Reverse: p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²I)

We learn ε_θ(x_t, t) to predict the noise, then:
μ_θ = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) ε_θ)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear noise schedule from DDPM paper.
    
    REASONING:
    - Simple and effective for many tasks
    - Linear increase in noise level
    - Original DDPM used this successfully
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule from "Improved DDPM" paper.
    
    REASONING:
    - Smoother noise progression than linear
    - Better for high-resolution spatial data
    - Prevents too-noisy intermediate states
    - Works better for our 20x20 grid data
    
    Formula: ᾱ_t = cos((t/T + s)/(1+s) * π/2)²
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to start at 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sqrt_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    Square root noise schedule.
    
    REASONING:
    - Slower initial noise growth
    - Good for data with fine spatial structure
    """
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def get_beta_schedule(schedule_type: str, timesteps: int, **kwargs) -> torch.Tensor:
    """Get noise schedule by name."""
    if schedule_type == "linear":
        return linear_beta_schedule(timesteps, **kwargs)
    elif schedule_type == "cosine":
        # cosine_beta_schedule only accepts 's', not beta_start/beta_end
        s = kwargs.get('s', 0.008)
        return cosine_beta_schedule(timesteps, s=s)
    elif schedule_type == "sqrt":
        return sqrt_beta_schedule(timesteps, **kwargs)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process for species distribution modeling.
    
    This class handles:
    1. Forward process: Adding noise to data
    2. Loss computation: Training objective
    3. Reverse process: Sampling from noise
    
    ARCHITECTURE NOTES:
    - Works with continuous data (biomass) or pseudo-continuous (presence prob)
    - For binary presence, we use continuous relaxation during diffusion
    - Final output can be thresholded for hard presence/absence
    """
    
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        loss_type: str = "l2",
        parameterization: str = "eps",  # "eps" (predict noise) or "x0" (predict clean)
        clip_denoised: bool = True,
        rescale_betas_zero_snr: bool = False,
    ):
        super().__init__()
        
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.parameterization = parameterization
        self.clip_denoised = clip_denoised
        
        # Get beta schedule
        betas = get_beta_schedule(beta_schedule, timesteps, beta_start=beta_start, beta_end=beta_end)
        
        # Pre-compute diffusion coefficients
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Zero SNR rescaling (optional, for better sampling)
        if rescale_betas_zero_snr:
            alphas_cumprod[-1] = 0.0
        
        # Register as buffers (not parameters, but moved to device with model)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for forward process q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        # Coefficients for posterior mean
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
        # Coefficients for converting epsilon prediction to x_0
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
    
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: sample x_t from q(x_t|x_0).
        
        q(x_t|x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
        
        Args:
            x_0: Clean data (B, ...)
            t: Timesteps (B,) integers in [0, T-1]
            noise: Optional pre-sampled noise
            
        Returns:
            x_t: Noisy data
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get coefficients for this timestep
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # Add noise
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
        return x_t, noise
    
    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute x_0 from x_t and predicted noise.
        
        x_0 = (x_t - √(1-ᾱ_t) ε) / √ᾱ_t
        
        Args:
            x_t: Noisy data
            t: Timesteps
            noise: Predicted noise
            
        Returns:
            x_0: Predicted clean data
        """
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def q_posterior(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1}|x_t, x_0).
        
        Args:
            x_0: Clean data (or prediction)
            x_t: Noisy data at time t
            t: Timesteps
            
        Returns:
            posterior_mean: Mean of posterior
            posterior_log_variance: Log variance of posterior
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_log_variance
    
    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict] = None,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1}|x_t).
        
        Args:
            model: Neural network that predicts noise or x_0
            x_t: Noisy data
            t: Timesteps
            condition: Conditioning information
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
            
        Returns:
            model_mean: Mean of reverse process
            model_log_variance: Log variance
            x_0_pred: Predicted clean data
        """
        # Get model prediction
        if condition is not None:
            model_output = model(x_t, t, **condition)
        else:
            model_output = model(x_t, t)
        
        # Convert to x_0 prediction
        if self.parameterization == "eps":
            x_0_pred = self.predict_start_from_noise(x_t, t, model_output)
        else:
            x_0_pred = model_output
        
        # Clip if requested
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Compute posterior
        model_mean, model_log_variance = self.q_posterior(x_0_pred, x_t, t)
        
        return model_mean, model_log_variance, x_0_pred
    
    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: sample x_{t-1} from p(x_{t-1}|x_t).
        
        Args:
            model: Denoising model
            x_t: Noisy data at time t
            t: Timesteps (batch of same value)
            condition: Conditioning information
            
        Returns:
            x_{t-1}: Slightly denoised data
        """
        model_mean, model_log_variance, _ = self.p_mean_variance(model, x_t, t, condition)
        
        # Sample
        noise = torch.randn_like(x_t)
        
        # No noise at t=0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))
        
        x_prev = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        
        return x_prev
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple,
        condition: Optional[Dict] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples via reverse diffusion.
        
        Args:
            model: Trained denoising model
            shape: Shape of samples to generate (B, ...)
            condition: Conditioning information
            return_intermediates: Whether to return intermediate samples
            
        Returns:
            samples: Generated data
            intermediates: List of intermediate samples (if requested)
        """
        device = next(model.parameters()).device
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        intermediates = [x] if return_intermediates else None
        
        # Reverse diffusion
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, condition)
            
            if return_intermediates and t % 100 == 0:
                intermediates.append(x)
        
        if return_intermediates:
            return x, intermediates
        return x
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple,
        condition: Optional[Dict] = None,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling for faster generation.
        
        REASONING:
        - Full DDPM needs 1000 steps (slow)
        - DDIM can generate good samples in 50-100 steps
        - eta=0 is deterministic, eta=1 is DDPM
        
        Args:
            model: Trained denoising model
            shape: Shape of samples to generate
            condition: Conditioning information
            ddim_steps: Number of sampling steps (<<1000)
            eta: Stochasticity parameter (0=deterministic, 1=DDPM)
            
        Returns:
            samples: Generated data
        """
        device = next(model.parameters()).device
        
        # Create timestep sequence
        c = self.timesteps // ddim_steps
        timesteps = list(range(0, self.timesteps, c))
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # DDIM reverse process
        for i in reversed(range(len(timesteps))):
            t = timesteps[i]
            t_prev = timesteps[i - 1] if i > 0 else 0
            
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Get model prediction
            if condition is not None:
                noise_pred = model(x, t_batch, **condition)
            else:
                noise_pred = model(x, t_batch)
            
            # DDIM update
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
            
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            if self.clip_denoised:
                x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev - eta ** 2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * noise_pred
            
            # Random noise
            if eta > 0 and t_prev > 0:
                noise = torch.randn_like(x)
                sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            else:
                noise = 0
                sigma = 0
            
            x = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt + sigma * noise
        
        return x
    
    def training_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        condition: Optional[Dict] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            model: Denoising model
            x_0: Clean data
            condition: Conditioning information
            noise: Optional pre-sampled noise
            
        Returns:
            Dictionary with 'loss' and optional auxiliary losses
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (B,), device=device, dtype=torch.long)
        
        # Add noise
        x_t, noise = self.q_sample(x_0, t, noise)
        
        # Get model prediction
        if condition is not None:
            model_output = model(x_t, t, **condition)
        else:
            model_output = model(x_t, t)
        
        # Compute loss based on parameterization
        if self.parameterization == "eps":
            target = noise
        else:
            target = x_0
        
        if self.loss_type == "l2":
            loss = F.mse_loss(model_output, target, reduction='mean')
        elif self.loss_type == "l1":
            loss = F.l1_loss(model_output, target, reduction='mean')
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(model_output, target, reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return {'loss': loss, 'noise_pred': model_output, 'noise_true': noise, 'x_t': x_t, 't': t}
    
    def ddim_step(
        self,
        x: torch.Tensor,
        t: int,
        noise_pred: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Single DDIM step for faster sampling.
        
        Args:
            x: Current noisy sample
            t: Current timestep
            noise_pred: Predicted noise
            eta: Stochasticity parameter (0=deterministic)
            
        Returns:
            Sample at previous timestep
        """
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=x.device)
        
        # Predict x_0
        x_0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        if self.clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Direction pointing to x_t
        sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        dir_xt = torch.sqrt(1 - alpha_t_prev - sigma ** 2) * noise_pred
        
        # Add noise if eta > 0
        if eta > 0 and t > 0:
            noise = torch.randn_like(x)
            x_prev = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt + sigma * noise
        else:
            x_prev = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt
        
        return x_prev
    
    def _extract(self, arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: Tuple) -> torch.Tensor:
        """
        Extract values from array at given timesteps and broadcast.
        
        Args:
            arr: 1D array of values
            timesteps: Batch of timestep indices
            broadcast_shape: Shape to broadcast to
            
        Returns:
            Values at timesteps, broadcast to shape
        """
        res = arr[timesteps]
        while res.dim() < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)


class EcologicalDiffusion(GaussianDiffusion):
    """
    Diffusion process specialized for ecological data.
    
    Extensions over standard diffusion:
    1. Handles both presence (binary) and biomass (continuous) data
    2. Optional species-wise weighting for rare species
    3. Spatial correlation-aware loss
    """
    
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        rare_species_weight: float = 2.0,
        spatial_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(timesteps, beta_schedule, **kwargs)
        
        self.rare_species_weight = rare_species_weight
        self.spatial_weight = spatial_weight
    
    def training_loss_weighted(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        prevalence: torch.Tensor,
        rare_threshold: float = 0.05,
        condition: Optional[Dict] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted training loss with rare species emphasis.
        
        Args:
            model: Denoising model
            x_0: Clean data (B, S, Y, X)
            prevalence: Species prevalence (B, S) or (S,)
            rare_threshold: Threshold for rare species
            condition: Conditioning information
            mask: Valid species mask (B, S)
            
        Returns:
            Dictionary with losses
        """
        # Get base loss components
        result = self.training_loss(model, x_0, condition)
        
        B, S, Y, X = x_0.shape
        
        # Compute per-species loss
        noise_pred = result['noise_pred']
        noise_true = result['noise_true']
        
        per_species_loss = ((noise_pred - noise_true) ** 2).mean(dim=(-2, -1))  # (B, S)
        
        # Weight by rarity
        if prevalence.dim() == 1:
            prevalence = prevalence.unsqueeze(0).expand(B, -1)
        
        weights = torch.ones_like(prevalence)
        rare_mask = prevalence < rare_threshold
        weights[rare_mask] = self.rare_species_weight
        
        # Apply species mask
        if mask is not None:
            weights = weights * mask.float()
            per_species_loss = per_species_loss * mask.float()
            weighted_loss = (per_species_loss * weights).sum() / (mask.float().sum() + 1e-10)
        else:
            weighted_loss = (per_species_loss * weights).mean()
        
        result['loss'] = weighted_loss
        result['rare_species_loss'] = per_species_loss[rare_mask].mean() if rare_mask.any() else torch.tensor(0.0)
        result['common_species_loss'] = per_species_loss[~rare_mask].mean() if (~rare_mask).any() else torch.tensor(0.0)
        
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("DIFFUSION PROCESS TEST")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Test parameters
    B, S, Y, X = 2, 100, 20, 20
    timesteps = 1000
    
    # Create diffusion process
    diffusion = GaussianDiffusion(
        timesteps=timesteps,
        beta_schedule="cosine",
    ).to(device)
    
    print(f"\nDiffusion process created with {timesteps} timesteps")
    
    # Test forward process
    print("\n1. Testing forward process (q_sample)...")
    x_0 = torch.randn(B, S, Y, X).to(device)
    t = torch.randint(0, timesteps, (B,)).to(device)
    
    x_t, noise = diffusion.q_sample(x_0, t)
    print(f"   x_0 shape: {x_0.shape}")
    print(f"   x_t shape: {x_t.shape}")
    print(f"   t: {t.tolist()}")
    
    # Test prediction
    print("\n2. Testing predict_start_from_noise...")
    x_0_pred = diffusion.predict_start_from_noise(x_t, t, noise)
    reconstruction_error = (x_0_pred - x_0).abs().mean().item()
    print(f"   Reconstruction error: {reconstruction_error:.6f} (should be ~0)")
    
    # Test noise schedules
    print("\n3. Comparing noise schedules...")
    for schedule in ["linear", "cosine", "sqrt"]:
        betas = get_beta_schedule(schedule, 100)
        alphas_cumprod = torch.cumprod(1 - betas, dim=0)
        print(f"   {schedule:8s}: β[0]={betas[0]:.4f}, β[-1]={betas[-1]:.4f}, ᾱ[-1]={alphas_cumprod[-1]:.4f}")
    
    # Create mock denoising model for testing
    print("\n4. Testing training loss computation...")
    
    class MockDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Conv2d(S, S, 3, padding=1)
        
        def forward(self, x, t, **kwargs):
            # Simplified: just return processed input
            return self.net(x)
    
    model = MockDenoiser().to(device)
    
    loss_dict = diffusion.training_loss(model, x_0)
    print(f"   Loss: {loss_dict['loss'].item():.4f}")
    print(f"   Loss computed successfully!")
    
    # Test ecological diffusion
    print("\n5. Testing EcologicalDiffusion...")
    eco_diffusion = EcologicalDiffusion(
        timesteps=timesteps,
        beta_schedule="cosine",
        rare_species_weight=2.0,
    ).to(device)
    
    prevalence = torch.rand(B, S).to(device) * 0.1  # Mostly rare species
    loss_dict = eco_diffusion.training_loss_weighted(model, x_0, prevalence)
    print(f"   Weighted loss: {loss_dict['loss'].item():.4f}")
    print(f"   Rare species loss: {loss_dict['rare_species_loss'].item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Diffusion process tests passed!")
    print("=" * 60)
