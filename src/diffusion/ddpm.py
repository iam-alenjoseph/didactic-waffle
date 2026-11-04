"""Core DDPM (Denoising Diffusion Probabilistic Models) Engine for Video Synthesis.

Implements the forward Markovian diffusion process, analytical posterior calculation,
and the reverse ancestral sampling loop for spatio-temporal video tensors (B, C, T, H, W).
Developed in collaboration with researchers from Google DeepMind and ASU.
"""

from typing import Dict, Any, Optional, Tuple, Union, List, Callable
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    no_grad = torch.no_grad
except ImportError:
    HAS_TORCH = False
    def no_grad():
        def decorator(fn):
            return fn
        return decorator

from .schedules import DiffusionSchedule


class VideoDDPM:
    """Complete DDPM algorithm implementation for video synthesis.

    Operates on 5D tensors of shape (Batch, Channels, Time, Height, Width).
    Supports:
        - Arbitrary noise schedules (linear, cosine, sigmoid)
        - Epsilon, x_0, and v-prediction parameterizations
        - Multimodal conditioning (depth maps + surface normal maps)
        - Classifier-Free Guidance (CFG)
        - Analytical posterior sampling with variance clipping
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        device: str = "cpu"
    ):
        self.model = model
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_range = clip_range
        self.device = device

        self.schedule = DiffusionSchedule(
            timesteps=timesteps,
            schedule_type=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device
        )

        if HAS_TORCH:
            self.torch_constants = self.schedule.to_torch(device)

    def q_sample(
        self,
        x_start: Union["torch.Tensor", np.ndarray],
        t: Union["torch.Tensor", np.ndarray],
        noise: Optional[Union["torch.Tensor", np.ndarray]] = None
    ) -> Union["torch.Tensor", np.ndarray]:
        """Forward diffusion process q(x_t | x_0).

        Diffuse the data x_0 for a given number of diffusion steps t:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_start: Initial video tensor x_0 of shape (B, C, T, H, W).
            t: Discrete timesteps of shape (B,).
            noise: Optional Gaussian noise epsilon ~ N(0, I). If None, sampled from N(0, I).

        Returns:
            Noised video tensor x_t of shape (B, C, T, H, W).
        """
        if HAS_TORCH and isinstance(x_start, torch.Tensor):
            if noise is None:
                noise = torch.randn_like(x_start)

            sqrt_alpha_bar = self.schedule.extract(
                self.torch_constants["sqrt_alphas_cumprod"], t, x_start.shape
            )
            sqrt_one_minus_alpha_bar = self.schedule.extract(
                self.torch_constants["sqrt_one_minus_alphas_cumprod"], t, x_start.shape
            )
            return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
        else:
            if noise is None:
                noise = np.random.randn(*x_start.shape).astype(np.float32)

            sqrt_alpha_bar = self.schedule.extract(
                self.schedule.sqrt_alphas_cumprod, t, x_start.shape
            )
            sqrt_one_minus_alpha_bar = self.schedule.extract(
                self.schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def predict_start_from_noise(
        self,
        x_t: "torch.Tensor",
        t: "torch.Tensor",
        noise: "torch.Tensor"
    ) -> "torch.Tensor":
        """Reconstruct x_0 estimate given x_t and predicted noise epsilon."""
        sqrt_recip_alpha_bar = self.schedule.extract(
            self.torch_constants["sqrt_recip_alphas_cumprod"], t, x_t.shape
        )
        sqrt_recipm1_alpha_bar = self.schedule.extract(
            self.torch_constants["sqrt_recipm1_alphas_cumprod"], t, x_t.shape
        )
        return sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * noise

    def predict_noise_from_start(
        self,
        x_t: "torch.Tensor",
        t: "torch.Tensor",
        x_0: "torch.Tensor"
    ) -> "torch.Tensor":
        """Compute implied noise epsilon from x_t and reconstructed x_0."""
        sqrt_alpha_bar = self.schedule.extract(
            self.torch_constants["sqrt_alphas_cumprod"], t, x_t.shape
        )
        sqrt_one_minus_alpha_bar = self.schedule.extract(
            self.torch_constants["sqrt_one_minus_alphas_cumprod"], t, x_t.shape
        )
        return (x_t - sqrt_alpha_bar * x_0) / sqrt_one_minus_alpha_bar

    def q_posterior(
        self,
        x_start: "torch.Tensor",
        x_t: "torch.Tensor",
        t: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Compute the mean, variance, and log variance of analytical posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean_coef1 = self.schedule.extract(
            self.torch_constants["posterior_mean_coef1"], t, x_t.shape
        )
        posterior_mean_coef2 = self.schedule.extract(
            self.torch_constants["posterior_mean_coef2"], t, x_t.shape
        )
        posterior_variance = self.schedule.extract(
            self.torch_constants["posterior_variance"], t, x_t.shape
        )
        posterior_log_variance = self.schedule.extract(
            self.torch_constants["posterior_log_variance_clipped"], t, x_t.shape
        )

        posterior_mean = posterior_mean_coef1 * x_start + posterior_mean_coef2 * x_t
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        x_t: "torch.Tensor",
        t: "torch.Tensor",
        condition: Optional["torch.Tensor"] = None,
        cfg_scale: float = 1.0
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Calculates predicted mean mu_theta and variance for reverse step p_theta(x_{t-1} | x_t).

        Applies Classifier-Free Guidance (CFG) when cfg_scale > 1.0 and condition is provided.
        """
        if self.model is None:
            raise RuntimeError("Model must be assigned before executing reverse diffusion.")

        if cfg_scale > 1.0 and condition is not None:
            # Model forward with condition
            model_output_cond = self.model(x_t, t, condition=condition)
            # Model forward unconditional (zero or null condition)
            uncond = torch.zeros_like(condition)
            model_output_uncond = self.model(x_t, t, condition=uncond)
            # CFG extrapolation: eps = eps_uncond + s * (eps_cond - eps_uncond)
            model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)
        else:
            model_output = self.model(x_t, t, condition=condition)

        # Interpret output according to prediction_type
        if self.prediction_type == "epsilon":
            pred_noise = model_output
            x_recon = self.predict_start_from_noise(x_t, t=t, noise=pred_noise)
        elif self.prediction_type == "sample":
            x_recon = model_output
            pred_noise = self.predict_noise_from_start(x_t, t=t, x_0=x_recon)
        elif self.prediction_type == "v_prediction":
            # v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x_0
            sqrt_alpha_bar = self.schedule.extract(self.torch_constants["sqrt_alphas_cumprod"], t, x_t.shape)
            sqrt_one_minus_alpha_bar = self.schedule.extract(self.torch_constants["sqrt_one_minus_alphas_cumprod"], t, x_t.shape)
            x_recon = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * model_output
            pred_noise = sqrt_alpha_bar * model_output + sqrt_one_minus_alpha_bar * x_t
        else:
            raise ValueError(f"Unknown prediction_type {self.prediction_type}")

        if self.clip_sample:
            x_recon = torch.clamp(x_recon, min=self.clip_range[0], max=self.clip_range[1])

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x_t, t=t
        )

        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @no_grad()
    def p_sample(
        self,
        x_t: "torch.Tensor",
        t: "torch.Tensor",
        condition: Optional["torch.Tensor"] = None,
        cfg_scale: float = 1.0
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Single reverse ancestral sampling step x_{t-1} ~ p_theta(x_{t-1} | x_t)."""
        model_mean, _, model_log_variance, pred_x0 = self.p_mean_variance(
            x_t=x_t, t=t, condition=condition, cfg_scale=cfg_scale
        )

        # No noise added at step 0 (t == 0)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        x_prev = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return x_prev, pred_x0

    @no_grad()
    def sample_loop(
        self,
        shape: Tuple[int, int, int, int, int],
        condition: Optional["torch.Tensor"] = None,
        cfg_scale: float = 1.0,
        return_intermediates: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        """Full DDPM reverse generation loop from x_T ~ N(0, I) down to x_0.

        Args:
            shape: Video tensor shape (Batch, Channels, Time, Height, Width).
            condition: Optional multimodal condition (Depth + Normal maps).
            cfg_scale: Classifier-free guidance multiplier.
            return_intermediates: Whether to record trajectory snapshots.
            progress_callback: Optional callback receiving (current_step, total_steps).

        Returns:
            Generated video tensor of shape (B, C, T, H, W) normalized in [-1, 1].
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for sample_loop.")

        batch_size = shape[0]
        img = torch.randn(shape, device=self.device)
        intermediates = [img] if return_intermediates else None

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img, _ = self.p_sample(img, t, condition=condition, cfg_scale=cfg_scale)

            if return_intermediates and (i % max(1, self.timesteps // 10) == 0 or i == 0):
                intermediates.append(img)

            if progress_callback:
                progress_callback(self.timesteps - i, self.timesteps)

        if return_intermediates:
            return img, intermediates
        return img
