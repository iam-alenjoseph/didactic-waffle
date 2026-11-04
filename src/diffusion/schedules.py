"""Diffusion Noise Schedules.

Implements Linear, Cosine (Nichol & Dhariwal, 2021), and Sigmoid noise variance
schedules for Denoising Diffusion Probabilistic Models (DDPM).

Computes analytical pre-factors for forward diffusion:
    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
and the reverse posterior distribution:
    q(x_{t-1} | x_t, x_0) = N(x_{t-1}; mu_tilde_t(x_t, x_0), beta_tilde_t * I)
"""

import math
from typing import Dict, Union, Tuple
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> np.ndarray:
    """Standard linear beta schedule from Ho et al. (2020).

    Args:
        timesteps: Total number of diffusion steps T.
        beta_start: Starting variance beta_1.
        beta_end: Ending variance beta_T.

    Returns:
        1D numpy array of shape (timesteps,) containing beta_t for t in [1, T].
    """
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> np.ndarray:
    """Cosine beta schedule proposed in Nichol & Dhariwal (2021) "Improved DDPM".

    Prevents abrupt noise drop-off at small t and preserves subtle structural details
    vital for spatio-temporal video coherence.

    Args:
        timesteps: Total number of diffusion steps T.
        s: Small offset to prevent beta_t from being 0 near t = 0.
        max_beta: Upper clip threshold for beta values to avoid singularities.

    Returns:
        1D numpy array of shape (timesteps,) containing beta_t.
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-5, max_beta)


def sigmoid_beta_schedule(timesteps: int, start: float = -3.0, end: float = 3.0, tau: float = 1.0) -> np.ndarray:
    """Sigmoid beta schedule for smooth progression in complex multi-scale video diffusion.

    Args:
        timesteps: Total number of diffusion steps T.
        start: Lower sigmoid asymptote bound.
        end: Upper sigmoid asymptote bound.
        tau: Temperature scaling parameter.

    Returns:
        1D numpy array of shape (timesteps,) containing beta_t.
    """
    steps = timesteps + 1
    t = np.linspace(0, timesteps, steps, dtype=np.float64)
    v_start = 1.0 / (1.0 + np.exp(-start / tau))
    v_end = 1.0 / (1.0 + np.exp(-end / tau))
    alphas_cumprod = 1.0 / (1.0 + np.exp(-(start + (end - start) * (t / timesteps)) / tau))
    alphas_cumprod = (alphas_cumprod - v_end) / (v_start - v_end)
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-5, 0.999)


class DiffusionSchedule:
    """Comprehensive container for all precomputed diffusion constants and parameters.

    Provides exact analytical coefficients for both forward noising and reverse sampling.
    Can export tensors to either PyTorch (GPU/CPU) or NumPy.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule_type: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu"
    ):
        self.timesteps = timesteps
        self.schedule_type = schedule_type
        self.device = device

        if schedule_type == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule_type == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule_type '{schedule_type}'. Choose 'linear', 'cosine', or 'sigmoid'.")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # Signal-to-Noise Ratio (SNR) for Min-SNR weighting
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        # beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Clipped because posterior variance is 0 at t = 0
        self.posterior_log_variance_clipped = np.log(
            np.maximum(self.posterior_variance, 1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        # Coefficients for predicting x_0 from epsilon
        self.pred_x0_coef1 = 1.0 / self.sqrt_alphas_cumprod
        self.pred_x0_coef2 = self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod

    def to_torch(self, device: str = None):
        """Converts all numpy precomputed coefficients to PyTorch tensors on the target device."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed in the current environment.")
        dev = device or self.device
        attrs = [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "log_one_minus_alphas_cumprod", "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod", "snr", "posterior_variance",
            "posterior_log_variance_clipped", "posterior_mean_coef1",
            "posterior_mean_coef2", "pred_x0_coef1", "pred_x0_coef2"
        ]
        torch_dict = {}
        for attr in attrs:
            arr = getattr(self, attr)
            t = torch.from_numpy(arr).float().to(dev)
            setattr(self, f"torch_{attr}", t)
            torch_dict[attr] = t
        return torch_dict

    def extract(self, a: Union[np.ndarray, "torch.Tensor"], t: Union[np.ndarray, "torch.Tensor"], x_shape: Tuple[int, ...]):
        """Extract coefficients at specified timesteps t and reshape for broadcasting over x.

        Supports input tensor shape: (Batch, Channels, Time, Height, Width) for 5D video tensors.

        Args:
            a: Precomputed 1D coefficient array or tensor.
            t: Batch of timesteps, shape (Batch,).
            x_shape: Target broadcast shape, e.g. (B, C, T, H, W).

        Returns:
            Extracted values reshaped to (B, 1, 1, 1, 1) for 5D video tensors.
        """
        if HAS_TORCH and isinstance(a, torch.Tensor):
            out = a.gather(-1, t)
            reshape_dims = [t.shape[0]] + [1] * (len(x_shape) - 1)
            return out.reshape(*reshape_dims)
        else:
            if isinstance(t, (int, np.integer)):
                val = a[t]
                return np.full(x_shape, val)
            out = np.take(a, t)
            reshape_dims = [len(t)] + [1] * (len(x_shape) - 1)
            return out.reshape(*reshape_dims)

