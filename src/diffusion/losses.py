"""Loss Objectives for Video Diffusion Models.

Implements:
1. Standard MSE Loss for epsilon/v-prediction
2. Min-SNR Gamma weighting (Hang et al., 2023) to eliminate loss spikes at low SNR
3. Spatio-Temporal Consistency Loss: penalizes inter-frame motion jitter and enforces smooth flow
"""

from typing import Optional, Dict
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class VideoDiffusionLoss(nn.Module if HAS_TORCH else object):
    """Loss module for video DDPM training."""

    def __init__(
        self,
        loss_type: str = "mse",
        min_snr_gamma: Optional[float] = 5.0,
        temporal_loss_weight: float = 0.1
    ):
        if not HAS_TORCH:
            return
        super().__init__()
        self.loss_type = loss_type
        self.min_snr_gamma = min_snr_gamma
        self.temporal_loss_weight = temporal_loss_weight

    def forward(
        self,
        pred_noise: "torch.Tensor",
        target_noise: "torch.Tensor",
        timesteps: "torch.Tensor",
        snr: Optional["torch.Tensor"] = None,
        pred_x0: Optional["torch.Tensor"] = None,
        target_x0: Optional["torch.Tensor"] = None
    ) -> Dict[str, "torch.Tensor"]:
        """Computes composite diffusion training loss.

        Args:
            pred_noise: Model output epsilon_theta of shape (B, C, T, H, W).
            target_noise: Ground truth noise epsilon of shape (B, C, T, H, W).
            timesteps: Batch timesteps (B,).
            snr: Signal-to-noise ratio tensor for current timesteps.
            pred_x0: Reconstructed x_0 video tensor.
            target_x0: Original ground truth x_0 video tensor.

        Returns:
            Dictionary containing 'total_loss', 'base_loss', and 'temporal_loss'.
        """
        # Base prediction loss (MSE or L1)
        if self.loss_type == "l1":
            base_loss = F.l1_loss(pred_noise, target_noise, reduction="none")
        else:
            base_loss = F.mse_loss(pred_noise, target_noise, reduction="none")

        # Reduce over spatial and temporal dimensions: (B, C, T, H, W) -> (B,)
        base_loss = base_loss.mean(dim=list(range(1, len(base_loss.shape))))

        # Min-SNR Gamma weighting: w(t) = min(SNR(t), gamma) / SNR(t)
        if self.min_snr_gamma is not None and snr is not None:
            gamma = torch.tensor(self.min_snr_gamma, device=snr.device)
            snr_t = snr.gather(-1, timesteps)
            weights = torch.minimum(snr_t, gamma) / snr_t
            base_loss = base_loss * weights

        mean_base_loss = base_loss.mean()

        # Spatio-Temporal Consistency Loss: penalizes motion gradient discrepancy
        temporal_loss = torch.tensor(0.0, device=pred_noise.device)
        if self.temporal_loss_weight > 0.0 and pred_x0 is not None and target_x0 is not None:
            # Temporal gradient: delta_t = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
            if pred_x0.shape[2] > 1:
                pred_temporal_diff = pred_x0[:, :, 1:, :, :] - pred_x0[:, :, :-1, :, :]
                target_temporal_diff = target_x0[:, :, 1:, :, :] - target_x0[:, :, :-1, :, :]
                temporal_loss = F.mse_loss(pred_temporal_diff, target_temporal_diff)

        total_loss = mean_base_loss + self.temporal_loss_weight * temporal_loss

        return {
            "total_loss": total_loss,
            "base_loss": mean_base_loss,
            "temporal_loss": temporal_loss
        }

