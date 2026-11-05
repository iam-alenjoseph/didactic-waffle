"""Multimodal Conditioning and Classifier-Free Guidance (CFG) for Video Diffusion.

Handles conditioning on 2D geometric projections, Depth maps (1 channel),
and Surface Normal maps (3 channels) acquired via Unreal Engine.
"""

from typing import Optional, Tuple, Dict
import random

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MultimodalConditionEncoder(nn.Module if HAS_TORCH else object):
    """Encodes Depth (1ch) and Normal (3ch) maps into spatio-temporal representations.

    Can be used either for cross-attention conditioning or feature concatenation.
    """

    def __init__(
        self,
        in_channels: int = 4, # 1 (depth) + 3 (normal)
        out_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (64, 128, 256)
    ):
        if not HAS_TORCH:
            return
        super().__init__()
        layers = []
        curr_in = in_channels
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv3d(curr_in, h_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.GroupNorm(num_groups=min(8, h_dim), num_channels=h_dim),
                nn.SiLU(),
            ])
            curr_in = h_dim

        self.net = nn.Sequential(*layers)
        self.proj = nn.Conv3d(curr_in, out_dim, kernel_size=1)

    def forward(self, condition: "torch.Tensor") -> "torch.Tensor":
        """Args:
            condition: Multimodal tensor of shape (B, 4, T, H, W) where channels 0:1
                       are depth and 1:4 are RGB-encoded surface normal maps.

        Returns:
            Condition feature map of shape (B, out_dim, T, H, W).
        """
        features = self.net(condition)
        return self.proj(features)


class ClassifierFreeGuidanceHelper:
    """Manages conditional masking for Classifier-Free Guidance during training and sampling."""

    def __init__(self, uncond_prob: float = 0.1):
        self.uncond_prob = uncond_prob

    def mask_condition_for_training(
        self,
        condition: Optional["torch.Tensor"]
    ) -> Optional["torch.Tensor"]:
        """Randomly zeroes out the condition tensor with probability `uncond_prob`."""
        if condition is None or not HAS_TORCH:
            return condition

        batch_size = condition.shape[0]
        # Generate independent dropout mask per batch item
        mask = torch.rand(batch_size, device=condition.device) > self.uncond_prob
        mask = mask.view(batch_size, *([1] * (len(condition.shape) - 1)))
        return condition * mask.float()

    @staticmethod
    def combine_guided_predictions(
        uncond_pred: "torch.Tensor",
        cond_pred: "torch.Tensor",
        guidance_scale: float
    ) -> "torch.Tensor":
        """Computes classifier-free guided noise estimate:
            eps = eps_uncond + s * (eps_cond - eps_uncond)
        """
        return uncond_pred + guidance_scale * (cond_pred - uncond_pred)

