"""Timestep and conditioning embeddings for Spatio-Temporal Video UNets."""

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SinusoidalPositionalEmbedding(nn.Module if HAS_TORCH else object):
    """Sinusoidal positional embeddings for diffusion timesteps."""

    def __init__(self, dim: int, max_period: int = 10000):
        if not HAS_TORCH:
            return
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: "torch.Tensor") -> "torch.Tensor":
        """Args:
            timesteps: 1D Tensor of shape (B,)

        Returns:
            Positional embeddings of shape (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class TimestepEmbedding(nn.Module if HAS_TORCH else object):
    """Projects sinusoidal embeddings to the UNet hidden channels with SiLU activations."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        if not HAS_TORCH:
            return
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else nn.GELU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: "torch.Tensor") -> "torch.Tensor":
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample

