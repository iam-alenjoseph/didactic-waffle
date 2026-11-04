"""Temporal Self-Attention and Cross-Attention Modules for Video Diffusion.

Maintains inter-frame temporal coherence, object permanence, and smooth motion
by computing multi-head attention across the time dimension T.
Zero-initialized output projection ensures identity initialization when inflating 2D UNets.
"""

import math
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TemporalSelfAttention(nn.Module if HAS_TORCH else object):
    """Multi-Head Self-Attention along the temporal dimension T."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        zero_init: bool = True
    ):
        if not HAS_TORCH:
            return
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim or (channels // num_heads)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        inner_dim = self.num_heads * self.head_dim

        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.to_qkv = nn.Linear(channels, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, channels),
            nn.Dropout(dropout)
        )

        if zero_init:
            # Zero-initialize the final projection so the layer initially acts as an identity
            nn.init.zeros_(self.to_out[0].weight)
            if self.to_out[0].bias is not None:
                nn.init.zeros_(self.to_out[0].bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Args:
            x: Video feature tensor of shape (B, C, T, H, W)

        Returns:
            Tensor of shape (B, C, T, H, W) with temporal self-attention applied.
        """
        residual = x
        b, c, t, h, w = x.shape

        # Normalize across channels
        x_norm = self.norm(x)

        # Reshape to (B * H * W, T, C) so each spatial coordinate is an independent sequence of length T
        x_flat = x_norm.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)

        qkv = self.to_qkv(x_flat).chunk(3, dim=-1)
        q, k, v = map(
            lambda t_mat: t_mat.reshape(b * h * w, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3),
            qkv
        ) # (B*H*W, num_heads, T, head_dim)

        # Scaled dot-product attention along T
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ v # (B*H*W, num_heads, T, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(b * h * w, t, self.num_heads * self.head_dim)
        out = self.to_out(out)

        # Reshape back to (B, C, T, H, W)
        out = out.reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2)

        return residual + out


class TemporalCrossAttention(nn.Module if HAS_TORCH else object):
    """Cross-Attention between video features and external multimodal condition tokens."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        zero_init: bool = True
    ):
        if not HAS_TORCH:
            return
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or (query_dim // num_heads)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        inner_dim = self.num_heads * self.head_dim

        self.norm = nn.GroupNorm(num_groups=min(32, query_dim), num_channels=query_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        if zero_init:
            nn.init.zeros_(self.to_out[0].weight)
            if self.to_out[0].bias is not None:
                nn.init.zeros_(self.to_out[0].bias)

    def forward(self, x: "torch.Tensor", context: "torch.Tensor") -> "torch.Tensor":
        """Args:
            x: Video tensor (B, C, T, H, W)
            context: Condition feature tensor (B, C_cond, T, H, W) or (B, Seq, C_cond)

        Returns:
            Conditioned video tensor (B, C, T, H, W)
        """
        residual = x
        b, c, t, h, w = x.shape

        x_norm = self.norm(x)
        # Reshape query to (B * H * W, T, C)
        q = self.to_q(x_norm.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c))

        if context.dim() == 5:
            # Spatio-temporal condition tensor
            b_c, c_c, t_c, h_c, w_c = context.shape
            context_flat = context.permute(0, 3, 4, 2, 1).reshape(b_c * h_c * w_c, t_c, c_c)
        else:
            context_flat = context

        k = self.to_k(context_flat)
        v = self.to_v(context_flat)

        # Multi-head reshape
        q = q.reshape(b * h * w, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(-1, k.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(-1, v.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(b * h * w, t, self.num_heads * self.head_dim)
        out = self.to_out(out)

        out = out.reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2)
        return residual + out

