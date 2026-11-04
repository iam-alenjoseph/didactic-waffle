"""3D Axial Rotary Positional Embeddings (Spatio-Temporal RoPE).

Decomposes the query/key channel space into temporal, height, and width sub-spaces,
applying rotary frequency modulations independently along each axis:
    R_3D(t, y, x) = diag( R_theta(t), R_theta(y), R_theta(x) )
This eliminates fixed context window limitations and ensures continuous phase stability
for video rollouts.
Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding".
"""

import math
from typing import Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def precompute_freqs_1d(dim: int, max_seq_len: int, theta: float = 10000.0) -> "torch.Tensor":
    """Computes complex exponential rotary frequencies for 1D sequence."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # Return polar coordinates (cos, sin)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb_1d(
    x: "torch.Tensor", # (..., SeqLen, HeadDim)
    cos: "torch.Tensor", # (SeqLen, HeadDim // 2)
    sin: "torch.Tensor"  # (SeqLen, HeadDim // 2)
) -> "torch.Tensor":
    """Applies 1D rotary embedding to input tensor."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]

    # Rotate vector: [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    return torch.cat([rotated_x1, rotated_x2], dim=-1)


class SpatioTemporalRoPE(nn.Module if HAS_TORCH else object):
    """3D Axial Rotary Positional Embedding for Spatio-Temporal Video Attention.

    Splits the head dimension into 3 parts:
        d_temporal = head_dim * temporal_ratio
        d_height   = (head_dim - d_temporal) // 2
        d_width    = (head_dim - d_temporal) // 2
    and applies independent 1D rotary phase shifts along time T, height H, and width W.
    """

    def __init__(
        self,
        head_dim: int = 64,
        max_t: int = 64,
        max_h: int = 128,
        max_w: int = 128,
        temporal_ratio: float = 0.5,
        theta: float = 10000.0
    ):
        if not HAS_TORCH:
            return
        super().__init__()
        self.head_dim = head_dim
        self.d_t = int(head_dim * temporal_ratio)
        if self.d_t % 2 != 0:
            self.d_t += 1
        rem = head_dim - self.d_t
        self.d_h = rem // 2
        if self.d_h % 2 != 0:
            self.d_h += 1
        self.d_w = head_dim - self.d_t - self.d_h

        # Precompute frequencies
        cos_t, sin_t = precompute_freqs_1d(self.d_t, max_t, theta=theta)
        cos_h, sin_h = precompute_freqs_1d(self.d_h, max_h, theta=theta)
        cos_w, sin_w = precompute_freqs_1d(self.d_w, max_w, theta=theta)

        self.register_buffer("cos_t", cos_t, persistent=False)
        self.register_buffer("sin_t", sin_t, persistent=False)
        self.register_buffer("cos_h", cos_h, persistent=False)
        self.register_buffer("sin_h", sin_h, persistent=False)
        self.register_buffer("cos_w", cos_w, persistent=False)
        self.register_buffer("sin_w", sin_w, persistent=False)

    def forward(
        self,
        q: "torch.Tensor", # (B, Heads, T, H, W, HeadDim)
        k: "torch.Tensor"  # (B, Heads, T, H, W, HeadDim)
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Applies 3D rotary transformation to queries and keys."""
        if not HAS_TORCH:
            return q, k

        B, Heads, T, H, W, D = q.shape

        # Slice channel dimension into T, H, W components
        q_t, q_h, q_w = q[..., : self.d_t], q[..., self.d_t : self.d_t + self.d_h], q[..., self.d_t + self.d_h :]
        k_t, k_h, k_w = k[..., : self.d_t], k[..., self.d_t : self.d_t + self.d_h], k[..., self.d_t + self.d_h :]

        # 1. Temporal rotation (along T dimension)
        cos_t = self.cos_t[:T].view(1, 1, T, 1, 1, -1)
        sin_t = self.sin_t[:T].view(1, 1, T, 1, 1, -1)
        q_t = apply_rotary_emb_1d(q_t, cos_t, sin_t)
        k_t = apply_rotary_emb_1d(k_t, cos_t, sin_t)

        # 2. Height rotation (along H dimension)
        cos_h = self.cos_h[:H].view(1, 1, 1, H, 1, -1)
        sin_h = self.sin_h[:H].view(1, 1, 1, H, 1, -1)
        q_h = apply_rotary_emb_1d(q_h, cos_h, sin_h)
        k_h = apply_rotary_emb_1d(k_h, cos_h, sin_h)

        # 3. Width rotation (along W dimension)
        cos_w = self.cos_w[:W].view(1, 1, 1, 1, W, -1)
        sin_w = self.sin_w[:W].view(1, 1, 1, 1, W, -1)
        q_w = apply_rotary_emb_1d(q_w, cos_w, sin_w)
        k_w = apply_rotary_emb_1d(k_w, cos_w, sin_w)

        # Re-concatenate rotated dimensions
        q_out = torch.cat([q_t, q_h, q_w], dim=-1)
        k_out = torch.cat([k_t, k_h, k_w], dim=-1)
        return q_out, k_out

