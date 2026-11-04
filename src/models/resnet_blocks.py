"""Spatio-Temporal ResNet blocks and 3D down/upsampling modules for Video UNets."""

from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .temporal_attention import TemporalSelfAttention


class SpatioTemporalResnetBlock(nn.Module if HAS_TORCH else object):
    """(2D + 1D) Spatio-Temporal Residual Block.

    Decomposes 3D convolutions into:
    1. Spatial Conv2D (or Conv3D with kernel 1x3x3) over each frame.
    2. Temporal Conv1D (or Conv3D with kernel 3x1x1) across time T.
    This factorized approach dramatically reduces parameter count and GPU memory
    while ensuring seamless weight transfer from pretrained 2D Hugging Face UNets.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        time_embed_dim: Optional[int] = None,
        groups: int = 32,
        dropout: float = 0.0,
        temporal_kernel_size: int = 3,
        use_temporal_attention: bool = True,
        zero_init_temporal: bool = True
    ):
        if not HAS_TORCH:
            return
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_temporal_attention = use_temporal_attention

        # Spatial path (1x3x3)
        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv_spatial1 = nn.Conv3d(
            in_channels, self.out_channels,
            kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )

        # Timestep conditioning injection
        if time_embed_dim is not None:
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, self.out_channels * 2) # scale and shift
            )
        else:
            self.time_proj = None

        self.norm2 = nn.GroupNorm(min(groups, self.out_channels), self.out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv_spatial2 = nn.Conv3d(
            self.out_channels, self.out_channels,
            kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )

        # Temporal path (3x1x1)
        pad_t = temporal_kernel_size // 2
        self.temporal_norm = nn.GroupNorm(min(groups, self.out_channels), self.out_channels)
        self.temporal_act = nn.SiLU()
        self.conv_temporal = nn.Conv3d(
            self.out_channels, self.out_channels,
            kernel_size=(temporal_kernel_size, 1, 1), padding=(pad_t, 0, 0)
        )

        if zero_init_temporal:
            nn.init.zeros_(self.conv_temporal.weight)
            if self.conv_temporal.bias is not None:
                nn.init.zeros_(self.conv_temporal.bias)

        # Optional Temporal Attention block
        if use_temporal_attention:
            self.temporal_attn = TemporalSelfAttention(channels=self.out_channels)
        else:
            self.temporal_attn = None

        # Residual shortcut
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv3d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(
        self,
        x: "torch.Tensor",
        time_emb: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        """Args:
            x: Video tensor of shape (B, C_in, T, H, W)
            time_emb: Timestep embedding of shape (B, time_embed_dim)

        Returns:
            Output video tensor of shape (B, C_out, T, H, W)
        """
        residual = self.shortcut(x)

        # 1. First Spatial Convolution
        h = self.act1(self.norm1(x))
        h = self.conv_spatial1(h)

        # 2. Inject Timestep Embedding (AdaGroupNorm scale & shift)
        if self.time_proj is not None and time_emb is not None:
            time_params = self.time_proj(time_emb) # (B, 2 * out_channels)
            time_params = time_params[:, :, None, None, None] # (B, 2 * out_channels, 1, 1, 1)
            scale, shift = time_params.chunk(2, dim=1)
            h = self.norm2(h) * (1.0 + scale) + shift
            h = self.act2(h)
        else:
            h = self.act2(self.norm2(h))

        h = self.dropout(h)
        h = self.conv_spatial2(h)

        # 3. Temporal Convolution (along T dimension)
        h_temp = self.temporal_act(self.temporal_norm(h))
        h_temp = self.conv_temporal(h_temp)
        h = h + h_temp

        # 4. Temporal Attention
        if self.temporal_attn is not None:
            h = self.temporal_attn(h)

        return residual + h


class Downsample3D(nn.Module if HAS_TORCH else object):
    """Spatial downsampling by factor of 2 while preserving temporal resolution T."""

    def __init__(self, channels: int, downsample_time: bool = False):
        if not HAS_TORCH:
            return
        super().__init__()
        self.downsample_time = downsample_time
        stride = (2, 2, 2) if downsample_time else (1, 2, 2)
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.conv(x)


class Upsample3D(nn.Module if HAS_TORCH else object):
    """Spatial upsampling by factor of 2 while preserving temporal resolution T."""

    def __init__(self, channels: int, upsample_time: bool = False):
        if not HAS_TORCH:
            return
        super().__init__()
        self.upsample_time = upsample_time
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        b, c, t, h, w = x.shape
        scale_factor = (2, 2, 2) if self.upsample_time else (1, 2, 2)
        x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")
        return self.conv(x)

