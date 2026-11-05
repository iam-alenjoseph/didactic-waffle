"""Video ControlNet with Zero Convolutions for Multimodal Conditioning.

Implements Video ControlNet adapting frozen base video diffusion models to external
structural condition streams (Depth + Normal maps).
Zero-convolutions (1x1x1 convs initialized to zero weights and biases) ensure that
training starts with the exact behavior of the base model before smoothly integrating
control guidance.
Reference: Zhang & Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models".
"""

from typing import List, Optional, Tuple, Dict, Any
import copy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .resnet_blocks import SpatioTemporalResnetBlock, Downsample3D
from .embeddings import SinusoidalPositionalEmbedding, TimestepEmbedding
from .temporal_attention import TemporalSelfAttention


def zero_module(module: "nn.Module") -> "nn.Module":
    """Initializes all weights and biases of a module to zero."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ZeroConv3D(nn.Module if HAS_TORCH else object):
    """1x1x1 convolution with weights and biases initialized to zero."""

    def __init__(self, in_channels: int, out_channels: int):
        if not HAS_TORCH:
            return
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        zero_module(self.conv)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.conv(x)


class VideoControlNet(nn.Module if HAS_TORCH else object):
    """ControlNet architecture tailored for 5D Spatio-Temporal Video Diffusion.

    Extracts multi-scale representations from Depth (1ch) and Normal (3ch) streams,
    projecting residual control features into the skip connections of the base UNet decoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        cond_channels: int = 4, # 1 depth + 3 normal
        model_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        temporal_conv_kernel_size: int = 3
    ):
        if not HAS_TORCH:
            return
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult

        time_embed_dim = model_channels * 4
        self.time_proj = SinusoidalPositionalEmbedding(dim=model_channels)
        self.time_embedding = TimestepEmbedding(model_channels, time_embed_dim)

        # Condition input pre-processor: 4ch (depth+normal) -> model_channels
        self.control_input_conv = nn.Sequential(
            nn.Conv3d(cond_channels, model_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(model_channels, model_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            zero_module(nn.Conv3d(model_channels, model_channels, kernel_size=1))
        )

        # Video input projection: 3ch RGB -> model_channels
        self.conv_in = nn.Conv3d(in_channels, model_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # Downsample encoder blocks
        self.down_blocks = nn.ModuleList()
        self.zero_convs = nn.ModuleList()

        curr_channels = model_channels
        self.zero_convs.append(ZeroConv3D(curr_channels, curr_channels))

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            res_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                res_blocks.append(
                    SpatioTemporalResnetBlock(
                        in_channels=curr_channels,
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        temporal_kernel_size=temporal_conv_kernel_size
                    )
                )
                curr_channels = out_ch
                self.zero_convs.append(ZeroConv3D(curr_channels, curr_channels))

            downsample = Downsample3D(curr_channels) if level < len(channel_mult) - 1 else nn.Identity()
            self.down_blocks.append(nn.ModuleDict({"blocks": res_blocks, "downsample": downsample}))

        # Middle bottleneck
        self.mid_block = SpatioTemporalResnetBlock(
            in_channels=curr_channels,
            out_channels=curr_channels,
            time_embed_dim=time_embed_dim,
            temporal_kernel_size=temporal_conv_kernel_size
        )
        self.mid_temporal_attn = TemporalSelfAttention(channels=curr_channels)
        self.mid_zero_conv = ZeroConv3D(curr_channels, curr_channels)

    def forward(
        self,
        x: "torch.Tensor", # (B, 3, T, H, W)
        timesteps: "torch.Tensor", # (B,)
        condition: "torch.Tensor" # (B, 4, T, H, W)
    ) -> List["torch.Tensor"]:
        """Returns multi-scale control residuals to inject into base UNet decoder skips."""
        # 1. Embed timesteps
        t_emb = self.time_embedding(self.time_proj(timesteps))

        # 2. Add processed condition to noisy video input
        h_cond = self.control_input_conv(condition)
        h = self.conv_in(x) + h_cond

        control_residuals = []
        zero_conv_idx = 0

        # Save initial projection residual
        control_residuals.append(self.zero_convs[zero_conv_idx](h))
        zero_conv_idx += 1

        # Downsample levels
        for down_level in self.down_blocks:
            for block in down_level["blocks"]:
                h = block(h, time_emb=t_emb)
                control_residuals.append(self.zero_convs[zero_conv_idx](h))
                zero_conv_idx += 1
            h = down_level["downsample"](h)

        # Middle bottleneck
        h = self.mid_block(h, time_emb=t_emb)
        h = self.mid_temporal_attn(h)
        control_residuals.append(self.mid_zero_conv(h))

        return control_residuals

