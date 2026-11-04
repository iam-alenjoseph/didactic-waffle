"""Unified Spatio-Temporal Video UNet Architecture.

Implements a full 3D / (2D + 1D) Spatio-Temporal Video UNet with cross-attention
and concatenation for multimodal conditioning (depth + normal maps).
Developed in collaboration with researchers from Google DeepMind and ASU.
"""

from typing import Optional, Tuple, Dict, Any
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .embeddings import SinusoidalPositionalEmbedding, TimestepEmbedding
from .resnet_blocks import SpatioTemporalResnetBlock, Downsample3D, Upsample3D
from .temporal_attention import TemporalSelfAttention, TemporalCrossAttention
from .hf_unet_wrapper import HuggingFaceVideoUNet


class SpatioTemporalUNet(nn.Module if HAS_TORCH else object):
    """End-to-End Spatio-Temporal Video Diffusion UNet.

    Accepts 5D video input: (Batch, Channels, Time, Height, Width).
    Conditions on:
        - Timesteps: (Batch,)
        - Multimodal conditioning: (Batch, 4, Time, Height, Width) [1ch Depth + 3ch Normal]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        cond_channels: int = 4,
        model_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        temporal_conv_kernel_size: int = 3,
        use_temporal_attention: bool = True,
        dropout: float = 0.1,
        use_hf_backbone: bool = False
    ):
        if not HAS_TORCH:
            return
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.model_channels = model_channels
        self.use_hf_backbone = use_hf_backbone

        if use_hf_backbone:
            self.hf_unet = HuggingFaceVideoUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                cond_channels=cond_channels,
                model_channels=model_channels,
                channel_mult=channel_mult,
                num_res_blocks=num_res_blocks,
                use_temporal_attention=use_temporal_attention
            )
            return

        time_embed_dim = model_channels * 4
        self.time_proj = SinusoidalPositionalEmbedding(dim=model_channels)
        self.time_embedding = TimestepEmbedding(model_channels, time_embed_dim)

        # Initial 3D Conv projecting (RGB + Condition) to model_channels
        total_in = in_channels + cond_channels
        self.conv_in = nn.Conv3d(total_in, model_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # Downsample levels
        self.down_blocks = nn.ModuleList()
        curr_channels = model_channels

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            res_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                res_blocks.append(
                    SpatioTemporalResnetBlock(
                        in_channels=curr_channels,
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        temporal_kernel_size=temporal_conv_kernel_size,
                        use_temporal_attention=use_temporal_attention
                    )
                )
                curr_channels = out_ch
            downsample = Downsample3D(curr_channels) if level < len(channel_mult) - 1 else nn.Identity()
            self.down_blocks.append(nn.ModuleDict({"blocks": res_blocks, "downsample": downsample}))

        # Middle bottleneck
        self.mid_block1 = SpatioTemporalResnetBlock(
            in_channels=curr_channels,
            out_channels=curr_channels,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            temporal_kernel_size=temporal_conv_kernel_size,
            use_temporal_attention=use_temporal_attention
        )
        self.mid_temporal_attn = TemporalSelfAttention(channels=curr_channels)
        self.mid_block2 = SpatioTemporalResnetBlock(
            in_channels=curr_channels,
            out_channels=curr_channels,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            temporal_kernel_size=temporal_conv_kernel_size,
            use_temporal_attention=use_temporal_attention
        )

        # Upsample levels
        self.up_blocks = nn.ModuleList()
        reversed_mult = list(reversed(channel_mult))

        for level, mult in enumerate(reversed_mult):
            out_ch = model_channels * mult
            res_blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                # Skip connection doubles channels
                in_ch = curr_channels + out_ch
                res_blocks.append(
                    SpatioTemporalResnetBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        temporal_kernel_size=temporal_conv_kernel_size,
                        use_temporal_attention=use_temporal_attention
                    )
                )
                curr_channels = out_ch
            upsample = Upsample3D(curr_channels) if level < len(channel_mult) - 1 else nn.Identity()
            self.up_blocks.append(nn.ModuleDict({"blocks": res_blocks, "upsample": upsample}))

        # Final projection to output RGB channels
        self.final_norm = nn.GroupNorm(min(32, curr_channels), curr_channels)
        self.final_act = nn.SiLU()
        self.conv_out = nn.Conv3d(curr_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(
        self,
        x: "torch.Tensor",
        timesteps: "torch.Tensor",
        condition: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        """Args:
            x: Video tensor (B, 3, T, H, W)
            timesteps: 1D Tensor (B,)
            condition: Multimodal condition (B, 4, T, H, W)

        Returns:
            Noise prediction epsilon_theta of shape (B, 3, T, H, W)
        """
        if self.use_hf_backbone:
            return self.hf_unet(x, timesteps, condition=condition)

        b, c, t, h, w = x.shape

        if condition is None:
            condition = torch.zeros((b, self.cond_channels, t, h, w), device=x.device, dtype=x.dtype)

        # Concatenate inputs along channel dimension
        h_feat = torch.cat([x, condition], dim=1)
        h_feat = self.conv_in(h_feat)

        # Timestep embeddings
        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)

        skips = []

        # Downsample
        for down_level in self.down_blocks:
            for block in down_level["blocks"]:
                h_feat = block(h_feat, time_emb=t_emb)
                skips.append(h_feat)
            h_feat = down_level["downsample"](h_feat)

        # Mid block
        h_feat = self.mid_block1(h_feat, time_emb=t_emb)
        h_feat = self.mid_temporal_attn(h_feat)
        h_feat = self.mid_block2(h_feat, time_emb=t_emb)

        # Upsample
        for up_level in self.up_blocks:
            for block in up_level["blocks"]:
                skip = skips.pop()
                h_feat = torch.cat([h_feat, skip], dim=1)
                h_feat = block(h_feat, time_emb=t_emb)
            h_feat = up_level["upsample"](h_feat)

        # Final output
        out = self.conv_out(self.final_act(self.final_norm(h_feat)))
        return out

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        if not HAS_TORCH:
            return 0
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

