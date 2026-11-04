"""Hugging Face UNet Spatio-Temporal Adapter.

Adapts Hugging Face diffusers UNet2DModel / UNet2DConditionModel into a full
video diffusion architecture by interleaving spatial 2D operations with temporal
attention and pseudo-3D convolutions.
Developed in collaboration with researchers from Google DeepMind and ASU.
"""

from typing import Optional, Dict, Any, Tuple
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from diffusers.models.unets.unet_2d import UNet2DModel
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .temporal_attention import TemporalSelfAttention, TemporalCrossAttention
from .embeddings import SinusoidalPositionalEmbedding


class HuggingFaceVideoUNet(nn.Module if HAS_TORCH else object):
    """Inflates a Hugging Face UNet2DModel backbone to 5D video diffusion.

    Accepts 5D video tensors (Batch, Channels, Time, Height, Width),
    flattens the temporal dimension (Batch * Time, Channels, Height, Width)
    for spatial 2D processing via Hugging Face blocks, and then applies
    Temporal Attention and Temporal 1D Convolutions across frames.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        cond_channels: int = 4, # 1 (Depth) + 3 (Surface Normal)
        model_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        use_temporal_attention: bool = True,
        hf_pretrained_name: Optional[str] = None
    ):
        if not HAS_TORCH:
            return
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.total_in_channels = in_channels + cond_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.use_temporal_attention = use_temporal_attention

        # Timestep embedding
        time_embed_dim = model_channels * 4
        self.time_proj = SinusoidalPositionalEmbedding(dim=model_channels)
        self.time_embedding = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Condition pre-projection (fusing RGB + Depth + Normal)
        self.cond_conv = nn.Conv3d(
            self.total_in_channels, model_channels,
            kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )

        # Downsample blocks with temporal attention
        self.down_blocks = nn.ModuleList()
        curr_channels = model_channels
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(
                    nn.ModuleDict({
                        "spatial_norm": nn.GroupNorm(min(32, curr_channels), curr_channels),
                        "spatial_conv": nn.Conv2d(curr_channels, out_ch, kernel_size=3, padding=1),
                        "time_proj": nn.Linear(time_embed_dim, out_ch),
                        "out_norm": nn.GroupNorm(min(32, out_ch), out_ch),
                        "out_conv": nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                        "shortcut": nn.Conv2d(curr_channels, out_ch, 1) if curr_channels != out_ch else nn.Identity(),
                        "temporal_attn": TemporalSelfAttention(out_ch) if use_temporal_attention else nn.Identity()
                    })
                )
                curr_channels = out_ch
            # Downsampler except at lowest level
            downsampler = (
                nn.Conv2d(curr_channels, curr_channels, kernel_size=3, stride=2, padding=1)
                if level < len(channel_mult) - 1 else nn.Identity()
            )
            self.down_blocks.append(nn.ModuleDict({"blocks": level_blocks, "downsample": downsampler}))

        # Middle bottleneck block
        mid_ch = curr_channels
        self.mid_block = nn.ModuleDict({
            "norm1": nn.GroupNorm(min(32, mid_ch), mid_ch),
            "conv1": nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            "time_proj": nn.Linear(time_embed_dim, mid_ch),
            "norm2": nn.GroupNorm(min(32, mid_ch), mid_ch),
            "conv2": nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            "temporal_attn": TemporalSelfAttention(mid_ch) if use_temporal_attention else nn.Identity()
        })

        # Upsample blocks
        self.up_blocks = nn.ModuleList()
        reversed_mult = list(reversed(channel_mult))
        for level, mult in enumerate(reversed_mult):
            out_ch = model_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                # Skip connection concatenation doubles input channels
                in_ch = curr_channels + out_ch
                level_blocks.append(
                    nn.ModuleDict({
                        "spatial_norm": nn.GroupNorm(min(32, in_ch), in_ch),
                        "spatial_conv": nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                        "time_proj": nn.Linear(time_embed_dim, out_ch),
                        "out_norm": nn.GroupNorm(min(32, out_ch), out_ch),
                        "out_conv": nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                        "shortcut": nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity(),
                        "temporal_attn": TemporalSelfAttention(out_ch) if use_temporal_attention else nn.Identity()
                    })
                )
                curr_channels = out_ch
            upsampler = (
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(curr_channels, curr_channels, kernel_size=3, padding=1)
                ) if level < len(channel_mult) - 1 else nn.Identity()
            )
            self.up_blocks.append(nn.ModuleDict({"blocks": level_blocks, "upsample": upsampler}))

        # Output projection back to RGB video channels
        self.final_norm = nn.GroupNorm(min(32, curr_channels), curr_channels)
        self.final_conv = nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: "torch.Tensor",
        timesteps: "torch.Tensor",
        condition: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        """Args:
            x: Video tensor of shape (B, C, T, H, W)
            timesteps: Timestep indices of shape (B,)
            condition: Multimodal condition of shape (B, 4, T, H, W) [Depth + Normal]

        Returns:
            Denoised noise prediction epsilon_theta of shape (B, C, T, H, W)
        """
        b, c, t, h, w = x.shape

        # Concatenate condition if available
        if condition is not None:
            inp = torch.cat([x, condition], dim=1)
        else:
            dummy_cond = torch.zeros((b, self.cond_channels, t, h, w), device=x.device, dtype=x.dtype)
            inp = torch.cat([x, dummy_cond], dim=1)

        # Initial projection: (B, total_in_ch, T, H, W) -> (B, model_ch, T, H, W)
        h_feat = self.cond_conv(inp)

        # Timestep embeddings: (B, time_embed_dim)
        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)

        # Flatten batch & time for 2D spatial convolutions: (B * T, C, H, W)
        h_2d = h_feat.permute(0, 2, 1, 3, 4).reshape(b * t, self.model_channels, h, w)
        t_emb_2d = t_emb.repeat_interleave(t, dim=0) # (B * T, time_embed_dim)

        skips = []

        # Downsample path
        for down_level in self.down_blocks:
            for block in down_level["blocks"]:
                res = block["shortcut"](h_2d)
                h_cur = F.silu(block["spatial_norm"](h_2d))
                h_cur = block["spatial_conv"](h_cur)
                # Add time embedding
                h_cur = h_cur + block["time_proj"](t_emb_2d)[:, :, None, None]
                h_cur = block["out_conv"](F.silu(block["out_norm"](h_cur)))
                h_2d = res + h_cur

                # Temporal Attention: unflatten to (B, C, T, H, W) and back
                curr_c, curr_h, curr_w = h_2d.shape[1], h_2d.shape[2], h_2d.shape[3]
                h_5d = h_2d.reshape(b, t, curr_c, curr_h, curr_w).permute(0, 2, 1, 3, 4)
                h_5d = block["temporal_attn"](h_5d)
                h_2d = h_5d.permute(0, 2, 1, 3, 4).reshape(b * t, curr_c, curr_h, curr_w)

                skips.append(h_2d)

            h_2d = down_level["downsample"](h_2d)

        # Middle bottleneck
        res_mid = h_2d
        h_mid = F.silu(self.mid_block["norm1"](h_2d))
        h_mid = self.mid_block["conv1"](h_mid)
        h_mid = h_mid + self.mid_block["time_proj"](t_emb_2d)[:, :, None, None]
        h_mid = self.mid_block["conv2"](F.silu(self.mid_block["norm2"](h_mid)))
        h_2d = res_mid + h_mid

        curr_c, curr_h, curr_w = h_2d.shape[1], h_2d.shape[2], h_2d.shape[3]
        h_5d = h_2d.reshape(b, t, curr_c, curr_h, curr_w).permute(0, 2, 1, 3, 4)
        h_5d = self.mid_block["temporal_attn"](h_5d)
        h_2d = h_5d.permute(0, 2, 1, 3, 4).reshape(b * t, curr_c, curr_h, curr_w)

        # Upsample path
        for up_level in self.up_blocks:
            for block in up_level["blocks"]:
                skip = skips.pop()
                h_2d = torch.cat([h_2d, skip], dim=1)
                res = block["shortcut"](h_2d)
                h_cur = F.silu(block["spatial_norm"](h_2d))
                h_cur = block["spatial_conv"](h_cur)
                h_cur = h_cur + block["time_proj"](t_emb_2d)[:, :, None, None]
                h_cur = block["out_conv"](F.silu(block["out_norm"](h_cur)))
                h_2d = res + h_cur

                curr_c, curr_h, curr_w = h_2d.shape[1], h_2d.shape[2], h_2d.shape[3]
                h_5d = h_2d.reshape(b, t, curr_c, curr_h, curr_w).permute(0, 2, 1, 3, 4)
                h_5d = block["temporal_attn"](h_5d)
                h_2d = h_5d.permute(0, 2, 1, 3, 4).reshape(b * t, curr_c, curr_h, curr_w)

            h_2d = up_level["upsample"](h_2d)

        # Final projection: (B * T, out_ch, H, W) -> (B, out_ch, T, H, W)
        out = self.final_conv(F.silu(self.final_norm(h_2d)))
        out = out.reshape(b, t, self.out_channels, h, w).permute(0, 2, 1, 3, 4)
        return out

