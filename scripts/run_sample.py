"""Inference and Sampling Script for Generative Video Diffusion Model.

Generates video sequences from noise conditioned on 2D projections, Depth maps,
and Surface Normal maps using DDPM ancestral sampling or fast DDIM.
Developed in collaboration with researchers from Google DeepMind and ASU.
"""

import os
import sys
import yaml
import argparse
from typing import Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models.spatio_temporal_unet import SpatioTemporalUNet
from src.diffusion.ddpm import VideoDDPM
from src.diffusion.ddim import VideoDDIMSampler
from src.diffusion.schedules import DiffusionSchedule
from src.data.video_dataset import VideoMultimodalDataset
from src.utils.video_io import save_video_as_gif, create_multimodal_comparison_grid


def main():
    parser = argparse.ArgumentParser(description="Sample / Generate Video from Diffusion Model")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--sampler", type=str, choices=["ddim", "ddpm"], default="ddim", help="Sampling method")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of sampling steps for DDIM")
    parser.add_argument("--cfg_scale", type=float, default=3.5, help="Classifier-Free Guidance scale")
    parser.add_argument("--output_path", type=str, default="./outputs/generated_video.gif", help="Output GIF path")
    parser.add_argument("--test_scene", type=str, default=None, help="Path to scene folder with depth and normal maps")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    print("==================================================================")
    print("      VIDEO DIFFUSION INFERENCE & MULTIMODAL SYNTHESIS ENGINE     ")
    print("  In collaboration with researchers from Google DeepMind and ASU  ")
    print("==================================================================")

    if not HAS_TORCH:
        print("[ERROR] PyTorch is required to execute video diffusion inference.")
        print("Please install PyTorch in your Python environment: pip install torch diffusers")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    model = SpatioTemporalUNet(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 3),
        cond_channels=model_cfg.get("cond_channels", 4),
        model_channels=model_cfg.get("model_channels", 64),
        channel_mult=tuple(model_cfg.get("channel_mult", [1, 2, 4])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        temporal_conv_kernel_size=model_cfg.get("temporal_conv_kernel_size", 3),
        use_temporal_attention=model_cfg.get("temporal_attention", True),
        dropout=0.0
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading weights from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    else:
        print("[NOTICE] No checkpoint specified or file not found. Running with initialized weights for demonstration.")

    model.eval()

    num_frames = config["dataset"].get("num_frames", 16)
    height, width = config["dataset"].get("resolution", [64, 64])
    video_shape = (1, 3, num_frames, height, width)

    # Condition setup
    condition = None
    depth_cond = None
    normal_cond = None
    gt_rgb = None

    if args.test_scene and os.path.exists(args.test_scene):
        print(f"Loading condition maps from scene: {args.test_scene}")
        dataset = VideoMultimodalDataset(
            data_dir=os.path.dirname(args.test_scene),
            num_frames=num_frames,
            resolution=(height, width)
        )
        sample = dataset[0]
        condition = sample["condition"].unsqueeze(0).to(device) # (1, 4, T, H, W)
        depth_cond = sample["depth"].unsqueeze(0)
        normal_cond = sample["normal"].unsqueeze(0)
        gt_rgb = sample["rgb"].unsqueeze(0)
    else:
        print("[INFO] Generating synthetic conditioning signals (depth + normal)...")
        # Synthetic placeholder depth + normal condition
        depth_cond = torch.linspace(0.2, 0.8, num_frames).view(1, 1, num_frames, 1, 1).repeat(1, 1, 1, height, width)
        normal_cond = torch.zeros((1, 3, num_frames, height, width))
        normal_cond[:, 2, :, :, :] = 1.0 # Facing forward normal
        condition = torch.cat([depth_cond, normal_cond], dim=1).to(device)

    print(f"Generating video clip ({num_frames} frames, {height}x{width}) via {args.sampler.upper()}...")

    if args.sampler == "ddim":
        diff_schedule = DiffusionSchedule(
            timesteps=config["diffusion"].get("timesteps", 1000),
            schedule_type=config["diffusion"].get("beta_schedule", "cosine"),
            device=device
        )
        diff_schedule.to_torch(device)
        sampler = VideoDDIMSampler(
            diffusion_schedule=diff_schedule,
            model=model,
            ddim_steps=args.ddim_steps,
            eta=config["sampling"].get("ddim_eta", 0.0),
            device=device
        )
        generated_video = sampler.sample(
            shape=video_shape,
            condition=condition,
            cfg_scale=args.cfg_scale
        )
    else:
        ddpm = VideoDDPM(
            model=model,
            timesteps=config["diffusion"].get("timesteps", 1000),
            beta_schedule=config["diffusion"].get("beta_schedule", "cosine"),
            device=device
        )
        generated_video = ddpm.sample_loop(
            shape=video_shape,
            condition=condition,
            cfg_scale=args.cfg_scale
        )

    # Build side-by-side comparison grid
    grid = create_multimodal_comparison_grid(
        condition_depth=depth_cond,
        condition_normal=normal_cond,
        generated_rgb=generated_video,
        ground_truth_rgb=gt_rgb
    )

    save_video_as_gif(grid, output_path=args.output_path, fps=config["sampling"].get("fps", 8))
    print(f"Video synthesis complete! Saved comparison grid to: {args.output_path}")


if __name__ == "__main__":
    main()

