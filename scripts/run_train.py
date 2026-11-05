"""Training Script for Generative Video Diffusion Model (DDPM).

Trains Spatio-Temporal Video UNet with multimodal conditioning (depth + normal maps)
using Hugging Face UNet architectural paradigms and cosine noise schedules.
Developed in collaboration with researchers from Google DeepMind and ASU.
"""

import os
import sys
import time
import yaml
import argparse
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models.spatio_temporal_unet import SpatioTemporalUNet
from src.diffusion.ddpm import VideoDDPM
from src.diffusion.losses import VideoDiffusionLoss
from src.diffusion.conditioning import ClassifierFreeGuidanceHelper
from src.data.video_dataset import VideoMultimodalDataset
from src.utils.video_io import save_video_as_gif, create_multimodal_comparison_grid
from src.utils.logger import setup_logger, MetricTracker


def train_epoch(
    model: Any,
    ddpm: VideoDDPM,
    dataloader: Any,
    optimizer: Any,
    criterion: VideoDiffusionLoss,
    cfg_helper: ClassifierFreeGuidanceHelper,
    device: str,
    metric_tracker: MetricTracker,
    grad_clip: float = 1.0
):
    model.train()

    for step, batch in enumerate(dataloader):
        rgb = batch["rgb"].to(device) # (B, 3, T, H, W)
        condition = batch["condition"].to(device) # (B, 4, T, H, W) [1ch depth + 3ch normal]

        b, c, t, h, w = rgb.shape

        # Sample uniform discrete timesteps t in [0, T-1]
        timesteps = torch.randint(0, ddpm.timesteps, (b,), device=device).long()

        # Sample Gaussian noise epsilon ~ N(0, I)
        noise = torch.randn_like(rgb)

        # Forward diffusion q(x_t | x_0)
        x_noisy = ddpm.q_sample(x_start=rgb, t=timesteps, noise=noise)

        # Apply random condition dropout for Classifier-Free Guidance (CFG)
        cond_masked = cfg_helper.mask_condition_for_training(condition)

        # Model forward pass: predict noise epsilon_theta
        optimizer.zero_grad()
        pred_noise = model(x_noisy, timesteps, condition=cond_masked)

        # Calculate loss with Min-SNR weighting and temporal consistency
        snr_tensor = ddpm.torch_constants["snr"]
        loss_dict = criterion(
            pred_noise=pred_noise,
            target_noise=noise,
            timesteps=timesteps,
            snr=snr_tensor
        )

        loss = loss_dict["total_loss"]
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        metric_tracker.update("loss", loss.item(), n=b)
        metric_tracker.update("base_loss", loss_dict["base_loss"].item(), n=b)


def main():
    parser = argparse.ArgumentParser(description="Train Video Diffusion Model")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory override")
    parser.add_argument("--output_dir", type=str, default="./experiments/run_001", help="Experiment outputs")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda/cpu)")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output_dir=output_dir, name="TrainVideoDDPM")

    logger.info("==================================================================")
    logger.info("       GENERATIVE DIFFUSION MODEL FOR VIDEO TRAINING PIPELINE      ")
    logger.info("  In collaboration with researchers from Google DeepMind and ASU  ")
    logger.info("==================================================================")

    if not HAS_TORCH:
        logger.error("PyTorch is not available in the current environment.")
        logger.error("To run training, please ensure PyTorch and Diffusers are installed:")
        logger.error("    pip install torch torchvision diffusers accelerate")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Compute Device: {device}")

    # Dataset directory resolution
    data_dir = args.data_dir or config["dataset"].get("data_dir", "./data/synthetic_dataset")
    if not os.path.exists(data_dir):
        # Fall back to synthetic data
        synth_dir = config["dataset"].get("synthetic_dir", "./data/synthetic_dataset")
        if os.path.exists(synth_dir):
            data_dir = synth_dir
        else:
            logger.info(f"Dataset path {data_dir} not found. Generating synthetic dataset automatically...")
            from src.data.synthetic_generator import SyntheticVideoDatasetGenerator
            gen = SyntheticVideoDatasetGenerator(
                output_dir=synth_dir,
                num_scenes=10,
                num_frames=config["dataset"].get("num_frames", 16),
                resolution=tuple(config["dataset"].get("resolution", [64, 64]))
            )
            gen.generate_all()
            data_dir = synth_dir

    logger.info(f"Loading multimodal video dataset from: {data_dir}")
    dataset = VideoMultimodalDataset(
        data_dir=data_dir,
        num_frames=config["dataset"].get("num_frames", 16),
        frame_stride=config["dataset"].get("frame_stride", 1),
        resolution=tuple(config["dataset"].get("resolution", [64, 64]))
    )
    logger.info(f"Loaded {len(dataset)} video scenes.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Initialize Spatio-Temporal Video UNet
    model_cfg = config.get("model", {})
    logger.info("Constructing Spatio-Temporal Video UNet architecture...")
    model = SpatioTemporalUNet(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 3),
        cond_channels=model_cfg.get("cond_channels", 4),
        model_channels=model_cfg.get("model_channels", 64),
        channel_mult=tuple(model_cfg.get("channel_mult", [1, 2, 4])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        temporal_conv_kernel_size=model_cfg.get("temporal_conv_kernel_size", 3),
        use_temporal_attention=model_cfg.get("temporal_attention", True),
        dropout=model_cfg.get("dropout", 0.1),
        use_hf_backbone=model_cfg.get("use_hf_unet_backbone", False)
    ).to(device)

    total_params = model.get_num_parameters()
    logger.info(f"Model successfully built. Total parameters: {total_params:,}")

    # Diffusion Engine
    diff_cfg = config.get("diffusion", {})
    ddpm = VideoDDPM(
        model=model,
        timesteps=diff_cfg.get("timesteps", 1000),
        beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
        beta_start=diff_cfg.get("beta_start", 0.0001),
        beta_end=diff_cfg.get("beta_end", 0.02),
        prediction_type=diff_cfg.get("prediction_type", "epsilon"),
        device=device
    )

    criterion = VideoDiffusionLoss(
        loss_type=diff_cfg.get("loss_type", "mse"),
        min_snr_gamma=diff_cfg.get("min_snr_gamma", 5.0),
        temporal_loss_weight=diff_cfg.get("temporal_loss_weight", 0.1)
    )

    cfg_helper = ClassifierFreeGuidanceHelper(uncond_prob=0.1)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(config["training"].get("adam_beta1", 0.9), config["training"].get("adam_beta2", 0.999)),
        weight_decay=config["training"].get("adam_weight_decay", 1e-4)
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    tracker = MetricTracker()

    logger.info(f"Commencing training across {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        tracker.reset()
        t0 = time.time()

        train_epoch(
            model=model,
            ddpm=ddpm,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            cfg_helper=cfg_helper,
            device=device,
            metric_tracker=tracker
        )

        scheduler.step()
        epoch_time = time.time() - t0
        curr_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"Loss: {tracker.get_average('loss'):.4f} "
            f"(Base: {tracker.get_average('base_loss'):.4f}) | "
            f"LR: {curr_lr:.2e} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Save checkpoint periodically
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    total_time = time.time() - start_time
    logger.info(f"Training successfully completed in {total_time:.2f}s.")


if __name__ == "__main__":
    main()

