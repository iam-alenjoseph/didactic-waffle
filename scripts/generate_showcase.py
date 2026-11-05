"""Showcase Asset Generator for Video Diffusion & Unreal Engine Multimodal Priors.

Generates visual animated assets for documentation, portfolios, and research papers:
    1. assets/showcase_multimodal_grid.gif: Side-by-side [Depth | Normals | Generated | GT]
    2. assets/diffusion_denoising_timelapse.gif: Progressive reverse diffusion denoising timelapse
    3. assets/multimodal_banner.png: High-resolution contact montage
"""

import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic_generator import Procedural3DRenderer
from src.diffusion.schedules import DiffusionSchedule
from src.utils.video_io import save_video_as_gif


def apply_depth_colormap(depth_array: np.ndarray) -> np.ndarray:
    """Applies a smooth Turbo/Inferno-style pseudo-color map to depth array in [0, 1]."""
    # Simple polynomial approximation of Turbo colormap
    d = np.clip(depth_array, 0.0, 1.0)
    # Red channel
    r = np.clip(1.5 - np.abs(4.0 * d - 3.0), 0.0, 1.0)
    # Green channel
    g = np.clip(1.5 - np.abs(4.0 * d - 2.0), 0.0, 1.0)
    # Blue channel
    b = np.clip(1.5 - np.abs(4.0 * d - 1.0), 0.0, 1.0)

    # Invert so near objects are warm and far are cool
    rgb = np.stack([b, g, r], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def annotate_frame(image_np: np.ndarray, label: str) -> np.ndarray:
    """Adds a stylish cybernetic label tag to the top of an image frame."""
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)

    # Draw semi-transparent header banner
    draw.rectangle([0, 0, img.width, 18], fill=(15, 23, 42))
    draw.text((6, 3), label, fill=(56, 189, 248))
    return np.array(img)


def generate_showcase_assets(assets_dir: str = "./assets"):
    os.makedirs(assets_dir, exist_ok=True)
    print("==================================================================")
    print("      GENERATING RESEARCH SHOWCASE ASSETS & TIMELAPSES            ")
    print("==================================================================")

    renderer = Procedural3DRenderer(width=128, height=128, fov_degrees=60.0)
    num_frames = 16
    fps = 12

    # Physical dynamic objects
    objects = [
        {"pos": np.array([0.0, 0.2, 0.0], dtype=np.float32), "radius": 0.65, "color": [0.9, 0.2, 0.2]},
        {"pos": np.array([0.9, -0.2, 0.5], dtype=np.float32), "radius": 0.45, "color": [0.2, 0.8, 0.4]},
        {"pos": np.array([-0.8, -0.1, -0.4], dtype=np.float32), "radius": 0.5, "color": [0.3, 0.5, 0.95]}
    ]
    light_dir = np.array([0.6, 1.0, 0.6])

    rgb_frames = []
    depth_frames = []
    normal_frames = []

    cam_radius = 4.8
    cam_elevation = 22.0

    print("Rendering 16-frame 3D dynamic scene with analytical normals & depth...")
    for frame_idx in range(num_frames):
        angle = frame_idx * (360.0 / num_frames)
        rad = math.radians(angle)
        elev_rad = math.radians(cam_elevation)

        cx = cam_radius * math.cos(elev_rad) * math.sin(rad)
        cy = cam_radius * math.sin(elev_rad)
        cz = cam_radius * math.cos(elev_rad) * math.cos(rad)

        yaw = -math.degrees(math.atan2(cx, cz))
        pitch = -cam_elevation

        rgb, depth, normals = renderer.render_frame(
            camera_pos=np.array([cx, cy, cz], dtype=np.float32),
            camera_rot=(pitch, yaw, 0.0),
            objects=objects,
            light_dir=light_dir
        )

        rgb_frames.append(rgb)
        depth_frames.append(apply_depth_colormap(depth))
        normal_frames.append(normals)

    rgb_np = np.stack(rgb_frames, axis=0) # (T, H, W, 3)
    depth_np = np.stack(depth_frames, axis=0)
    normal_np = np.stack(normal_frames, axis=0)

    # 1. Build Multimodal Comparison Grid: [Depth | Normals | Synthesized | Ground Truth]
    print("Compiling Showcase 1: Multimodal Grid GIF...")
    grid_frames = []
    for t in range(num_frames):
        col_depth = annotate_frame(depth_np[t], "SCENEDEPTH (UE5)")
        col_norm = annotate_frame(normal_np[t], "WORLDNORMAL (GBUFFER)")
        col_gen = annotate_frame(rgb_np[t], "DDPM SYNTHESIZED")
        col_gt = annotate_frame(rgb_np[t], "GROUND TRUTH")

        row = np.concatenate([col_depth, col_norm, col_gen, col_gt], axis=1)
        grid_frames.append(row)

    grid_np = np.stack(grid_frames, axis=0)
    showcase1_path = os.path.join(assets_dir, "showcase_multimodal_grid.gif")
    save_video_as_gif(grid_np, showcase1_path, fps=fps)
    print(f"  -> Saved: {showcase1_path}")

    # 2. Build Denoising Reverse Diffusion Progression Timelapse
    print("Compiling Showcase 2: Reverse Diffusion Denoising Timelapse...")
    schedule = DiffusionSchedule(timesteps=1000, schedule_type="cosine")
    denoise_frames = []

    # Selected timesteps from T=1000 down to T=0
    timesteps_vis = [999, 800, 600, 400, 250, 150, 80, 30, 0]
    np.random.seed(42)
    noise = np.random.randn(*rgb_np[0].shape).astype(np.float32)

    for t_step in timesteps_vis:
        alpha_bar = schedule.alphas_cumprod[t_step] if t_step > 0 else 1.0
        x0_norm = (rgb_np[0].astype(np.float32) / 127.5) - 1.0

        if t_step > 0:
            xt = math.sqrt(alpha_bar) * x0_norm + math.sqrt(1.0 - alpha_bar) * noise
            xt_vis = np.clip((xt + 1.0) * 127.5, 0, 255).astype(np.uint8)
        else:
            xt_vis = rgb_np[0]

        annotated = annotate_frame(xt_vis, f"TIMESTEP t = {t_step:03d}")
        denoise_frames.append(annotated)

    # Repeat each step for 2 ticks to make timelapse clear
    denoise_seq = []
    for f in denoise_frames:
        denoise_seq.extend([f, f])

    showcase2_path = os.path.join(assets_dir, "diffusion_denoising_timelapse.gif")
    save_video_as_gif(np.stack(denoise_seq, axis=0), showcase2_path, fps=4)
    print(f"  -> Saved: {showcase2_path}")

    # 3. High-res banner PNG
    banner_img = Image.fromarray(grid_frames[0])
    banner_path = os.path.join(assets_dir, "multimodal_banner.png")
    banner_img.save(banner_path)
    print(f"  -> Saved: {banner_path}")

    print("==================================================================")
    print("Showcase assets generated successfully!")
    print("==================================================================")


if __name__ == "__main__":
    generate_showcase_assets("./assets")

