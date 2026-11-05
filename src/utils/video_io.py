"""Video Input/Output and Visualization Utilities.

Saves video tensors to GIF, MP4, and multi-modal comparison grids
(Condition Depth + Normal | Ground Truth | Diffusion Generated Video).
"""

import os
from typing import Union, List, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def tensor_to_numpy_video(video: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """Converts 4D/5D video tensor in [-1, 1] or [0, 1] to uint8 (T, H, W, C) in [0, 255]."""
    if HAS_TORCH and isinstance(video, torch.Tensor):
        arr = video.detach().cpu().float().numpy()
    else:
        arr = np.array(video)

    # If 5D (B, C, T, H, W), take first batch item
    if arr.ndim == 5:
        arr = arr[0]

    # If (C, T, H, W), transpose to (T, H, W, C)
    if arr.ndim == 4 and arr.shape[0] in [1, 3, 4]:
        arr = np.transpose(arr, (1, 2, 3, 0))

    # If 1-channel (grayscale/depth), repeat across 3 RGB channels
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # Normalize to [0, 255]
    if arr.min() < -0.05: # Assume [-1, 1] range
        arr = (arr + 1.0) * 127.5
    elif arr.max() <= 1.05: # Assume [0, 1] range
        arr = arr * 255.0

    return np.clip(arr, 0, 255).astype(np.uint8)


def save_video_as_gif(
    video: Union[np.ndarray, "torch.Tensor"],
    output_path: str,
    fps: int = 10,
    loop: int = 0
):
    """Saves a video tensor as an animated GIF."""
    frames_np = tensor_to_numpy_video(video) # (T, H, W, 3)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    pil_images = [Image.fromarray(frame) for frame in frames_np]
    duration_ms = int(1000.0 / fps)

    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True
    )


def create_multimodal_comparison_grid(
    condition_depth: Union[np.ndarray, "torch.Tensor"],
    condition_normal: Union[np.ndarray, "torch.Tensor"],
    generated_rgb: Union[np.ndarray, "torch.Tensor"],
    ground_truth_rgb: Optional[Union[np.ndarray, "torch.Tensor"]] = None
) -> np.ndarray:
    """Concatenates video streams horizontally into a comparison grid:
        [Depth Map | Normal Map | Generated Video | Ground Truth (optional)]

    Returns:
        Video array of shape (T, H, W * num_cols, 3) ready for GIF or MP4 export.
    """
    v_depth = tensor_to_numpy_video(condition_depth)
    v_normal = tensor_to_numpy_video(condition_normal)
    v_gen = tensor_to_numpy_video(generated_rgb)

    streams = [v_depth, v_normal, v_gen]

    if ground_truth_rgb is not None:
        v_gt = tensor_to_numpy_video(ground_truth_rgb)
        streams.append(v_gt)

    # Determine minimum frame length
    min_t = min(s.shape[0] for s in streams)
    grid_frames = []

    for t in range(min_t):
        frame_row = np.concatenate([s[t] for s in streams], axis=1) # Horizontal stack
        grid_frames.append(frame_row)

    return np.stack(grid_frames, axis=0)

