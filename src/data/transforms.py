"""Spatio-Temporal Transforms and Augmentations for Multimodal Video Data."""

import random
from typing import Tuple, Optional, Union
import numpy as np

try:
    import torch
    import torchvision.transforms.functional as TF
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def normalize_to_neg_one_to_one(img: np.ndarray) -> np.ndarray:
    """Normalizes image array from [0, 255] or [0, 1] to [-1, 1]."""
    if img.dtype == np.uint8:
        return (img.astype(np.float32) / 127.5) - 1.0
    return (img * 2.0) - 1.0


def unnormalize_to_zero_to_one(tensor: Union[np.ndarray, "torch.Tensor"]) -> Union[np.ndarray, "torch.Tensor"]:
    """Reverts [-1, 1] normalized tensor to [0, 1]."""
    return (tensor + 1.0) / 2.0


class VideoTransform:
    """Joint spatio-temporal transform for multimodal video tuples: (RGB, Depth, Normal).

    Ensures that spatial transformations (horizontal flips, crops) are applied
    identically across time frames and across all modal streams (RGB, depth, normal).
    """

    def __init__(
        self,
        target_resolution: Tuple[int, int] = (64, 64),
        horizontal_flip_prob: float = 0.5,
        temporal_reverse_prob: float = 0.0
    ):
        self.target_resolution = target_resolution
        self.horizontal_flip_prob = horizontal_flip_prob
        self.temporal_reverse_prob = temporal_reverse_prob

    def __call__(
        self,
        rgb_frames: np.ndarray,      # (T, H, W, 3)
        depth_frames: np.ndarray,    # (T, H, W, 1) or (T, H, W)
        normal_frames: np.ndarray    # (T, H, W, 3)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Applies consistent random augmentations across time and modalities."""
        T, H, W, C = rgb_frames.shape

        # Random horizontal flip
        if random.random() < self.horizontal_flip_prob:
            rgb_frames = np.flip(rgb_frames, axis=2)
            if depth_frames.ndim == 4:
                depth_frames = np.flip(depth_frames, axis=2)
            else:
                depth_frames = np.flip(depth_frames, axis=1)
            normal_frames = np.flip(normal_frames, axis=2)
            # Invert X-axis of normals after horizontal flip
            normal_frames = normal_frames.copy()
            if normal_frames.dtype == np.uint8:
                normal_frames[..., 0] = 255 - normal_frames[..., 0]
            else:
                normal_frames[..., 0] = -normal_frames[..., 0]

        # Random temporal reversal (for symmetric physical motions)
        if random.random() < self.temporal_reverse_prob:
            rgb_frames = np.flip(rgb_frames, axis=0)
            depth_frames = np.flip(depth_frames, axis=0)
            normal_frames = np.flip(normal_frames, axis=0)

        # Normalize to [-1, 1]
        rgb_norm = normalize_to_neg_one_to_one(rgb_frames)
        normal_norm = normalize_to_neg_one_to_one(normal_frames)

        if depth_frames.ndim == 3:
            depth_frames = depth_frames[..., None]
        depth_norm = (depth_frames.astype(np.float32) / 255.0) if depth_frames.dtype == np.uint8 else depth_frames
        depth_norm = (depth_norm * 2.0) - 1.0 # to [-1, 1]

        # Transpose from (T, H, W, C) to (C, T, H, W) for PyTorch 3D Convolutions
        rgb_out = np.transpose(rgb_norm, (3, 0, 1, 2))
        depth_out = np.transpose(depth_norm, (3, 0, 1, 2))
        normal_out = np.transpose(normal_norm, (3, 0, 1, 2))

        return (
            np.ascontiguousarray(rgb_out, dtype=np.float32),
            np.ascontiguousarray(depth_out, dtype=np.float32),
            np.ascontiguousarray(normal_out, dtype=np.float32)
        )

