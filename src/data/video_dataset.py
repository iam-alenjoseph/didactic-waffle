"""Multimodal Video Dataset for Generative Diffusion Models.

Loads paired video clips with:
    - 2D RGB Projections: (3, T, H, W)
    - Depth Maps: (1, T, H, W)
    - Surface Normal Maps: (3, T, H, W)
    - Camera Metadata & Intrinsics
"""

import os
import glob
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image

try:
    from torch.utils.data import Dataset
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object

from .transforms import VideoTransform


class VideoMultimodalDataset(Dataset):
    """Dataset loader for Unreal Engine and synthetic multi-modal video sequences."""

    def __init__(
        self,
        data_dir: str,
        num_frames: int = 16,
        frame_stride: int = 1,
        resolution: Tuple[int, int] = (64, 64),
        horizontal_flip_prob: float = 0.5,
        temporal_reverse_prob: float = 0.0
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.resolution = resolution

        self.transform = VideoTransform(
            target_resolution=resolution,
            horizontal_flip_prob=horizontal_flip_prob,
            temporal_reverse_prob=temporal_reverse_prob
        )

        self.scenes = self._discover_scenes()

    def _discover_scenes(self) -> List[str]:
        """Discovers scene subdirectories containing rgb, depth, and normals folders."""
        if not os.path.exists(self.data_dir):
            return []

        scene_paths = []
        for entry in sorted(os.listdir(self.data_dir)):
            full_path = os.path.join(self.data_dir, entry)
            if os.path.isdir(full_path):
                rgb_dir = os.path.join(full_path, "rgb")
                depth_dir = os.path.join(full_path, "depth")
                normals_dir = os.path.join(full_path, "normals")
                if os.path.exists(rgb_dir) and os.path.exists(depth_dir) and os.path.exists(normals_dir):
                    scene_paths.append(full_path)

        return scene_paths

    def __len__(self) -> int:
        return len(self.scenes)

    def _load_image_sequence(self, folder: str, filenames: List[str], is_grayscale: bool = False) -> np.ndarray:
        """Loads and resizes an ordered sequence of images into a numpy array (T, H, W, C)."""
        frames = []
        for fn in filenames:
            path = os.path.join(folder, fn)
            img = Image.open(path)
            if is_grayscale:
                img = img.convert("L")
            else:
                img = img.convert("RGB")

            # Resize to target resolution if needed
            if img.size != (self.resolution[1], self.resolution[0]):
                img = img.resize((self.resolution[1], self.resolution[0]), Image.BILINEAR)

            arr = np.array(img)
            frames.append(arr)

        arr_stacked = np.stack(frames, axis=0) # (T, H, W, C) or (T, H, W)
        return arr_stacked

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene_path = self.scenes[idx]
        rgb_dir = os.path.join(scene_path, "rgb")
        depth_dir = os.path.join(scene_path, "depth")
        normals_dir = os.path.join(scene_path, "normals")

        rgb_files = sorted(os.listdir(rgb_dir))
        depth_files = sorted(os.listdir(depth_dir))
        normals_files = sorted(os.listdir(normals_dir))

        total_avail = len(rgb_files)
        required_span = self.num_frames * self.frame_stride

        # Slice frame sequence
        if total_avail >= required_span:
            start_idx = 0
            selected_indices = list(range(start_idx, start_idx + required_span, self.frame_stride))
        else:
            # Repeat or pad frames if sequence is shorter than required
            indices = list(range(total_avail))
            while len(indices) < self.num_frames:
                indices = indices + indices
            selected_indices = indices[:self.num_frames]

        sel_rgb_files = [rgb_files[i % total_avail] for i in selected_indices]
        sel_depth_files = [depth_files[i % total_avail] for i in selected_indices]
        sel_normals_files = [normals_files[i % total_avail] for i in selected_indices]

        # Load raw image sequences
        rgb_raw = self._load_image_sequence(rgb_dir, sel_rgb_files, is_grayscale=False)
        depth_raw = self._load_image_sequence(depth_dir, sel_depth_files, is_grayscale=True)
        normals_raw = self._load_image_sequence(normals_dir, sel_normals_files, is_grayscale=False)

        # Apply transforms and normalization to [-1, 1]
        rgb_tensor, depth_tensor, normal_tensor = self.transform(
            rgb_frames=rgb_raw,
            depth_frames=depth_raw,
            normal_frames=normals_raw
        )

        # Concatenate Depth (1) + Normal (3) -> Multimodal condition (4, T, H, W)
        condition_tensor = np.concatenate([depth_tensor, normal_tensor], axis=0)

        # Load metadata if present
        meta_path = os.path.join(scene_path, "metadata.json")
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                try:
                    metadata = json.load(f)
                except Exception:
                    pass

        sample = {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "normal": normal_tensor,
            "condition": condition_tensor,
            "scene_path": scene_path,
            "metadata": metadata
        }

        # If PyTorch is used, convert to torch.Tensor
        if HAS_TORCH:
            sample["rgb"] = torch.from_numpy(sample["rgb"]).float()
            sample["depth"] = torch.from_numpy(sample["depth"]).float()
            sample["normal"] = torch.from_numpy(sample["normal"]).float()
            sample["condition"] = torch.from_numpy(sample["condition"]).float()

        return sample

