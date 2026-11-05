"""Evaluation Metrics for Video Diffusion Synthesis.

Implements:
1. PSNR (Peak Signal-to-Noise Ratio)
2. SSIM (Structural Similarity Index Measure)
3. Temporal Consistency Score (inter-frame gradient stability)
4. Fréchet Video Distance (FVD) calculation engine
"""

from typing import Union, Dict, Any, Tuple
import math
import numpy as np
from scipy import linalg
from skimage.metrics import structural_similarity as compute_skimage_ssim

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def to_numpy_frames(v: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """Converts input tensor to (T, H, W, C) float32 in range [0, 1]."""
    if HAS_TORCH and isinstance(v, torch.Tensor):
        arr = v.detach().cpu().float().numpy()
    else:
        arr = np.array(v, dtype=np.float32)

    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[0] in [1, 3]:
        arr = np.transpose(arr, (1, 2, 3, 0))

    if arr.min() < -0.05:
        arr = (arr + 1.0) / 2.0
    elif arr.max() > 1.5:
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0)


def calculate_psnr(
    gt: Union[np.ndarray, "torch.Tensor"],
    pred: Union[np.ndarray, "torch.Tensor"],
    max_val: float = 1.0
) -> float:
    """Calculates average PSNR across all frames in the video sequence."""
    gt_np = to_numpy_frames(gt)
    pred_np = to_numpy_frames(pred)

    mse = np.mean((gt_np - pred_np) ** 2)
    if mse == 0:
        return float("inf")

    psnr = 20.0 * math.log10(max_val / math.sqrt(mse))
    return float(psnr)


def calculate_ssim(
    gt: Union[np.ndarray, "torch.Tensor"],
    pred: Union[np.ndarray, "torch.Tensor"]
) -> float:
    """Calculates frame-averaged Structural Similarity Index (SSIM)."""
    gt_np = to_numpy_frames(gt)
    pred_np = to_numpy_frames(pred)

    num_frames = min(gt_np.shape[0], pred_np.shape[0])
    ssim_vals = []

    for t in range(num_frames):
        val = compute_skimage_ssim(
            gt_np[t],
            pred_np[t],
            channel_axis=-1,
            data_range=1.0
        )
        ssim_vals.append(val)

    return float(np.mean(ssim_vals))


def calculate_temporal_consistency(video: Union[np.ndarray, "torch.Tensor"]) -> float:
    """Measures temporal smoothness by calculating mean inter-frame pixel variation."""
    v_np = to_numpy_frames(video)
    if v_np.shape[0] <= 1:
        return 0.0

    # Inter-frame difference
    diff = v_np[1:] - v_np[:-1]
    motion_energy = np.mean(np.abs(diff))
    return float(motion_energy)


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """Analytical calculation of Fréchet Distance between two Gaussian video distributions:
        FVD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * (sigma1 * sigma2)^(1/2))
    """
    diff = mu1 - mu2
    res = linalg.sqrtm(sigma1.dot(sigma2))
    covmean = res[0] if isinstance(res, tuple) else res

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        res = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        covmean = res[0] if isinstance(res, tuple) else res

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
