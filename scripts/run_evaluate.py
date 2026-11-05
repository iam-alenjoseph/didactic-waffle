"""Quantitative Evaluation Script for Video Diffusion Models.

Computes:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - Temporal Consistency Score
    - Fréchet Video Distance (FVD) between test ground truth and generated videos.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.video_dataset import VideoMultimodalDataset
from src.utils.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_temporal_consistency,
    calculate_frechet_distance
)


def evaluate_dataset_pair(
    gt_dataset_dir: str,
    gen_dataset_dir: str,
    num_frames: int = 16,
    resolution: tuple = (64, 64)
) -> dict:
    """Evaluates metrics between ground truth dataset and generated video sequences."""
    gt_ds = VideoMultimodalDataset(gt_dataset_dir, num_frames=num_frames, resolution=resolution)
    gen_ds = VideoMultimodalDataset(gen_dataset_dir, num_frames=num_frames, resolution=resolution)

    num_samples = min(len(gt_ds), len(gen_ds))
    if num_samples == 0:
        raise ValueError("No video scenes found in one or both directories.")

    psnr_scores = []
    ssim_scores = []
    temp_scores_gt = []
    temp_scores_gen = []

    print(f"[EVALUATE] Comparing {num_samples} video sequences...")

    for i in range(num_samples):
        sample_gt = gt_ds[i]["rgb"]
        sample_gen = gen_ds[i]["rgb"]

        psnr_val = calculate_psnr(sample_gt, sample_gen)
        ssim_val = calculate_ssim(sample_gt, sample_gen)
        temp_gt = calculate_temporal_consistency(sample_gt)
        temp_gen = calculate_temporal_consistency(sample_gen)

        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)
        temp_scores_gt.append(temp_gt)
        temp_scores_gen.append(temp_gen)

    # Simplified feature distribution for FVD demonstration
    feats_gt = np.array(temp_scores_gt)[:, None]
    feats_gen = np.array(temp_scores_gen)[:, None]
    mu1, sigma1 = np.mean(feats_gt, axis=0), np.cov(feats_gt, rowvar=False)
    mu2, sigma2 = np.mean(feats_gen, axis=0), np.cov(feats_gen, rowvar=False)
    if sigma1.ndim == 0:
        sigma1 = sigma1[None, None]
    if sigma2.ndim == 0:
        sigma2 = sigma2[None, None]
    fvd_approx = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    finite_psnr = [p for p in psnr_scores if np.isfinite(p)]
    mean_psnr = float(np.mean(finite_psnr)) if finite_psnr else float("inf")
    std_psnr = float(np.std(finite_psnr)) if finite_psnr else 0.0

    results = {
        "num_evaluated_sequences": num_samples,
        "mean_psnr_db": mean_psnr,
        "std_psnr_db": std_psnr,
        "mean_ssim": float(np.mean(ssim_scores)),
        "std_ssim": float(np.std(ssim_scores)),
        "mean_temporal_consistency_gen": float(np.mean(temp_scores_gen)),
        "mean_temporal_consistency_gt": float(np.mean(temp_scores_gt)),
        "approx_frechet_video_distance": float(fvd_approx)
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Video Diffusion Generation Quality")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to ground truth video dataset")
    parser.add_argument("--gen_dir", type=str, required=True, help="Path to generated video dataset")
    parser.add_argument("--output_json", type=str, default="./outputs/metrics_report.json", help="Report JSON path")
    args = parser.parse_args()

    results = evaluate_dataset_pair(args.gt_dir, args.gen_dir)

    print("\n==================================================================")
    print("                VIDEO DIFFUSION EVALUATION REPORT                 ")
    print("==================================================================")
    print(f"Evaluated Clips           : {results['num_evaluated_sequences']}")
    print(f"Mean PSNR (dB)            : {results['mean_psnr_db']:.2f} ± {results['std_psnr_db']:.2f}")
    print(f"Mean SSIM                 : {results['mean_ssim']:.4f} ± {results['std_ssim']:.4f}")
    print(f"Temporal Consistency (Gen): {results['mean_temporal_consistency_gen']:.4f}")
    print(f"Temporal Consistency (GT) : {results['mean_temporal_consistency_gt']:.4f}")
    print(f"Approx. FVD Score         : {results['approx_frechet_video_distance']:.4f}")
    print("==================================================================")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
