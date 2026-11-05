"""Unit tests for video generation evaluation metrics (PSNR, SSIM, FVD)."""

import unittest
import numpy as np

from src.utils.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_temporal_consistency,
    calculate_frechet_distance
)


class TestVideoMetrics(unittest.TestCase):

    def test_psnr_identical(self):
        v = np.random.rand(8, 32, 32, 3).astype(np.float32)
        psnr = calculate_psnr(v, v)
        self.assertEqual(psnr, float("inf"))

    def test_psnr_noisy(self):
        v = np.ones((8, 32, 32, 3), dtype=np.float32) * 0.5
        v_noisy = v + np.random.normal(0, 0.05, v.shape).astype(np.float32)
        psnr = calculate_psnr(v, v_noisy)
        self.assertGreater(psnr, 20.0)
        self.assertLess(psnr, 40.0)

    def test_ssim_identical(self):
        v = np.random.rand(4, 32, 32, 3).astype(np.float32)
        ssim = calculate_ssim(v, v)
        self.assertAlmostEqual(ssim, 1.0, places=4)

    def test_temporal_consistency_static(self):
        # Static video (identical frames) should have zero temporal motion variation
        frame = np.random.rand(32, 32, 3).astype(np.float32)
        v = np.stack([frame] * 8, axis=0)
        tc = calculate_temporal_consistency(v)
        self.assertAlmostEqual(tc, 0.0, places=6)

    def test_frechet_distance_identical_distributions(self):
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.eye(3) * 0.5
        fvd = calculate_frechet_distance(mu, sigma, mu, sigma)
        self.assertAlmostEqual(fvd, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

