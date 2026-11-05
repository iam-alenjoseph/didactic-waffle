"""Unit tests for diffusion noise schedules and mathematical precomputations."""

import unittest
import numpy as np
from src.diffusion.schedules import (
    linear_beta_schedule,
    cosine_beta_schedule,
    sigmoid_beta_schedule,
    DiffusionSchedule
)


class TestDiffusionSchedules(unittest.TestCase):

    def test_linear_schedule_bounds(self):
        betas = linear_beta_schedule(1000, beta_start=0.0001, beta_end=0.02)
        self.assertEqual(len(betas), 1000)
        self.assertAlmostEqual(betas[0], 0.0001, places=6)
        self.assertAlmostEqual(betas[-1], 0.02, places=6)
        self.assertTrue(np.all(np.diff(betas) >= 0), "Linear betas should be non-decreasing")

    def test_cosine_schedule_bounds(self):
        betas = cosine_beta_schedule(1000, s=0.008)
        self.assertEqual(len(betas), 1000)
        self.assertTrue(np.all(betas > 0), "Betas must be strictly positive")
        self.assertTrue(np.all(betas < 1.0), "Betas must be less than 1.0")

        # In cosine schedule, alpha_cumprod should start near 1.0 and end near 0.0
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        self.assertGreater(alphas_cumprod[0], 0.99)
        self.assertLess(alphas_cumprod[-1], 0.01)

    def test_sigmoid_schedule(self):
        betas = sigmoid_beta_schedule(500)
        self.assertEqual(len(betas), 500)
        self.assertTrue(np.all(betas > 0))
        self.assertTrue(np.all(betas < 1.0))

    def test_diffusion_schedule_container(self):
        schedule = DiffusionSchedule(timesteps=100, schedule_type="cosine")
        self.assertEqual(len(schedule.alphas_cumprod), 100)
        self.assertEqual(len(schedule.posterior_variance), 100)
        self.assertEqual(len(schedule.posterior_mean_coef1), 100)
        self.assertEqual(len(schedule.posterior_mean_coef2), 100)

        # Test extraction broadcasting for 5D video tensor (B, C, T, H, W)
        t = np.array([10, 50])
        extracted = schedule.extract(schedule.sqrt_alphas_cumprod, t, (2, 3, 16, 64, 64))
        self.assertEqual(extracted.shape, (2, 1, 1, 1, 1))
        self.assertAlmostEqual(extracted[0, 0, 0, 0, 0], schedule.sqrt_alphas_cumprod[10])
        self.assertAlmostEqual(extracted[1, 0, 0, 0, 0], schedule.sqrt_alphas_cumprod[50])


if __name__ == "__main__":
    unittest.main()

