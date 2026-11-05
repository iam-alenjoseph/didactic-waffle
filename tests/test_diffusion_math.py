"""Unit tests verifying analytical diffusion formulas and posterior derivation."""

import unittest
import numpy as np
from src.diffusion.schedules import DiffusionSchedule
from src.diffusion.ddpm import VideoDDPM


class TestDiffusionMath(unittest.TestCase):

    def setUp(self):
        self.schedule = DiffusionSchedule(timesteps=500, schedule_type="linear")
        self.ddpm = VideoDDPM(timesteps=500, beta_schedule="linear")

    def test_posterior_variance_formula(self):
        """Validates: beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t."""
        for t in [1, 50, 100, 250, 499]:
            alpha_bar_t = self.schedule.alphas_cumprod[t]
            alpha_bar_prev = self.schedule.alphas_cumprod_prev[t]
            beta_t = self.schedule.betas[t]

            expected_variance = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t
            actual_variance = self.schedule.posterior_variance[t]

            self.assertAlmostEqual(expected_variance, actual_variance, places=7)

    def test_posterior_mean_coefficients_sum_to_one(self):
        """In the limit x_0 = x_t = x, posterior mean mu_tilde_t should equal x * sqrt(alpha_t)."""
        for t in [1, 10, 100, 300]:
            c1 = self.schedule.posterior_mean_coef1[t]
            c2 = self.schedule.posterior_mean_coef2[t]

            alpha_bar_t = self.schedule.alphas_cumprod[t]
            alpha_bar_prev = self.schedule.alphas_cumprod_prev[t]
            alpha_t = self.schedule.alphas[t]
            beta_t = self.schedule.betas[t]

            # Direct formula test
            expected_c1 = beta_t * np.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
            expected_c2 = (1.0 - alpha_bar_prev) * np.sqrt(alpha_t) / (1.0 - alpha_bar_t)

            self.assertAlmostEqual(c1, expected_c1, places=6)
            self.assertAlmostEqual(c2, expected_c2, places=6)

    def test_forward_diffusion_moments(self):
        """Tests that forward q(x_t | x_0) has theoretical mean sqrt(alpha_bar_t) * x_0
        and variance (1 - alpha_bar_t).
        """
        np.random.seed(42)
        x0 = np.ones((1000, 1, 1, 1, 1), dtype=np.float32) * 2.0
        t = np.full((1000,), 100)

        # Draw 1000 Monte-Carlo samples
        samples = self.ddpm.q_sample(x0, t)

        expected_mean = np.sqrt(self.schedule.alphas_cumprod[100]) * 2.0
        expected_var = 1.0 - self.schedule.alphas_cumprod[100]

        sample_mean = float(np.mean(samples))
        sample_var = float(np.var(samples))

        self.assertAlmostEqual(sample_mean, expected_mean, delta=0.1)
        self.assertAlmostEqual(sample_var, expected_var, delta=0.1)


if __name__ == "__main__":
    unittest.main()

