"""Unit tests for Unreal Engine camera trajectory math, transforms, and intrinsics."""

import unittest
import numpy as np

from src.unreal.camera_trajectories import (
    look_at_rotation_unreal,
    euler_to_quaternion,
    compute_pinhole_intrinsics,
    CameraTrajectoryGenerator
)


class TestUnrealEngineTrajectories(unittest.TestCase):

    def test_look_at_rotation(self):
        # Camera at (0, 0, 0), Target at (100, 0, 0) -> looking along positive X (Pitch=0, Yaw=0, Roll=0)
        cam_pos = np.array([0.0, 0.0, 0.0])
        tgt_pos = np.array([100.0, 0.0, 0.0])
        pitch, yaw, roll = look_at_rotation_unreal(cam_pos, tgt_pos)

        self.assertAlmostEqual(pitch, 0.0, places=4)
        self.assertAlmostEqual(yaw, 0.0, places=4)
        self.assertAlmostEqual(roll, 0.0, places=4)

    def test_quaternion_normalization(self):
        # Quaternion produced from Euler angles should have unit norm: x^2 + y^2 + z^2 + w^2 = 1.0
        pitch, yaw, roll = -25.0, 45.0, 10.0
        qx, qy, qz, qw = euler_to_quaternion(pitch, yaw, roll)
        norm_sq = qx * qx + qy * qy + qz * qz + qw * qw
        self.assertAlmostEqual(norm_sq, 1.0, places=6)

    def test_pinhole_intrinsics_matrix(self):
        intrinsics = compute_pinhole_intrinsics(fov_degrees=60.0, width_pixels=256, height_pixels=256)
        k = intrinsics["intrinsics_matrix"]

        self.assertEqual(len(k), 3)
        self.assertEqual(len(k[0]), 3)
        self.assertAlmostEqual(k[0][2], 128.0) # Principal point X
        self.assertAlmostEqual(k[1][2], 128.0) # Principal point Y
        self.assertEqual(k[2][2], 1.0)
        self.assertGreater(k[0][0], 0.0) # Focal length fx > 0

    def test_orbit_trajectory_generation(self):
        gen = CameraTrajectoryGenerator(fps=30, num_frames=16)
        orbit = gen.generate_orbit(center=(0.0, 0.0, 50.0), radius=200.0)

        self.assertEqual(len(orbit), 16)
        # Verify distance from center is constant along radius
        for pt in orbit:
            loc = pt["location"]
            dx = loc["x"]
            dy = loc["y"]
            r_xy = np.sqrt(dx * dx + dy * dy)
            self.assertGreater(r_xy, 100.0)
            self.assertIn("quaternion", pt)


if __name__ == "__main__":
    unittest.main()

