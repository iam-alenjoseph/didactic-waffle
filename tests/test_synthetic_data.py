"""Unit tests for procedural synthetic multimodal dataset generation."""

import os
import shutil
import tempfile
import unittest
import numpy as np
from PIL import Image

from src.data.synthetic_generator import (
    Procedural3DRenderer,
    SyntheticVideoDatasetGenerator
)


class TestSyntheticDataGenerator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.renderer = Procedural3DRenderer(width=32, height=32, fov_degrees=60.0)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_renderer_outputs(self):
        cam_pos = np.array([0.0, 1.0, 3.0], dtype=np.float32)
        cam_rot = (-15.0, 0.0, 0.0) # pitch, yaw, roll
        objects = [{
            "pos": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "radius": 0.5,
            "color": [1.0, 0.0, 0.0]
        }]
        light_dir = np.array([0.5, 1.0, 0.5])

        rgb, depth, normals = self.renderer.render_frame(cam_pos, cam_rot, objects, light_dir)

        # Check shapes
        self.assertEqual(rgb.shape, (32, 32, 3))
        self.assertEqual(depth.shape, (32, 32))
        self.assertEqual(normals.shape, (32, 32, 3))

        # Check types and ranges
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(normals.dtype, np.uint8)
        self.assertTrue(0.0 <= depth.min() <= depth.max() <= 1.0)

    def test_batch_generator_creates_valid_structure(self):
        generator = SyntheticVideoDatasetGenerator(
            output_dir=self.temp_dir,
            num_scenes=2,
            num_frames=4,
            resolution=(32, 32),
            fps=10,
            seed=123
        )
        generator.generate_all()

        scene0 = os.path.join(self.temp_dir, "scene_0000")
        self.assertTrue(os.path.exists(scene0))
        self.assertTrue(os.path.exists(os.path.join(scene0, "rgb")))
        self.assertTrue(os.path.exists(os.path.join(scene0, "depth")))
        self.assertTrue(os.path.exists(os.path.join(scene0, "normals")))
        self.assertTrue(os.path.exists(os.path.join(scene0, "metadata.json")))

        # Check frame file count
        rgb_files = os.listdir(os.path.join(scene0, "rgb"))
        self.assertEqual(len(rgb_files), 4)


if __name__ == "__main__":
    unittest.main()

