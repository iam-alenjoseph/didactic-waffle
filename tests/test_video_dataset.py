"""Unit tests for multimodal video dataset loader and spatio-temporal transforms."""

import os
import shutil
import tempfile
import unittest
import numpy as np

from src.data.synthetic_generator import SyntheticVideoDatasetGenerator
from src.data.video_dataset import VideoMultimodalDataset
from src.data.transforms import VideoTransform


class TestVideoDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        generator = SyntheticVideoDatasetGenerator(
            output_dir=cls.temp_dir,
            num_scenes=2,
            num_frames=8,
            resolution=(32, 32),
            seed=42
        )
        generator.generate_all()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_dataset_discovery_and_length(self):
        dataset = VideoMultimodalDataset(
            data_dir=self.temp_dir,
            num_frames=8,
            resolution=(32, 32)
        )
        self.assertEqual(len(dataset), 2)

    def test_sample_shapes_and_ranges(self):
        dataset = VideoMultimodalDataset(
            data_dir=self.temp_dir,
            num_frames=4,
            resolution=(32, 32)
        )
        sample = dataset[0]

        # Check keys
        self.assertIn("rgb", sample)
        self.assertIn("depth", sample)
        self.assertIn("normal", sample)
        self.assertIn("condition", sample)
        self.assertIn("metadata", sample)

        rgb = sample["rgb"]
        depth = sample["depth"]
        normal = sample["normal"]
        condition = sample["condition"]

        # Expected shapes: (C, T, H, W)
        self.assertEqual(rgb.shape, (3, 4, 32, 32))
        self.assertEqual(depth.shape, (1, 4, 32, 32))
        self.assertEqual(normal.shape, (3, 4, 32, 32))
        self.assertEqual(condition.shape, (4, 4, 32, 32)) # 1 depth + 3 normal = 4

        # Values should be normalized to [-1, 1]
        self.assertGreaterEqual(float(np.min(rgb)), -1.0)
        self.assertLessEqual(float(np.max(rgb)), 1.0)
        self.assertGreaterEqual(float(np.min(normal)), -1.0)
        self.assertLessEqual(float(np.max(normal)), 1.0)

    def test_transforms_augmentation_consistency(self):
        transform = VideoTransform(
            target_resolution=(32, 32),
            horizontal_flip_prob=1.0, # Force flip
            temporal_reverse_prob=1.0 # Force reverse
        )
        rgb_raw = np.random.randint(0, 255, (4, 32, 32, 3), dtype=np.uint8)
        depth_raw = np.random.randint(0, 255, (4, 32, 32), dtype=np.uint8)
        norm_raw = np.random.randint(0, 255, (4, 32, 32, 3), dtype=np.uint8)

        rgb_t, depth_t, norm_t = transform(rgb_raw, depth_raw, norm_raw)

        # Result shape (C, T, H, W)
        self.assertEqual(rgb_t.shape, (3, 4, 32, 32))
        self.assertEqual(depth_t.shape, (1, 4, 32, 32))
        self.assertEqual(norm_t.shape, (3, 4, 32, 32))


if __name__ == "__main__":
    unittest.main()

