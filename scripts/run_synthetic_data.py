"""CLI Entry Point: Generate Synthetic Multimodal Video Dataset.

Simulates the Unreal Engine data collection pipeline using procedural 3D ray-tracing.
Produces paired RGB video sequences, depth maps, surface normal maps, and metadata JSON.
"""

import os
import sys
import argparse

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic_generator import SyntheticVideoDatasetGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Multimodal Video Dataset")
    parser.add_argument("--output_dir", type=str, default="./data/synthetic_dataset", help="Output directory path")
    parser.add_argument("--num_scenes", type=int, default=15, help="Number of video scenes to generate")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames per video sequence")
    parser.add_argument("--resolution", type=int, nargs=2, default=[64, 64], help="Resolution [height, width]")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=================================================================")
    print("      SYNTHETIC MULTIMODAL VIDEO DATASET GENERATION              ")
    print("=================================================================")
    print(f"Output Directory : {args.output_dir}")
    print(f"Total Scenes     : {args.num_scenes}")
    print(f"Frames per Scene : {args.num_frames}")
    print(f"Resolution       : {args.resolution[0]}x{args.resolution[1]}")
    print(f"Framerate (FPS)  : {args.fps}")
    print("-----------------------------------------------------------------")

    generator = SyntheticVideoDatasetGenerator(
        output_dir=args.output_dir,
        num_scenes=args.num_scenes,
        num_frames=args.num_frames,
        resolution=tuple(args.resolution),
        fps=args.fps,
        seed=args.seed
    )

    generator.generate_all()
    print("=================================================================")
    print("Dataset generation complete! Dataset ready for DDPM training.")
    print("=================================================================")


if __name__ == "__main__":
    main()

