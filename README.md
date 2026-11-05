# Generative Diffusion Model for Video with Unreal Engine 5 Multimodal Priors

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" />
  <img src="https://img.shields.io/badge/Hugging%20Face-Diffusers-yellow.svg" />
  <img src="https://img.shields.io/badge/Unreal%20Engine-5.0+-313131.svg" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
  <img src="https://img.shields.io/badge/Tests-21%20Passed-brightgreen.svg" />
</p>

A video diffusion model implemented with PyTorch, incorporating spatio-temporal attention, 3D RoPE, multimodal conditioning, and Unreal Engine 5 automated data generation.

---

## Features

- **DDPM:** Forward diffusion, reverse sampling, and configurable noise schedules.
- **Spatio-Temporal UNet:** Spatial and temporal feature processing.
- **3D RoPE:** Axial rotary positional embeddings.
- **Video ControlNet:** Zero-convolution conditioning.
- **DDIM:** Accelerated video sampling.
- **Multimodal Data:** RGB, depth, and surface normal maps.
- **Unreal Engine 5:** Automated rendering, stereo cameras, and procedural trajectories.
- **Synthetic Generator:** Standalone 3D procedural ray-tracer.
- **Visualization:** Three.js camera viewer, multimodal frame comparison, and diffusion metrics.
- **Evaluation:** PSNR, SSIM, FVD, and temporal consistency.

---

## Installation

```bash
git clone https://github.com/iam-alenjoseph/didactic-waffle.git
cd didactic-waffle

pip install -r requirements.txt
```

---

## Usage

### Run Unit Tests
```bash
python -m unittest discover tests
```

### Generate Synthetic Multimodal Dataset
```bash
python scripts/run_synthetic_data.py --output_dir ./data/synthetic_dataset --num_scenes 20 --num_frames 16 --resolution 64 64
```

### Launch Interactive 3D Visualizer
```bash
python scripts/run_visualizer.py 8080
```
Open `http://localhost:8080` in your browser.
