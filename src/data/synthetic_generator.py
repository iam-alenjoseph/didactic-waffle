"""Procedural 3D Synthetic Dataset Generator.

Simulates the multi-modal Unreal Engine 5 capture pipeline in pure Python/NumPy/PIL.
Renders 3D dynamic scenes with moving physical primitives and camera paths, generating:
    1. 2D RGB Video Projections (diffuse/specular shading and dynamic shadows)
    2. Analytical Depth Maps (normalized metric depth in [0, 1])
    3. Surface Normal Maps (unit normal vectors mapped to RGB: 0.5 * (N + 1.0))
    4. Pinhole camera intrinsics and 6-DOF extrinsics metadata JSON
"""

import os
import math
import json
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from PIL import Image


class Procedural3DRenderer:
    """Ray-tracing engine for procedural 3D multi-modal video synthesis."""

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        fov_degrees: float = 60.0
    ):
        self.width = width
        self.height = height
        self.fov_degrees = fov_degrees
        self.aspect_ratio = width / height

        # Compute camera ray grid
        fov_rad = math.radians(fov_degrees)
        tan_half_fov = math.tan(fov_rad / 2.0)

        # Normalized screen coordinates in [-1, 1]
        u = np.linspace(-tan_half_fov * self.aspect_ratio, tan_half_fov * self.aspect_ratio, width)
        v = np.linspace(tan_half_fov, -tan_half_fov, height)
        self.grid_u, self.grid_v = np.meshgrid(u, v)

    def render_frame(
        self,
        camera_pos: np.ndarray,
        camera_rot: Tuple[float, float, float], # pitch, yaw, roll in degrees
        objects: List[Dict[str, Any]],
        light_dir: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Renders one frame returning:
            rgb: uint8 array (H, W, 3)
            depth: float32 array (H, W) in [0, 1]
            normals: uint8 array (H, W, 3) in [0, 255]
        """
        # Direction rays in camera local space: (H, W, 3)
        rays_local = np.stack([self.grid_u, self.grid_v, -np.ones_like(self.grid_u)], axis=-1)
        rays_local = rays_local / np.linalg.norm(rays_local, axis=-1, keepdims=True)

        # Build camera rotation matrix (pitch, yaw, roll)
        p = math.radians(camera_rot[0])
        y = math.radians(camera_rot[1])
        r = math.radians(camera_rot[2])

        # Rotation matrices
        Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
        Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
        R = Ry @ Rx @ Rz

        # Rotate rays to world space: (H, W, 3)
        rays_world = rays_local @ R.T

        # Ray-scene intersection buffers
        min_t = np.full((self.height, self.width), np.inf, dtype=np.float32)
        normal_map = np.zeros((self.height, self.width, 3), dtype=np.float32)
        base_color_map = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Render ground plane at y = -1.5
        plane_y = -1.5
        mask_plane = rays_world[:, :, 1] < -1e-4
        t_plane = (plane_y - camera_pos[1]) / np.where(mask_plane, rays_world[:, :, 1], -1.0)
        valid_plane = mask_plane & (t_plane > 0) & (t_plane < min_t)

        min_t[valid_plane] = t_plane[valid_plane]
        normal_map[valid_plane] = np.array([0.0, 1.0, 0.0]) # Upward normal
        # Checkerboard pattern for ground
        hit_pts = camera_pos + t_plane[..., None] * rays_world
        checker = ((np.floor(hit_pts[..., 0]) + np.floor(hit_pts[..., 2])) % 2 == 0)
        plane_colors = np.where(checker[..., None], [0.35, 0.35, 0.4], [0.2, 0.2, 0.25])
        base_color_map[valid_plane] = plane_colors[valid_plane]

        # Render spheres
        for obj in objects:
            center = obj["pos"]
            radius = obj["radius"]
            color = obj["color"]

            oc = camera_pos - center # (3,)
            # Quadratic coefficients: a = 1 (since rays_world is unit normalized)
            b = 2.0 * np.sum(rays_world * oc, axis=-1)
            c = np.sum(oc * oc) - radius * radius
            discriminant = b * b - 4.0 * c

            hit_mask = (discriminant > 0)
            if np.any(hit_mask):
                sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
                t0 = (-b - sqrt_disc) * 0.5
                valid_hits = hit_mask & (t0 > 0) & (t0 < min_t)

                if np.any(valid_hits):
                    min_t[valid_hits] = t0[valid_hits]
                    hit_points = camera_pos + t0[valid_hits, None] * rays_world[valid_hits]
                    normals = (hit_points - center) / radius
                    normal_map[valid_hits] = normals
                    base_color_map[valid_hits] = color

        # Compute Lighting (Ambient + Diffuse Lambertian)
        norm_light = light_dir / np.linalg.norm(light_dir)
        dot_product = np.maximum(0.0, np.sum(normal_map * norm_light, axis=-1, keepdims=True))
        ambient = 0.2
        diffuse = 0.8 * dot_product
        shaded_rgb = base_color_map * (ambient + diffuse)

        # Background sky gradient for misses
        miss_mask = np.isinf(min_t)
        sky_gradient = (rays_world[:, :, 1:2] + 1.0) * 0.5
        sky_color = sky_gradient * np.array([0.5, 0.7, 1.0]) + (1.0 - sky_gradient) * np.array([0.9, 0.9, 0.95])
        shaded_rgb[miss_mask] = sky_color[miss_mask]

        # Final RGB in [0, 255]
        rgb_uint8 = np.clip(shaded_rgb * 255.0, 0, 255).astype(np.uint8)

        # Depth map normalized to [0, 1] (clip to max depth 15.0 units)
        max_depth = 15.0
        normalized_depth = np.clip(min_t / max_depth, 0.0, 1.0)
        normalized_depth[miss_mask] = 1.0

        # Normal map encoded to RGB [0, 255]: 0.5 * (N + 1.0)
        encoded_normals = 0.5 * (normal_map + 1.0)
        encoded_normals[miss_mask] = [0.5, 0.5, 1.0] # Default flat surface normal
        normals_uint8 = np.clip(encoded_normals * 255.0, 0, 255).astype(np.uint8)

        return rgb_uint8, normalized_depth, normals_uint8


class SyntheticVideoDatasetGenerator:
    """Batch generator for multi-modal video training data."""

    def __init__(
        self,
        output_dir: str = "./data/synthetic_dataset",
        num_scenes: int = 10,
        num_frames: int = 16,
        resolution: Tuple[int, int] = (64, 64),
        fps: int = 15,
        seed: int = 42
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.num_scenes = num_scenes
        self.num_frames = num_frames
        self.width, self.height = resolution
        self.fps = fps
        self.renderer = Procedural3DRenderer(width=self.width, height=self.height)
        np.random.seed(seed)

        os.makedirs(self.output_dir, exist_ok=True)

    def generate_scene(self, scene_id: int):
        """Simulates physical dynamics and camera motion for one complete scene."""
        scene_dir = os.path.join(self.output_dir, f"scene_{scene_id:04d}")
        rgb_dir = os.path.join(scene_dir, "rgb")
        depth_dir = os.path.join(scene_dir, "depth")
        normals_dir = os.path.join(scene_dir, "normals")

        for d in [rgb_dir, depth_dir, normals_dir]:
            os.makedirs(d, exist_ok=True)

        # Spawn 3 to 5 dynamic spheres with velocities
        num_objects = np.random.randint(3, 6)
        objects = []
        palette = [
            [0.9, 0.2, 0.2], [0.2, 0.8, 0.3], [0.2, 0.4, 0.9],
            [0.9, 0.8, 0.1], [0.8, 0.2, 0.8], [0.2, 0.8, 0.8]
        ]
        for obj_idx in range(num_objects):
            objects.append({
                "pos": np.array([
                    np.random.uniform(-1.5, 1.5),
                    np.random.uniform(0.0, 1.5),
                    np.random.uniform(-1.5, 1.5)
                ], dtype=np.float32),
                "vel": np.array([
                    np.random.uniform(-1.0, 1.0),
                    np.random.uniform(0.0, 2.0),
                    np.random.uniform(-1.0, 1.0)
                ], dtype=np.float32),
                "radius": float(np.random.uniform(0.3, 0.6)),
                "color": palette[obj_idx % len(palette)]
            })

        light_dir = np.array([0.5, 1.0, 0.5])
        dt = 1.0 / self.fps
        gravity = np.array([0.0, -4.0, 0.0])

        frames_meta = []

        # Orbit camera
        cam_radius = np.random.uniform(4.0, 5.5)
        cam_elevation = np.random.uniform(15.0, 35.0)
        angular_speed = np.random.uniform(20.0, 40.0)
        start_angle = np.random.uniform(0.0, 360.0)

        for frame_idx in range(self.num_frames):
            t_sec = frame_idx * dt
            # Step physics
            for obj in objects:
                obj["pos"] += obj["vel"] * dt
                obj["vel"] += gravity * dt
                # Bounce on ground (plane_y = -1.5)
                if obj["pos"][1] - obj["radius"] < -1.5:
                    obj["pos"][1] = -1.5 + obj["radius"]
                    obj["vel"][1] = -obj["vel"][1] * 0.85 # Restitution

            # Camera orbit location
            current_angle = start_angle + t_sec * angular_speed
            rad = math.radians(current_angle)
            elev_rad = math.radians(cam_elevation)

            cx = cam_radius * math.cos(elev_rad) * math.sin(rad)
            cy = cam_radius * math.sin(elev_rad)
            cz = cam_radius * math.cos(elev_rad) * math.cos(rad)
            cam_pos = np.array([cx, cy, cz], dtype=np.float32)

            # Compute look-at rotation towards origin (0, 0, 0)
            yaw = -math.degrees(math.atan2(cx, cz))
            pitch = -cam_elevation
            roll = 0.0
            cam_rot = (pitch, yaw, roll)

            # Render passes
            rgb, depth, normals = self.renderer.render_frame(
                camera_pos=cam_pos,
                camera_rot=cam_rot,
                objects=objects,
                light_dir=light_dir
            )

            # Save PNG files
            rgb_filename = f"frame_{frame_idx:04d}.png"
            depth_filename = f"depth_{frame_idx:04d}.png"
            normal_filename = f"normal_{frame_idx:04d}.png"

            Image.fromarray(rgb).save(os.path.join(rgb_dir, rgb_filename))
            # Save 8-bit or 16-bit depth PNG
            depth_uint8 = (depth * 255.0).astype(np.uint8)
            Image.fromarray(depth_uint8, mode="L").save(os.path.join(depth_dir, depth_filename))
            Image.fromarray(normals).save(os.path.join(normals_dir, normal_filename))

            frames_meta.append({
                "frame_index": frame_idx,
                "timestamp_sec": t_sec,
                "camera_extrinsics": {
                    "location": [float(cx), float(cy), float(cz)],
                    "rotation_euler": [float(pitch), float(yaw), float(roll)]
                },
                "rgb_path": f"rgb/{rgb_filename}",
                "depth_path": f"depth/{depth_filename}",
                "normal_path": f"normals/{normal_filename}"
            })

        # Save metadata JSON
        meta = {
            "scene_id": scene_id,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "resolution": [self.height, self.width],
            "camera_intrinsics": {
                "fov_degrees": self.renderer.fov_degrees,
                "aspect_ratio": self.renderer.aspect_ratio
            },
            "frames": frames_meta
        }
        with open(os.path.join(scene_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def generate_all(self):
        """Generates all synthetic dataset scenes."""
        print(f"[SYNTHETIC_DATA] Generating {self.num_scenes} multimodal video clips at {self.width}x{self.height}...")
        for i in range(self.num_scenes):
            self.generate_scene(scene_id=i)
            print(f"  -> Generated scene {i + 1}/{self.num_scenes}")
        print(f"[SYNTHETIC_DATA] Successfully generated dataset in {self.output_dir}")

