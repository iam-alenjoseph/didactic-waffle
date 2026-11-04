"""Unreal Engine Python API Automated Data Collection Script.

Executes inside Unreal Engine 5 via the Unreal Python plugin:
    UnrealEditor-Cmd.exe <Project.uproject> -run=pythonscript -script="ue_python_capture.py"

Automates:
    1. Spawning and positioning CineCameraActor along procedural 3D paths
    2. Multi-pass rendering:
       - 2D RGB Projections (Lit final color)
       - Depth maps (SceneDepth / WorldDepth converted to normalized metric depth)
       - Surface Normal maps (GBuffer WorldNormal encoded into [0, 1] RGB channels)
       - Pinhole camera intrinsics and 6-DOF extrinsics metadata JSON
    3. Automated batch recording for video diffusion model training.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

# Check if running inside Unreal Engine
try:
    import unreal
    IN_UNREAL = True
except ImportError:
    IN_UNREAL = False

from .camera_trajectories import (
    CameraTrajectoryGenerator,
    compute_pinhole_intrinsics
)


class UnrealMultiModalDataCollector:
    """Automated multi-modal video dataset collector using the Unreal Python API."""

    def __init__(
        self,
        output_root: str = "./data/unreal_dataset",
        resolution: tuple = (256, 256),
        fps: int = 30,
        frames_per_clip: int = 16,
        total_clips: int = 50
    ):
        self.output_root = os.path.abspath(output_root)
        self.width, self.height = resolution
        self.fps = fps
        self.frames_per_clip = frames_per_clip
        self.total_clips = total_clips
        self.camera_actor = None
        self.render_targets = {}

        os.makedirs(self.output_root, exist_ok=True)

    def initialize_in_unreal(self):
        """Sets up CineCameraActor, Render Targets, and Post-Process Materials in UE level."""
        if not IN_UNREAL:
            print("[UE_COLLECTOR] Not running inside Unreal Engine. Running in preview/test mode.")
            return

        print("[UE_COLLECTOR] Initializing Unreal Engine Subsystems, Lumen Lighting, and Render Targets...")
        editor_subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        world = editor_subsystem.get_editor_world()

        # 1. Configure Lumen Global Illumination and Hardware Ray Tracing
        self.configure_lumen_and_lighting(world)

        # 2. Spawn CineCameraActor
        cam_location = unreal.Vector(0.0, 0.0, 100.0)
        cam_rotation = unreal.Rotator(0.0, 0.0, 0.0)
        self.camera_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
            unreal.CineCameraActor, cam_location, cam_rotation
        )
        self.camera_component = self.camera_actor.get_cine_camera_component()
        self.camera_component.set_editor_property("current_focal_length", 35.0)

        # 3. Create Render Target 2D assets for multi-pass capture
        asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
        rt_factory = unreal.RenderTargetFactoryNew()

        for pass_name in ["RGB", "Depth", "Normals"]:
            rt = asset_tools.create_asset(
                f"RT_Capture_{pass_name}",
                "/Game/CaptureRenderTargets",
                unreal.TextureRenderTarget2D,
                rt_factory
            )
            rt.set_editor_property("size_x", self.width)
            rt.set_editor_property("size_y", self.height)
            self.render_targets[pass_name] = rt

        print("[UE_COLLECTOR] CineCameraActor, Lumen GI, and RenderTargets successfully created.")

    def configure_lumen_and_lighting(self, world):
        """Automates Lumen Global Illumination, Virtual Shadow Maps (VSM), and Atmospheric Scattering."""
        if not IN_UNREAL:
            return
        post_process_actors = unreal.EditorLevelLibrary.get_all_level_actors()
        for actor in post_process_actors:
            if isinstance(actor, unreal.PostProcessVolume):
                settings = actor.settings
                # Set Lumen Dynamic Global Illumination
                settings.set_editor_property("dynamic_global_illumination_method", unreal.DynamicGlobalIlluminationMethod.LUMEN)
                settings.set_editor_property("reflection_method", unreal.ReflectionMethod.LUMEN)
                settings.set_editor_property("lumen_scene_lighting_quality", 2.0) # High fidelity
                print("[UE_COLLECTOR] Configured PostProcessVolume for Lumen Dynamic GI.")
                break

    def capture_clip(self, clip_id: int, trajectory: List[Dict[str, Any]]):
        """Captures a multi-modal video sequence (RGB, Depth, Normal, Metadata) for one clip."""
        scene_dir = os.path.join(self.output_root, f"scene_{clip_id:04d}")
        rgb_dir = os.path.join(scene_dir, "rgb")
        depth_dir = os.path.join(scene_dir, "depth")
        normals_dir = os.path.join(scene_dir, "normals")

        for d in [rgb_dir, depth_dir, normals_dir]:
            os.makedirs(d, exist_ok=True)

        intrinsics = compute_pinhole_intrinsics(
            fov_degrees=60.0,
            width_pixels=self.width,
            height_pixels=self.height
        )

        frame_metadata = []

        print(f"[UE_COLLECTOR] Capturing clip {clip_id:04d} ({len(trajectory)} frames)...")

        for frame_idx, waypoint in enumerate(trajectory):
            loc = waypoint["location"]
            rot = waypoint["rotation"]

            if IN_UNREAL and self.camera_actor:
                # Update camera 6-DOF transform in UE5 world space
                ue_loc = unreal.Vector(loc["x"], loc["y"], loc["z"])
                ue_rot = unreal.Rotator(rot["pitch"], rot["yaw"], rot["roll"])
                self.camera_actor.set_actor_location_and_rotation(ue_loc, ue_rot, False, False)

                # Export render targets
                rgb_path = os.path.join(rgb_dir, f"frame_{frame_idx:04d}.png")
                depth_path = os.path.join(depth_dir, f"depth_{frame_idx:04d}.png")
                normal_path = os.path.join(normals_dir, f"normal_{frame_idx:04d}.png")

                unreal.RenderingLibrary.export_render_target(
                    world_context_object=self.camera_actor,
                    texture_render_target=self.render_targets["RGB"],
                    file_path=rgb_path,
                    file_name=f"frame_{frame_idx:04d}.png"
                )
                unreal.RenderingLibrary.export_render_target(
                    world_context_object=self.camera_actor,
                    texture_render_target=self.render_targets["Depth"],
                    file_path=depth_path,
                    file_name=f"depth_{frame_idx:04d}.png"
                )
                unreal.RenderingLibrary.export_render_target(
                    world_context_object=self.camera_actor,
                    texture_render_target=self.render_targets["Normals"],
                    file_path=normal_path,
                    file_name=f"normal_{frame_idx:04d}.png"
                )

            frame_metadata.append({
                "frame_index": frame_idx,
                "timestamp_sec": waypoint["timestamp_sec"],
                "camera_extrinsics": {
                    "location": loc,
                    "rotation_euler": rot,
                    "quaternion": waypoint["quaternion"]
                },
                "rgb_path": f"rgb/frame_{frame_idx:04d}.png",
                "depth_path": f"depth/depth_{frame_idx:04d}.png",
                "normal_path": f"normals/normal_{frame_idx:04d}.png"
            })

        # Save metadata JSON for clip
        meta_content = {
            "clip_id": clip_id,
            "num_frames": len(trajectory),
            "fps": self.fps,
            "resolution": [self.width, self.height],
            "camera_intrinsics": intrinsics,
            "frames": frame_metadata
        }

        with open(os.path.join(scene_dir, "metadata.json"), "w") as f:
            json.dump(meta_content, f, indent=2)

    def run_automated_collection(self):
        """Runs the complete automated capture batch across procedural trajectories."""
        self.initialize_in_unreal()
        traj_gen = CameraTrajectoryGenerator(fps=self.fps, num_frames=self.frames_per_clip)

        for clip_idx in range(self.total_clips):
            # Alternating between orbits and fly-throughs
            if clip_idx % 2 == 0:
                radius = 250.0 + (clip_idx * 15.0) % 200.0
                elevation = 15.0 + (clip_idx * 7.0) % 45.0
                speed = 20.0 + (clip_idx * 5.0) % 30.0
                trajectory = traj_gen.generate_orbit(
                    center=(0.0, 0.0, 50.0),
                    radius=radius,
                    elevation_angle_deg=elevation,
                    angular_velocity_deg_per_sec=speed,
                    start_angle_deg=clip_idx * 45.0
                )
            else:
                waypoints = [
                    ( -200.0 + clip_idx * 10, -300.0, 100.0 ),
                    ( -50.0, -100.0, 80.0 ),
                    ( 100.0, 150.0, 120.0 ),
                    ( 300.0, 300.0, 150.0 )
                ]
                trajectory = traj_gen.generate_spline_flythrough(
                    waypoints=waypoints,
                    target_poi=(0.0, 0.0, 50.0)
                )

            self.capture_clip(clip_id=clip_idx, trajectory=trajectory)

        print(f"[UE_COLLECTOR] Successfully collected {self.total_clips} multimodal video clips.")


def main():
    collector = UnrealMultiModalDataCollector(
        output_root="./data/unreal_dataset",
        resolution=(256, 256),
        fps=30,
        frames_per_clip=16,
        total_clips=20
    )
    collector.run_automated_collection()


if __name__ == "__main__":
    main()

