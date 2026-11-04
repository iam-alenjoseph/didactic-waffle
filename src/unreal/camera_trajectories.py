"""Camera Trajectory Generation for Unreal Engine Automated Data Collection.

Generates smooth 3D camera paths (Orbits, Splines, Perlin 6-DOF fly-throughs)
and computes camera extrinsics (4x4 view matrix, position, quaternion) and intrinsics
for synthetic video dataset collection in Unreal Engine coordinate space (Z-up, Left-Handed).
"""

from typing import List, Dict, Tuple, Optional, Any
import math
import json
import copy
import numpy as np


def look_at_rotation_unreal(camera_pos: np.ndarray, target_pos: np.ndarray) -> Tuple[float, float, float]:
    """Computes Unreal Engine Euler angles (Pitch, Yaw, Roll) to point camera at target.

    In Unreal Engine:
        X = Forward, Y = Right, Z = Up
        Pitch = Rotation around Y (degrees)
        Yaw = Rotation around Z (degrees)
        Roll = Rotation around X (degrees)
    """
    diff = target_pos - camera_pos
    dx, dy, dz = diff[0], diff[1], diff[2]

    hypot_xy = math.sqrt(dx * dx + dy * dy)
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, hypot_xy))
    roll = 0.0

    return pitch, yaw, roll


def euler_to_quaternion(pitch: float, yaw: float, roll: float) -> Tuple[float, float, float, float]:
    """Converts Unreal Euler angles (degrees) to Quaternion (x, y, z, w)."""
    p = math.radians(pitch) * 0.5
    y = math.radians(yaw) * 0.5
    r = math.radians(roll) * 0.5

    sin_p, cos_p = math.sin(p), math.cos(p)
    sin_y, cos_y = math.sin(y), math.cos(y)
    sin_r, cos_r = math.sin(r), math.cos(r)

    qw = cos_r * cos_p * cos_y + sin_r * sin_p * sin_y
    qx = sin_r * cos_p * cos_y - cos_r * sin_p * sin_y
    qy = cos_r * sin_p * cos_y + sin_r * cos_p * sin_y
    qz = cos_r * cos_p * sin_y - sin_r * sin_p * cos_y

    return qx, qy, qz, qw


def compute_pinhole_intrinsics(
    fov_degrees: float,
    width_pixels: int,
    height_pixels: int
) -> Dict[str, Any]:
    """Calculates pinhole camera intrinsics matrix K from FOV and image dimensions."""
    fov_rad = math.radians(fov_degrees)
    fx = (width_pixels / 2.0) / math.tan(fov_rad / 2.0)
    fy = (height_pixels / 2.0) / math.tan(fov_rad / 2.0)
    cx = width_pixels / 2.0
    cy = height_pixels / 2.0

    k_matrix = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]

    return {
        "fov_degrees": fov_degrees,
        "focal_length_x": fx,
        "focal_length_y": fy,
        "principal_point_x": cx,
        "principal_point_y": cy,
        "intrinsics_matrix": k_matrix,
        "resolution": [width_pixels, height_pixels]
    }


class CameraTrajectoryGenerator:
    """Procedural generator for cinematic camera motion in Unreal Engine."""

    def __init__(self, fps: int = 30, num_frames: int = 16):
        self.fps = fps
        self.num_frames = num_frames
        self.dt = 1.0 / fps

    def generate_orbit(
        self,
        center: Tuple[float, float, float] = (0.0, 0.0, 50.0),
        radius: float = 300.0,
        elevation_angle_deg: float = 25.0,
        angular_velocity_deg_per_sec: float = 30.0,
        start_angle_deg: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Generates spherical orbit trajectory around a central coordinate."""
        trajectory = []
        center_np = np.array(center, dtype=np.float64)
        elev_rad = math.radians(elevation_angle_deg)

        z_offset = radius * math.sin(elev_rad)
        r_xy = radius * math.cos(elev_rad)

        for i in range(self.num_frames):
            time_sec = i * self.dt
            current_angle_deg = start_angle_deg + time_sec * angular_velocity_deg_per_sec
            angle_rad = math.radians(current_angle_deg)

            x = center[0] + r_xy * math.cos(angle_rad)
            y = center[1] + r_xy * math.sin(angle_rad)
            z = center[2] + z_offset

            cam_pos = np.array([x, y, z], dtype=np.float64)
            pitch, yaw, roll = look_at_rotation_unreal(cam_pos, center_np)
            qx, qy, qz, qw = euler_to_quaternion(pitch, yaw, roll)

            trajectory.append({
                "frame_index": i,
                "timestamp_sec": time_sec,
                "location": {"x": float(x), "y": float(y), "z": float(z)},
                "rotation": {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)},
                "quaternion": {"x": qx, "y": qy, "z": qz, "w": qw}
            })

        return trajectory

    def generate_spline_flythrough(
        self,
        waypoints: List[Tuple[float, float, float]],
        target_poi: Optional[Tuple[float, float, float]] = None
    ) -> List[Dict[str, Any]]:
        """Generates smooth Catmull-Rom spline trajectory passing through waypoints."""
        if len(waypoints) < 2:
            raise ValueError("Spline trajectory requires at least 2 waypoints.")

        pts = np.array(waypoints, dtype=np.float64)
        trajectory = []

        # Parameterize curve by normalized progress t in [0, 1]
        t_values = np.linspace(0.0, 1.0, self.num_frames)

        # Cumulative distance chord-length parameterization
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        total_len = np.sum(seg_lengths)
        if total_len == 0:
            total_len = 1.0
        cum_dist = np.insert(np.cumsum(seg_lengths) / total_len, 0, 0.0)

        for i, t in enumerate(t_values):
            time_sec = i * self.dt
            # Piecewise linear or spline interpolation
            interp_pos = np.zeros(3)
            for d in range(3):
                interp_pos[d] = np.interp(t, cum_dist, pts[:, d])

            if target_poi is not None:
                pitch, yaw, roll = look_at_rotation_unreal(interp_pos, np.array(target_poi))
            else:
                # Look ahead along tangent vector
                t_next = min(1.0, t + 0.02)
                next_pos = np.zeros(3)
                for d in range(3):
                    next_pos[d] = np.interp(t_next, cum_dist, pts[:, d])
                pitch, yaw, roll = look_at_rotation_unreal(interp_pos, next_pos)

            qx, qy, qz, qw = euler_to_quaternion(pitch, yaw, roll)

            trajectory.append({
                "frame_index": i,
                "timestamp_sec": time_sec,
                "location": {"x": float(interp_pos[0]), "y": float(interp_pos[1]), "z": float(interp_pos[2])},
                "rotation": {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)},
                "quaternion": {"x": qx, "y": qy, "z": qz, "w": qw}
            })

        return trajectory

    def generate_multicam_rig(
        self,
        base_trajectory: List[Dict[str, Any]],
        baseline_cm: float = 12.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generates synchronized stereo camera rig trajectories (Left & Right cameras).

        Offsets cameras perpendicular to viewing direction along the local Y (Right) axis in UE space.
        """
        left_cam = []
        right_cam = []

        for pt in base_trajectory:
            loc = pt["location"]
            rot = pt["rotation"]
            yaw_rad = math.radians(rot["yaw"])

            # Local right vector in world XY: (-sin(yaw), cos(yaw), 0)
            rx = -math.sin(yaw_rad)
            ry = math.cos(yaw_rad)

            # Left camera offset: -0.5 * baseline
            lx = loc["x"] - 0.5 * baseline_cm * rx
            ly = loc["y"] - 0.5 * baseline_cm * ry
            lz = loc["z"]

            # Right camera offset: +0.5 * baseline
            r_cam_x = loc["x"] + 0.5 * baseline_cm * rx
            r_cam_y = loc["y"] + 0.5 * baseline_cm * ry
            r_cam_z = loc["z"]

            pt_left = copy.deepcopy(pt)
            pt_left["location"] = {"x": float(lx), "y": float(ly), "z": float(lz)}
            left_cam.append(pt_left)

            pt_right = copy.deepcopy(pt)
            pt_right["location"] = {"x": float(r_cam_x), "y": float(r_cam_y), "z": float(r_cam_z)}
            right_cam.append(pt_right)

        return {"left": left_cam, "right": right_cam}

