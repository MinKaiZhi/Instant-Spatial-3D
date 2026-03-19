from __future__ import annotations

import numpy as np

from is3d.config import PipelineSettings
from is3d.types import CameraIntrinsics


def estimate_intrinsics(height: int, width: int, fov_degrees: float = 60.0) -> CameraIntrinsics:
    half_fov = np.deg2rad(fov_degrees * 0.5)
    fx = 0.5 * width / np.tan(half_fov)
    fy = 0.5 * height / np.tan(half_fov)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return CameraIntrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))


def _smooth_depth(depth: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return depth

    padded = np.pad(depth, ((1, 1), (1, 1)), mode="edge")
    avg = (
        padded[1:-1, 1:-1]
        + padded[0:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, 0:-2]
        + padded[1:-1, 2:]
    ) / 5.0
    return (1.0 - strength) * depth + strength * avg


def estimate_metric_depth(features: np.ndarray, settings: PipelineSettings) -> np.ndarray:
    luminance = features[..., 0]
    edge_strength = features[..., 1]

    depth_raw = np.clip(1.0 - luminance, 0.0, 1.0)
    depth_raw = _smooth_depth(depth_raw, settings.depth_smoothing)
    depth_raw = np.clip(depth_raw + edge_strength * 0.08, 0.0, 1.0)

    min_depth = settings.min_depth_m
    max_depth = settings.max_depth_m
    depth_m = min_depth + (max_depth - min_depth) * depth_raw
    return depth_m.astype(np.float32)
