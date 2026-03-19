from __future__ import annotations

import numpy as np

from is3d.config import PipelineSettings
from is3d.types import CameraIntrinsics, GaussianCloud


def regress_gaussians(
    image: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    settings: PipelineSettings,
) -> GaussianCloud:
    height, width, _ = image.shape

    stride = max(1, int(settings.sample_stride))
    ys = np.arange(0, height, stride)
    xs = np.arange(0, width, stride)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")

    y_idx = grid_y.reshape(-1)
    x_idx = grid_x.reshape(-1)

    z = depth_m[y_idx, x_idx]
    x = ((x_idx.astype(np.float32) - intrinsics.cx) / intrinsics.fx) * z
    y = ((y_idx.astype(np.float32) - intrinsics.cy) / intrinsics.fy) * z
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)

    depth_grad = np.zeros_like(depth_m)
    depth_grad[:, 1:] += np.abs(depth_m[:, 1:] - depth_m[:, :-1])
    depth_grad[1:, :] += np.abs(depth_m[1:, :] - depth_m[:-1, :])
    edge = np.clip(depth_grad[y_idx, x_idx], 0.0, 1.0)

    opacity = np.clip(0.95 - edge * 0.9, 0.05, 0.99).astype(np.float32)

    scale_base = np.clip(z * settings.gaussian_scale_ratio, 0.004, 0.08)
    scale = np.stack([scale_base, scale_base, scale_base], axis=-1).astype(np.float32)

    rotation = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (xyz.shape[0], 1))

    color = image[y_idx, x_idx].astype(np.float32)
    synthetic_mask = np.zeros((xyz.shape[0],), dtype=bool)

    return GaussianCloud(
        xyz=xyz,
        scale=scale,
        rotation_xyzw=rotation,
        opacity=opacity,
        color=color,
        synthetic_mask=synthetic_mask,
    )
