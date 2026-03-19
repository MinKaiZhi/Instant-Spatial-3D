from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(slots=True)
class GaussianCloud:
    xyz: np.ndarray
    scale: np.ndarray
    rotation_xyzw: np.ndarray
    opacity: np.ndarray
    color: np.ndarray
    synthetic_mask: np.ndarray

    def __post_init__(self) -> None:
        count = self.xyz.shape[0]
        fields = [
            self.scale,
            self.rotation_xyzw,
            self.opacity,
            self.color,
            self.synthetic_mask,
        ]
        if any(field.shape[0] != count for field in fields):
            raise ValueError("All gaussian fields must have matching row counts")

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])

    def concat(self, other: "GaussianCloud") -> "GaussianCloud":
        return GaussianCloud(
            xyz=np.concatenate([self.xyz, other.xyz], axis=0),
            scale=np.concatenate([self.scale, other.scale], axis=0),
            rotation_xyzw=np.concatenate([self.rotation_xyzw, other.rotation_xyzw], axis=0),
            opacity=np.concatenate([self.opacity, other.opacity], axis=0),
            color=np.concatenate([self.color, other.color], axis=0),
            synthetic_mask=np.concatenate([self.synthetic_mask, other.synthetic_mask], axis=0),
        )
