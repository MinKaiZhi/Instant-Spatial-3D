from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from is3d.types import GaussianCloud


def build_scene_meta(cloud: GaussianCloud, tier_name: str) -> dict:
    if cloud.count == 0:
        bounds_min = [0.0, 0.0, 0.0]
        bounds_max = [0.0, 0.0, 0.0]
    else:
        bounds_min = cloud.xyz.min(axis=0).astype(float).tolist()
        bounds_max = cloud.xyz.max(axis=0).astype(float).tolist()

    synthetic_ratio = float(np.mean(cloud.synthetic_mask.astype(np.float32))) if cloud.count > 0 else 0.0

    return {
        "gaussian_count": cloud.count,
        "synthetic_ratio": synthetic_ratio,
        "tier_name": tier_name,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
    }


def write_scene_meta_json(path: str | Path, meta: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

