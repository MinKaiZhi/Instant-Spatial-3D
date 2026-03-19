from __future__ import annotations

import numpy as np

from is3d.config import PipelineSettings
from is3d.types import GaussianCloud


def hallucinate_backside(cloud: GaussianCloud, settings: PipelineSettings) -> GaussianCloud:
    ratio = float(np.clip(settings.completion_ratio, 0.0, 1.0))
    if ratio <= 0.0 or cloud.count == 0:
        return cloud

    extra_count = int(max(1, round(cloud.count * ratio)))
    extra_count = min(extra_count, cloud.count)

    candidate_idx = np.argsort(cloud.xyz[:, 2])[-extra_count:]

    xyz = cloud.xyz[candidate_idx].copy()
    xyz[:, 0] *= -1.0
    xyz[:, 2] += 0.08

    extra = GaussianCloud(
        xyz=xyz,
        scale=cloud.scale[candidate_idx].copy(),
        rotation_xyzw=cloud.rotation_xyzw[candidate_idx].copy(),
        opacity=np.clip(cloud.opacity[candidate_idx] * settings.completion_opacity_gain, 0.02, 0.9),
        color=cloud.color[candidate_idx].copy(),
        synthetic_mask=np.ones((extra_count,), dtype=bool),
    )
    return cloud.concat(extra)
