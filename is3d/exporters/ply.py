from __future__ import annotations

from pathlib import Path

import numpy as np

from is3d.types import GaussianCloud


def write_gaussian_ply(path: str | Path, cloud: GaussianCloud) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    n = cloud.count
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "property uchar synthetic",
        "end_header",
    ]

    with output.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header))
        handle.write("\n")

        zeros = np.zeros((n, 3), dtype=np.float32)
        for idx in range(n):
            row = np.concatenate(
                [
                    cloud.xyz[idx],
                    zeros[idx],
                    cloud.color[idx],
                    np.array([cloud.opacity[idx]], dtype=np.float32),
                    cloud.scale[idx],
                    cloud.rotation_xyzw[idx],
                    np.array([1.0 if cloud.synthetic_mask[idx] else 0.0], dtype=np.float32),
                ]
            )
            synthetic = int(row[-1])
            floats = " ".join(f"{value:.6f}" for value in row[:-1])
            handle.write(f"{floats} {synthetic}\n")
