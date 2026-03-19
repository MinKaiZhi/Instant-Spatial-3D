from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from is3d.types import GaussianCloud


@dataclass(slots=True)
class NavMeshConfig:
    cell_size_m: float = 0.2
    height_tolerance_m: float = 0.25
    min_points_per_cell: int = 2


@dataclass(slots=True)
class NavMeshData:
    cell_size_m: float
    origin_x: float
    origin_z: float
    width: int
    height: int
    walkable_cells: list[list[int]]
    ground_height_m: float


def build_navmesh(cloud: GaussianCloud, cfg: NavMeshConfig) -> NavMeshData:
    if cloud.count == 0:
        return NavMeshData(
            cell_size_m=cfg.cell_size_m,
            origin_x=0.0,
            origin_z=0.0,
            width=0,
            height=0,
            walkable_cells=[],
            ground_height_m=0.0,
        )

    xyz = cloud.xyz
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    ground_h = float(np.percentile(y, 20.0))
    ground_mask = y <= (ground_h + cfg.height_tolerance_m)
    if not np.any(ground_mask):
        ground_mask = np.ones_like(y, dtype=bool)

    gx = x[ground_mask]
    gz = z[ground_mask]

    min_x = float(np.min(gx))
    max_x = float(np.max(gx))
    min_z = float(np.min(gz))
    max_z = float(np.max(gz))

    cell = max(cfg.cell_size_m, 1e-3)
    width = int(np.floor((max_x - min_x) / cell)) + 1
    height = int(np.floor((max_z - min_z) / cell)) + 1

    ix = np.floor((gx - min_x) / cell).astype(np.int32)
    iz = np.floor((gz - min_z) / cell).astype(np.int32)

    counts = np.zeros((height, width), dtype=np.int32)
    for cx, cz in zip(ix, iz):
        if 0 <= cz < height and 0 <= cx < width:
            counts[cz, cx] += 1

    walkable = np.argwhere(counts >= max(1, cfg.min_points_per_cell))
    walkable_cells = [[int(col), int(row)] for row, col in walkable]

    return NavMeshData(
        cell_size_m=cell,
        origin_x=min_x,
        origin_z=min_z,
        width=width,
        height=height,
        walkable_cells=walkable_cells,
        ground_height_m=ground_h,
    )


def write_navmesh_json(path: str | Path, navmesh: NavMeshData) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(asdict(navmesh), handle, ensure_ascii=True, indent=2)
        handle.write("\n")

