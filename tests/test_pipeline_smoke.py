import json
from pathlib import Path

import numpy as np
from PIL import Image

from is3d.config import load_config
from is3d.pipeline import IS3DPipeline


def test_pipeline_generates_ply(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    output_path = tmp_path / "scene.ply"

    data = np.zeros((48, 64, 3), dtype=np.uint8)
    data[..., 0] = np.linspace(10, 230, data.shape[1], dtype=np.uint8)
    data[..., 1] = np.linspace(240, 40, data.shape[0], dtype=np.uint8)[:, None]
    data[..., 2] = 120
    Image.fromarray(data).save(image_path)

    cfg = load_config("configs/is3d_v0.yaml")
    pipeline = IS3DPipeline(cfg)
    result = pipeline.generate(image_path, output_path, tier="tier2")

    assert output_path.exists()
    assert result.num_gaussians > 0
    assert result.model_backend in {"torch", "fallback"}
    assert isinstance(result.loaded_weights, dict)
    assert {"fastvit", "triplane", "completion"} <= set(result.loaded_weights)

    with output_path.open("r", encoding="utf-8") as handle:
        content = handle.read()
    assert "element vertex" in content
    assert "end_header" in content


def test_pipeline_resizes_large_input_before_generation(tmp_path: Path) -> None:
    image_path = tmp_path / "large_input.png"
    output_path = tmp_path / "scene_large.ply"

    data = np.zeros((256, 512, 3), dtype=np.uint8)
    data[..., 0] = 200
    data[..., 1] = 80
    data[..., 2] = 40
    Image.fromarray(data).save(image_path)

    cfg = load_config("configs/is3d_v0.yaml")
    cfg.model_runtime.max_input_edge = 64
    pipeline = IS3DPipeline(cfg)
    result = pipeline.generate(image_path, output_path, tier="tier2")

    # 64x32 with stride=4 gives <=128 visible points and bounded completion expansion.
    assert result.num_gaussians <= 200


def test_pipeline_exports_navmesh_and_meta_sidecars(tmp_path: Path) -> None:
    image_path = tmp_path / "input_sidecar.png"
    output_path = tmp_path / "scene_sidecar.ply"

    data = np.zeros((80, 80, 3), dtype=np.uint8)
    data[..., 0] = 180
    data[..., 1] = 120
    data[..., 2] = 80
    Image.fromarray(data).save(image_path)

    cfg = load_config("configs/is3d_v0.yaml")
    pipeline = IS3DPipeline(cfg)
    result = pipeline.generate(
        image_path,
        output_path,
        tier="tier2",
        export_navmesh=True,
        export_meta=True,
    )

    assert result.navmesh_path is not None
    assert result.meta_path is not None
    assert result.navmesh_path.exists()
    assert result.meta_path.exists()

    with result.navmesh_path.open("r", encoding="utf-8") as handle:
        navmesh = json.load(handle)
    with result.meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    assert "walkable_cells" in navmesh
    assert "gaussian_count" in meta
