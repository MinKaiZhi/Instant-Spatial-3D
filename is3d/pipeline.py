from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from is3d.config import IS3DConfig, LatencyBudgetMs
from is3d.exporters.navmesh import NavMeshConfig, build_navmesh, write_navmesh_json
from is3d.exporters.ply import write_gaussian_ply
from is3d.exporters.scene_meta import build_scene_meta, write_scene_meta_json
from is3d.models.runtime import ModelRuntimeInfo, TorchModelRuntime
from is3d.stages.completion import hallucinate_backside
from is3d.stages.depth_scale import estimate_intrinsics, estimate_metric_depth
from is3d.stages.gaussian_regression import regress_gaussians
from is3d.stages.perception import extract_features, preprocess_image


@dataclass(slots=True)
class PipelineResult:
    output_path: Path
    tier_name: str
    num_gaussians: int
    stage_ms: dict[str, float]
    budget_warnings: list[str]
    model_backend: str
    model_device: str
    model_reason: str
    loaded_weights: dict[str, bool]
    navmesh_path: Path | None
    meta_path: Path | None


class IS3DPipeline:
    def __init__(self, config: IS3DConfig) -> None:
        self.config = config
        self._model_runtime: TorchModelRuntime | None = None
        self._runtime_info = ModelRuntimeInfo(
            active=False,
            backend="fallback",
            device="cpu",
            reason="not_initialized",
            loaded_weights={"fastvit": False, "triplane": False, "completion": False},
        )

        try:
            runtime = TorchModelRuntime(config.model_runtime)
            self._model_runtime = runtime
            self._runtime_info = runtime.info
        except Exception as exc:
            self._runtime_info = ModelRuntimeInfo(
                active=False,
                backend="fallback",
                device="cpu",
                reason=f"runtime_init_error: {exc}",
                loaded_weights={"fastvit": False, "triplane": False, "completion": False},
            )

    def generate(
        self,
        input_image_path: str | Path,
        output_path: str | Path,
        tier: str | None = None,
        export_navmesh: bool = False,
        export_meta: bool = False,
        navmesh_output_path: str | Path | None = None,
        meta_output_path: str | Path | None = None,
    ) -> PipelineResult:
        resolved_tier = self.config.hardware_tiers.resolve(tier)
        stage_ms: dict[str, float] = {}
        runtime_info = self._runtime_info

        t1_start = perf_counter()
        image = self._load_image_rgb(input_image_path)
        image = _resize_image_if_needed(image, self.config.model_runtime.max_input_edge)
        image_f32, semantic_mask = preprocess_image(image)
        stage_ms["t1_preprocess"] = (perf_counter() - t1_start) * 1000.0

        t2_start = perf_counter()
        height, width, _ = image_f32.shape
        intrinsics = estimate_intrinsics(height, width)

        cloud = None
        memory_tokens: np.ndarray | None = None

        if self._model_runtime is not None and self._model_runtime.is_active:
            try:
                cloud, memory_tokens = self._model_runtime.infer_visible_gaussians(
                    image_f32=image_f32,
                    intrinsics=intrinsics,
                    settings=self.config.pipeline,
                )
                runtime_info = self._model_runtime.info
            except Exception as exc:
                runtime_info = ModelRuntimeInfo(
                    active=False,
                    backend="fallback",
                    device=runtime_info.device,
                    reason=f"model_infer_failed: {exc}",
                    loaded_weights=runtime_info.loaded_weights,
                )

        if cloud is None:
            features = extract_features(image_f32, semantic_mask)
            depth_m = estimate_metric_depth(features, self.config.pipeline)
            cloud = regress_gaussians(image_f32, depth_m, intrinsics, self.config.pipeline)

        stage_ms["t2_inference"] = (perf_counter() - t2_start) * 1000.0

        t3_start = perf_counter()
        if resolved_tier.triplane_enabled:
            used_model_completion = (
                runtime_info.active
                and self._model_runtime is not None
                and memory_tokens is not None
                and self.config.model_runtime.use_completion_decoder
                and runtime_info.loaded_weights.get("completion", False)
                and not (
                    self.config.model_runtime.disable_completion_on_mps
                    and runtime_info.device == "mps"
                )
            )
            if used_model_completion:
                try:
                    cloud = self._model_runtime.infer_completion(cloud, memory_tokens, self.config.pipeline)
                except Exception as exc:
                    runtime_info = ModelRuntimeInfo(
                        active=False,
                        backend="fallback",
                        device=runtime_info.device,
                        reason=f"completion_infer_failed: {exc}",
                        loaded_weights=runtime_info.loaded_weights,
                    )
                    cloud = hallucinate_backside(cloud, self.config.pipeline)
            else:
                cloud = hallucinate_backside(cloud, self.config.pipeline)

        stage_ms["t3_resource_build"] = (perf_counter() - t3_start) * 1000.0

        t4_start = perf_counter()
        write_gaussian_ply(output_path, cloud)
        stage_ms["t4_activation"] = (perf_counter() - t4_start) * 1000.0

        navmesh_path = None
        meta_path = None
        output_base = Path(output_path)
        if export_navmesh:
            navmesh_cfg = NavMeshConfig(
                cell_size_m=self.config.pipeline.navmesh_cell_size_m,
                height_tolerance_m=self.config.pipeline.navmesh_height_tolerance_m,
                min_points_per_cell=self.config.pipeline.navmesh_min_points_per_cell,
            )
            navmesh = build_navmesh(cloud, navmesh_cfg)
            navmesh_path = Path(navmesh_output_path) if navmesh_output_path is not None else output_base.with_suffix(".navmesh.json")
            write_navmesh_json(navmesh_path, navmesh)

        if export_meta:
            meta = build_scene_meta(cloud, resolved_tier.name)
            meta_path = Path(meta_output_path) if meta_output_path is not None else output_base.with_suffix(".meta.json")
            write_scene_meta_json(meta_path, meta)

        warnings = _build_budget_warnings(stage_ms, self.config.latency_budget_ms)

        return PipelineResult(
            output_path=Path(output_path),
            tier_name=resolved_tier.name,
            num_gaussians=cloud.count,
            stage_ms=stage_ms,
            budget_warnings=warnings,
            model_backend=runtime_info.backend,
            model_device=runtime_info.device,
            model_reason=runtime_info.reason,
            loaded_weights=runtime_info.loaded_weights,
            navmesh_path=navmesh_path,
            meta_path=meta_path,
        )

    @staticmethod
    def _load_image_rgb(path: str | Path) -> np.ndarray:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def _build_budget_warnings(stage_ms: dict[str, float], budget: LatencyBudgetMs) -> list[str]:
    warnings: list[str] = []
    for key, limit in [
        ("t1_preprocess", budget.t1_preprocess),
        ("t2_inference", budget.t2_inference),
        ("t3_resource_build", budget.t3_resource_build),
        ("t4_activation", budget.t4_activation),
    ]:
        elapsed = stage_ms.get(key, 0.0)
        if elapsed > limit:
            warnings.append(f"{key} exceeded budget ({elapsed:.2f}ms > {limit:.2f}ms)")
    return warnings


def _resize_image_if_needed(image: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge <= 0:
        return image

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image

    scale = float(max_edge) / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    pil = Image.fromarray(image, mode="RGB")
    resized = pil.resize((new_width, new_height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)
