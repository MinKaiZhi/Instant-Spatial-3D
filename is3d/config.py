from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class LatencyBudgetMs:
    t1_preprocess: float = 100.0
    t2_inference: float = 600.0
    t3_resource_build: float = 300.0
    t4_activation: float = 150.0


@dataclass(slots=True)
class PipelineSettings:
    sample_stride: int = 4
    min_depth_m: float = 0.6
    max_depth_m: float = 5.0
    gaussian_scale_ratio: float = 0.015
    completion_ratio: float = 0.22
    completion_opacity_gain: float = 0.65
    depth_smoothing: float = 0.35
    navmesh_cell_size_m: float = 0.2
    navmesh_height_tolerance_m: float = 0.25
    navmesh_min_points_per_cell: int = 2


@dataclass(slots=True)
class TierSpec:
    name: str
    zero_copy: bool
    triplane_enabled: bool


@dataclass(slots=True)
class HardwareTiers:
    default: str
    tiers: dict[str, TierSpec]

    def resolve(self, tier_name: str | None) -> TierSpec:
        key = tier_name or self.default
        if key not in self.tiers:
            available = ", ".join(sorted(self.tiers))
            raise KeyError(f"Unknown tier '{key}', expected one of: {available}")
        return self.tiers[key]


@dataclass(slots=True)
class ModelWeights:
    fastvit: str | None = None
    triplane: str | None = None
    completion: str | None = None


@dataclass(slots=True)
class ModelRuntimeSettings:
    enabled: bool = True
    backend: str = "torch"
    device: str = "auto"
    dtype: str = "float32"
    strict_load: bool = False
    allow_uninitialized: bool = True
    use_completion_decoder: bool = True
    seed: int | None = 42

    encoder_embed_dim: int = 192
    regressor_latent_dim: int = 128
    decoder_model_dim: int = 128
    decoder_heads: int = 8
    decoder_layers: int = 2
    max_input_edge: int = 1280
    max_visible_points: int = 120000
    max_completion_queries: int = 4096
    max_completion_memory_tokens: int = 8192
    disable_completion_on_mps: bool = True

    weights: ModelWeights = field(default_factory=ModelWeights)


@dataclass(slots=True)
class IS3DConfig:
    latency_budget_ms: LatencyBudgetMs
    pipeline: PipelineSettings
    hardware_tiers: HardwareTiers
    model_runtime: ModelRuntimeSettings


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return loaded


def load_config(path: str | Path) -> IS3DConfig:
    config_path = Path(path)
    data = _read_yaml(config_path)

    latency = LatencyBudgetMs(**data.get("latency_budget_ms", {}))
    pipeline = PipelineSettings(**data.get("pipeline", {}))

    hardware_data = data.get("hardware_tier", {})
    tiers_data = hardware_data.get("tiers", {})
    tiers = {name: TierSpec(**spec) for name, spec in tiers_data.items()}
    if not tiers:
        raise ValueError("At least one hardware tier must be configured")
    hardware_tiers = HardwareTiers(default=hardware_data.get("default", next(iter(tiers))), tiers=tiers)

    model_data = data.get("model_runtime", {})
    weights = ModelWeights(**model_data.get("weights", {}))
    model_runtime = ModelRuntimeSettings(
        enabled=model_data.get("enabled", True),
        backend=model_data.get("backend", "torch"),
        device=model_data.get("device", "auto"),
        dtype=model_data.get("dtype", "float32"),
        strict_load=model_data.get("strict_load", False),
        allow_uninitialized=model_data.get("allow_uninitialized", True),
        use_completion_decoder=model_data.get("use_completion_decoder", True),
        seed=model_data.get("seed", 42),
        encoder_embed_dim=model_data.get("encoder_embed_dim", 192),
        regressor_latent_dim=model_data.get("regressor_latent_dim", 128),
        decoder_model_dim=model_data.get("decoder_model_dim", 128),
        decoder_heads=model_data.get("decoder_heads", 8),
        decoder_layers=model_data.get("decoder_layers", 2),
        max_input_edge=model_data.get("max_input_edge", 1280),
        max_visible_points=model_data.get("max_visible_points", 120000),
        max_completion_queries=model_data.get("max_completion_queries", 4096),
        max_completion_memory_tokens=model_data.get("max_completion_memory_tokens", 8192),
        disable_completion_on_mps=model_data.get("disable_completion_on_mps", True),
        weights=weights,
    )

    return IS3DConfig(
        latency_budget_ms=latency,
        pipeline=pipeline,
        hardware_tiers=hardware_tiers,
        model_runtime=model_runtime,
    )
