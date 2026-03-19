from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from is3d.config import ModelRuntimeSettings, PipelineSettings
from is3d.types import CameraIntrinsics, GaussianCloud


@dataclass(slots=True)
class ModelRuntimeInfo:
    active: bool
    backend: str
    device: str
    reason: str
    loaded_weights: dict[str, bool]


class TorchModelRuntime:
    def __init__(self, settings: ModelRuntimeSettings) -> None:
        self.settings = settings
        self._torch = None
        self._device = "cpu"
        self._dtype_name = "float32"
        self._active = False
        self._reason = ""
        self._loaded_weights: dict[str, bool] = {
            "fastvit": False,
            "triplane": False,
            "completion": False,
        }

        self.encoder: Any = None
        self.regressor: Any = None
        self.decoder: Any = None

        self._init_runtime()

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def info(self) -> ModelRuntimeInfo:
        return ModelRuntimeInfo(
            active=self._active,
            backend="torch" if self._torch is not None else "fallback",
            device=self._device,
            reason=self._reason,
            loaded_weights=dict(self._loaded_weights),
        )

    def _init_runtime(self) -> None:
        if not self.settings.enabled:
            self._reason = "disabled_by_config"
            return

        try:
            import torch
        except ImportError:
            self._reason = "torch_not_installed"
            return

        self._torch = torch
        self._device = _resolve_device(torch, self.settings.device)
        self._dtype_name = self.settings.dtype
        torch_dtype = _resolve_dtype(torch, self.settings.dtype)

        if self.settings.seed is not None:
            torch.manual_seed(int(self.settings.seed))

        from is3d.models.completion import CompletionDecoder
        from is3d.models.fastvit import FastViTEncoder
        from is3d.models.triplane import TriPlaneRegressor

        self.encoder = FastViTEncoder(embed_dim=self.settings.encoder_embed_dim).to(self._device, dtype=torch_dtype).eval()
        self.regressor = TriPlaneRegressor(
            channels=self.settings.encoder_embed_dim,
            latent_dim=self.settings.regressor_latent_dim,
        ).to(self._device, dtype=torch_dtype).eval()

        self.decoder = CompletionDecoder(
            memory_dim=self.settings.regressor_latent_dim,
            query_dim=10,
            model_dim=self.settings.decoder_model_dim,
            heads=self.settings.decoder_heads,
            layers=self.settings.decoder_layers,
        ).to(self._device, dtype=torch_dtype).eval()

        strict = self.settings.strict_load
        allow_uninitialized = self.settings.allow_uninitialized

        self._loaded_weights["fastvit"] = _load_weights(
            torch,
            module=self.encoder,
            path=self.settings.weights.fastvit,
            strict=strict,
            required=not allow_uninitialized,
        )
        self._loaded_weights["triplane"] = _load_weights(
            torch,
            module=self.regressor,
            path=self.settings.weights.triplane,
            strict=strict,
            required=not allow_uninitialized,
        )
        self._loaded_weights["completion"] = _load_weights(
            torch,
            module=self.decoder,
            path=self.settings.weights.completion,
            strict=strict,
            required=False,
        )

        self._active = True
        loaded_count = sum(1 for loaded in self._loaded_weights.values() if loaded)
        self._reason = f"initialized ({loaded_count}/3 checkpoints loaded)"

    def infer_visible_gaussians(
        self,
        image_f32: np.ndarray,
        intrinsics: CameraIntrinsics,
        settings: PipelineSettings,
    ) -> tuple[GaussianCloud, np.ndarray]:
        if not self._active or self._torch is None:
            raise RuntimeError("Torch runtime is not active")

        torch = self._torch
        torch_dtype = _resolve_dtype(torch, self._dtype_name)

        image_tensor = torch.from_numpy(image_f32).permute(2, 0, 1).unsqueeze(0).to(self._device, dtype=torch_dtype)

        with torch.inference_mode():
            encoded = self.encoder(image_tensor)
            pred = self.regressor(encoded)

            height, width = image_f32.shape[:2]
            depth_norm = _upsample_map(torch, pred["depth_norm"], (height, width)).squeeze(0).squeeze(0)
            opacity_map = _upsample_map(torch, pred["opacity"], (height, width)).squeeze(0).squeeze(0)
            scale_map = _upsample_map(torch, pred["scale_raw"], (height, width)).squeeze(0).permute(1, 2, 0)
            rot_xyz_map = _upsample_map(torch, pred["rotation_xyz"], (height, width)).squeeze(0).permute(1, 2, 0)

            latent_tokens = pred["latent"].flatten(2).transpose(1, 2).squeeze(0)

        depth_m = settings.min_depth_m + (settings.max_depth_m - settings.min_depth_m) * depth_norm.clamp(0.0, 1.0)
        depth_np = depth_m.detach().cpu().numpy().astype(np.float32)

        stride = max(1, int(settings.sample_stride))
        ys = np.arange(0, image_f32.shape[0], stride)
        xs = np.arange(0, image_f32.shape[1], stride)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")

        y_idx = grid_y.reshape(-1)
        x_idx = grid_x.reshape(-1)
        y_idx, x_idx = _cap_indices(y_idx, x_idx, self.settings.max_visible_points)

        z = depth_np[y_idx, x_idx]
        x = ((x_idx.astype(np.float32) - intrinsics.cx) / intrinsics.fx) * z
        y = ((y_idx.astype(np.float32) - intrinsics.cy) / intrinsics.fy) * z
        xyz = np.stack([x, y, z], axis=-1).astype(np.float32)

        scale_raw_np = scale_map.detach().cpu().numpy().astype(np.float32)
        scale = _to_positive_scale(scale_raw_np[y_idx, x_idx], settings.gaussian_scale_ratio)

        opacity_np = opacity_map.detach().cpu().numpy().astype(np.float32)
        opacity = np.clip(opacity_np[y_idx, x_idx], 0.02, 0.99)

        rot_xyz_np = rot_xyz_map.detach().cpu().numpy().astype(np.float32)
        rotation = _compose_rotation(rot_xyz_np[y_idx, x_idx])

        color = np.clip(image_f32[y_idx, x_idx], 0.0, 1.0).astype(np.float32)
        synthetic_mask = np.zeros((xyz.shape[0],), dtype=bool)

        cloud = GaussianCloud(
            xyz=xyz,
            scale=scale,
            rotation_xyzw=rotation,
            opacity=opacity,
            color=color,
            synthetic_mask=synthetic_mask,
        )

        memory_tokens = latent_tokens.detach().cpu().numpy().astype(np.float32)
        if memory_tokens.shape[0] > self.settings.max_completion_memory_tokens:
            keep = np.linspace(
                0,
                memory_tokens.shape[0] - 1,
                num=self.settings.max_completion_memory_tokens,
                dtype=np.int64,
            )
            memory_tokens = memory_tokens[keep]
        return cloud, memory_tokens

    def infer_completion(self, cloud: GaussianCloud, memory_tokens: np.ndarray, settings: PipelineSettings) -> GaussianCloud:
        if not self._active or self._torch is None:
            return cloud
        if not self.settings.use_completion_decoder:
            return cloud
        if self.settings.disable_completion_on_mps and self._device == "mps":
            return cloud
        if cloud.count == 0:
            return cloud

        ratio = float(np.clip(settings.completion_ratio, 0.0, 1.0))
        if ratio <= 0.0:
            return cloud

        extra_count = int(max(1, round(cloud.count * ratio)))
        extra_count = min(extra_count, cloud.count)
        extra_count = min(extra_count, self.settings.max_completion_queries)

        anchor_idx = np.argsort(cloud.xyz[:, 2])[-extra_count:]
        queries = np.concatenate(
            [
                cloud.xyz[anchor_idx],
                cloud.color[anchor_idx],
                cloud.opacity[anchor_idx, None],
                cloud.scale[anchor_idx],
            ],
            axis=1,
        ).astype(np.float32)

        torch = self._torch
        torch_dtype = _resolve_dtype(torch, self._dtype_name)

        with torch.inference_mode():
            memory = torch.from_numpy(memory_tokens).unsqueeze(0).to(self._device, dtype=torch_dtype)
            query = torch.from_numpy(queries).unsqueeze(0).to(self._device, dtype=torch_dtype)
            decoded = self.decoder(memory, query)

        delta_xyz = decoded["delta_xyz"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        delta_color = decoded["delta_color"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        delta_scale = decoded["delta_scale"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        rot_xyz = decoded["rotation_xyz"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        opacity = decoded["opacity_logit"].squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)

        xyz = cloud.xyz[anchor_idx] + delta_xyz * 0.2
        xyz[:, 0] = -xyz[:, 0]

        color = np.clip(cloud.color[anchor_idx] + delta_color * 0.1, 0.0, 1.0)
        scale = np.clip(cloud.scale[anchor_idx] * np.exp(delta_scale * 0.2), 0.002, 0.15)
        rotation = _compose_rotation(rot_xyz)

        opacity_sigmoid = 1.0 / (1.0 + np.exp(-opacity))
        opacity_out = np.clip(opacity_sigmoid * settings.completion_opacity_gain, 0.02, 0.9)

        extra = GaussianCloud(
            xyz=xyz,
            scale=scale.astype(np.float32),
            rotation_xyzw=rotation,
            opacity=opacity_out.astype(np.float32),
            color=color.astype(np.float32),
            synthetic_mask=np.ones((extra_count,), dtype=bool),
        )
        return cloud.concat(extra)


def _resolve_device(torch: Any, device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _resolve_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _load_weights(torch: Any, module: Any, path: str | None, strict: bool, required: bool) -> bool:
    if not path:
        if required:
            raise FileNotFoundError("Required checkpoint path is empty")
        return False

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        if required:
            raise FileNotFoundError(f"Required checkpoint not found: {checkpoint_path}")
        return False

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint structure: {checkpoint_path}")

    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state[key[7:]] = value
        else:
            cleaned_state[key] = value

    module.load_state_dict(cleaned_state, strict=strict)
    return True


def _upsample_map(torch: Any, tensor: Any, size: tuple[int, int]) -> Any:
    return torch.nn.functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


def _to_positive_scale(scale_raw: np.ndarray, ratio: float) -> np.ndarray:
    positive = np.log1p(np.exp(scale_raw)) * ratio
    return np.clip(positive, 0.002, 0.15).astype(np.float32)


def _compose_rotation(rot_xyz: np.ndarray) -> np.ndarray:
    xyz = np.clip(rot_xyz, -1.0, 1.0).astype(np.float32)
    xyz_norm_sq = np.sum(xyz * xyz, axis=1, keepdims=True)
    safe = np.maximum(0.0, 1.0 - xyz_norm_sq)
    w = np.sqrt(safe).astype(np.float32)
    quat = np.concatenate([xyz, w], axis=1)

    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return (quat / norm).astype(np.float32)


def _cap_indices(y_idx: np.ndarray, x_idx: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0:
        return y_idx, x_idx
    count = y_idx.shape[0]
    if count <= max_points:
        return y_idx, x_idx

    keep = np.linspace(0, count - 1, num=max_points, dtype=np.int64)
    return y_idx[keep], x_idx[keep]
