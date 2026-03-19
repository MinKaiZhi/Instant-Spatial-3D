# Instant-Spatial-3D

Instant-Spatial-3D (IS3D) is a cross-platform generator targeting:
- single-image to metric-scale 3D Gaussian scene export (`.ply`)
- fast time-to-interact pipeline framing (T1/T2/T3/T4)
- roaming-ready output with future full generative completion

## Current Backend Status

The backend now supports a model-driven runtime with checkpoint loading:
- `FastViTEncoder` interface
- `TriPlaneRegressor` interface
- `CompletionDecoder` interface
- checkpoint wiring for each module (`.pt` / `.pth`)

When model runtime is unavailable (for example, PyTorch not installed), pipeline falls back to deterministic geometric synthesis so CLI remains usable.

## Environments

Base (no torch required):

```bash
conda env create -f environment.yml
conda activate is3d
```

ML runtime (includes torch):

```bash
conda env create -f environment.ml.yml
conda activate is3d-ml
```

## Quick Validate

```bash
pytest
```

## Generate `.ply`

Without checkpoints (uninitialized model or fallback path):

```bash
is3d generate \
  --input /absolute/path/to/input.jpg \
  --output outputs/scene.ply \
  --config configs/is3d_v0.yaml \
  --tier tier2
```

By default the command also exports:
- `outputs/scene.navmesh.json` (roaming walkable grid)
- `outputs/scene.meta.json` (scene bounds and gaussian statistics)

With explicit checkpoints:

```bash
is3d generate \
  --input /absolute/path/to/input.jpg \
  --output outputs/scene.ply \
  --config configs/is3d_v0.yaml \
  --fastvit-ckpt /abs/ckpt/fastvit.pt \
  --triplane-ckpt /abs/ckpt/triplane.pt \
  --completion-ckpt /abs/ckpt/completion.pt \
  --device auto
```

Important:
- Do not use placeholder brackets like `<fastvit.pt>` in shell commands.
- Use real file paths, for example `/abs/checkpoints/fastvit.pt`.

CLI output includes:
- `model_backend`
- `model_device`
- `model_reason`
- `loaded_weights` (fastvit/triplane/completion)
- `navmesh` and `meta` sidecar paths

## Troubleshooting

- `zsh: no such file or directory: fastvit.pt`
  - Cause: using `<fastvit.pt>` in zsh is treated as I/O redirection.
  - Fix: remove `< >` and pass real path values.

- `MPSNDArray ... failed assertion ... abort`
  - Cause: Metal/MPS transformer path can abort on large token sizes.
  - Current mitigation in this repo:
    - auto resize input to `max_input_edge` (default 1280)
    - cap visible points and completion token/query counts
    - disable completion decoder on `mps` by default and fallback to heuristic completion

## Project Layout

- `is3d/`: pipeline, stages, exporter, model runtime, math utilities
- `configs/`: defaults including model runtime and checkpoint fields
- `tests/`: smoke and math tests
- `clients/`: cross-platform renderer placeholders
- `scripts/`: bootstrap helpers
