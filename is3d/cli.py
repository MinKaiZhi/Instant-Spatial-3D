from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from is3d.config import load_config
from is3d.pipeline import IS3DPipeline

app = typer.Typer(help="Instant-Spatial-3D pipeline CLI")
console = Console()


@app.command()
def generate(
    input: Path = typer.Option(..., exists=True, readable=True, help="Input RGB image path"),
    output: Path = typer.Option(Path("outputs/scene.ply"), help="Output PLY path"),
    config: Path = typer.Option(Path("configs/is3d_v0.yaml"), exists=True, readable=True, help="Pipeline config YAML"),
    tier: str | None = typer.Option(None, help="Hardware tier override (tier1/tier2/tier3)"),
    export_navmesh: bool = typer.Option(True, help="Export roaming navmesh JSON sidecar"),
    export_meta: bool = typer.Option(True, help="Export scene metadata JSON sidecar"),
    navmesh_output: Path | None = typer.Option(None, help="Optional navmesh output path (.json)"),
    meta_output: Path | None = typer.Option(None, help="Optional scene metadata output path (.json)"),
    device: str | None = typer.Option(None, help="Model device override: auto/cpu/cuda/mps"),
    fastvit_ckpt: Path | None = typer.Option(None, help="FastViT encoder checkpoint (.pt/.pth)"),
    triplane_ckpt: Path | None = typer.Option(None, help="Tri-plane regressor checkpoint (.pt/.pth)"),
    completion_ckpt: Path | None = typer.Option(None, help="Completion decoder checkpoint (.pt/.pth)"),
) -> None:
    cfg = load_config(config)

    if device is not None:
        cfg.model_runtime.device = device
    if fastvit_ckpt is not None:
        cfg.model_runtime.weights.fastvit = str(fastvit_ckpt)
    if triplane_ckpt is not None:
        cfg.model_runtime.weights.triplane = str(triplane_ckpt)
    if completion_ckpt is not None:
        cfg.model_runtime.weights.completion = str(completion_ckpt)

    pipeline = IS3DPipeline(cfg)
    result = pipeline.generate(
        input_image_path=input,
        output_path=output,
        tier=tier,
        export_navmesh=export_navmesh,
        export_meta=export_meta,
        navmesh_output_path=navmesh_output,
        meta_output_path=meta_output,
    )

    table = Table(title="IS3D Generation Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output", str(result.output_path))
    table.add_row("Tier", result.tier_name)
    table.add_row("Gaussians", str(result.num_gaussians))
    table.add_row("model_backend", result.model_backend)
    table.add_row("model_device", result.model_device)
    table.add_row("model_reason", result.model_reason)
    table.add_row(
        "loaded_weights",
        ", ".join([f"{k}={int(v)}" for k, v in result.loaded_weights.items()]),
    )
    table.add_row("navmesh", str(result.navmesh_path) if result.navmesh_path is not None else "disabled")
    table.add_row("meta", str(result.meta_path) if result.meta_path is not None else "disabled")

    total_ms = 0.0
    for key in ["t1_preprocess", "t2_inference", "t3_resource_build", "t4_activation"]:
        elapsed = result.stage_ms.get(key, 0.0)
        total_ms += elapsed
        table.add_row(key, f"{elapsed:.2f} ms")
    table.add_row("total", f"{total_ms:.2f} ms")

    if result.budget_warnings:
        table.add_row("warnings", " | ".join(result.budget_warnings))
    else:
        table.add_row("warnings", "none")

    console.print(table)


@app.command()
def version() -> None:
    console.print("instant-spatial-3d 0.1.0")


if __name__ == "__main__":
    app()
