from is3d.config import load_config


def test_model_runtime_config_defaults() -> None:
    cfg = load_config("configs/is3d_v0.yaml")

    assert cfg.model_runtime.backend == "torch"
    assert cfg.model_runtime.weights.fastvit is None
    assert cfg.model_runtime.weights.triplane is None
    assert cfg.model_runtime.weights.completion is None
    assert cfg.model_runtime.encoder_embed_dim > 0
