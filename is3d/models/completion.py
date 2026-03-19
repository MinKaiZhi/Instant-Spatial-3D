from __future__ import annotations

import torch
from torch import nn


class CompletionDecoder(nn.Module):
    """Transformer decoder for backside / occlusion gaussian synthesis."""

    def __init__(
        self,
        memory_dim: int = 128,
        query_dim: int = 10,
        model_dim: int = 128,
        heads: int = 8,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.memory_proj = nn.Linear(memory_dim, model_dim)
        self.query_proj = nn.Linear(query_dim, model_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)

        self.delta_xyz = nn.Linear(model_dim, 3)
        self.delta_color = nn.Linear(model_dim, 3)
        self.delta_scale = nn.Linear(model_dim, 3)
        self.rotation_xyz = nn.Linear(model_dim, 3)
        self.opacity_logit = nn.Linear(model_dim, 1)

    def forward(self, memory_tokens: torch.Tensor, query_features: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = self.memory_proj(memory_tokens)
        query = self.query_proj(query_features)
        decoded = self.decoder(tgt=query, memory=memory)

        return {
            "delta_xyz": self.delta_xyz(decoded),
            "delta_color": self.delta_color(decoded),
            "delta_scale": self.delta_scale(decoded),
            "rotation_xyz": torch.tanh(self.rotation_xyz(decoded)),
            "opacity_logit": self.opacity_logit(decoded),
        }
