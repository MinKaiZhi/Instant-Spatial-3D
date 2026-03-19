from __future__ import annotations

import torch
from torch import nn


class TriPlaneRegressor(nn.Module):
    """Tri-plane style regression heads for depth and gaussian attributes."""

    def __init__(self, channels: int = 192, latent_dim: int = 128) -> None:
        super().__init__()

        self.xy_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.xz_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.yz_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        self.depth_head = nn.Conv2d(channels, 1, 1)
        self.opacity_head = nn.Conv2d(channels, 1, 1)
        self.scale_head = nn.Conv2d(channels, 3, 1)
        self.rotation_head = nn.Conv2d(channels, 3, 1)
        self.latent_head = nn.Conv2d(channels, latent_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        xy = self.xy_branch(features)
        xz = self.xz_branch(features)
        yz = self.yz_branch(features)

        fused = self.fusion(torch.cat([xy, xz, yz], dim=1))

        depth_norm = torch.sigmoid(self.depth_head(fused))
        opacity = torch.sigmoid(self.opacity_head(fused))
        scale_raw = self.scale_head(fused)
        rotation_xyz = torch.tanh(self.rotation_head(fused))
        latent = self.latent_head(fused)

        return {
            "depth_norm": depth_norm,
            "opacity": opacity,
            "scale_raw": scale_raw,
            "rotation_xyz": rotation_xyz,
            "latent": latent,
        }
