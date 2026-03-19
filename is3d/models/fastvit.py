from __future__ import annotations

import torch
from torch import nn


class FastViTEncoder(nn.Module):
    """Compact FastViT-like encoder interface for IS3D inference wiring."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 192) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )

        self.mixer = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.mixer(x)
        return x
