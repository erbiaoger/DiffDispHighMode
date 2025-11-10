"""Lightweight UNet-style model for dispersion segmentation."""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: Iterable[int], dropout: float = 0.0):
        super().__init__()
        feats = list(features)
        if not feats:
            raise ValueError("features list must be non-empty")

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels
        for feat in feats:
            self.downs.append(DoubleConv(current_channels, feat, dropout=dropout))
            current_channels = feat

        self.bottleneck = DoubleConv(current_channels, current_channels * 2, dropout=dropout)
        current_channels = current_channels * 2

        for feat in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(current_channels, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feat * 2, feat, dropout=dropout))
            current_channels = feat

        self.final_conv = nn.Conv2d(feats[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        skip_connections: List[torch.Tensor] = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)
