"""Advanced architectures for dispersion curve modeling."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def sinusoidal_position_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-torch.log(torch.tensor(10000.0)) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def soft_argmax_velocities(logits: torch.Tensor, velocity_axis: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Differentiable velocity picks from heatmaps."""

    if velocity_axis.dim() == 1:
        if logits.shape[2] != velocity_axis.numel():
            raise ValueError("Velocity axis length must match logits velocity dimension")
        coords = velocity_axis.view(1, 1, -1, 1)
    elif velocity_axis.dim() == 2:
        if logits.shape[0] != velocity_axis.shape[0] or logits.shape[2] != velocity_axis.shape[1]:
            raise ValueError("Velocity axis shape must match batch and velocity dims")
        coords = velocity_axis.unsqueeze(1).unsqueeze(-1)
    else:
        raise ValueError("velocity_axis must be 1D or 2D tensor")

    probs = torch.softmax(logits / max(temperature, 1e-6), dim=2)
    velocities = (probs * coords).sum(dim=2)
    return velocities


class ConvEncoder(nn.Module):
    """Simple CNN encoder that downsamples spectra."""

    def __init__(self, in_channels: int, base_channels: int = 32, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
                )
            )
            channels = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrequencySequenceModel(nn.Module):
    """CNN/Transformer encoder with LSTM decoder predicting per-frequency velocities."""

    def __init__(
        self,
        in_channels: int,
        mode_count: int,
        hidden_dim: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mode_count = mode_count
        self.encoder = ConvEncoder(in_channels, base_channels=hidden_dim // 8, depth=3, dropout=dropout)
        self.proj = nn.Conv2d(self.encoder.out_channels, hidden_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=transformer_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, dropout=dropout, batch_first=True)
        self.velocity_head = nn.Linear(hidden_dim, mode_count)
        self.mask_head = nn.Linear(hidden_dim, mode_count)

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.encoder(x)  # (B, C, H, W)
        feats = self.proj(feats)
        seq = feats.mean(dim=2)  # average over velocity dimension -> (B, hidden, W)
        seq = seq.permute(2, 0, 1)  # (W, B, hidden)
        pe = sinusoidal_position_encoding(seq.size(0), seq.size(2), seq.device)
        seq = seq + pe.unsqueeze(1)
        trans_out = self.transformer(seq)
        trans_out = trans_out.permute(1, 0, 2)  # (B, W, hidden)
        lstm_out, _ = self.lstm(trans_out)
        velocity_tracks = self.velocity_head(lstm_out).permute(0, 2, 1)  # (B, modes, W)
        mask_logits = self.mask_head(lstm_out).permute(0, 2, 1)
        return {
            "velocity_tracks": velocity_tracks,
            "sequence_features": lstm_out,
            "mask_logits": mask_logits,
        }


class UpConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GraphCurveModel(nn.Module):
    """Graph-style model producing heatmaps for differentiable curve picking."""

    def __init__(
        self,
        in_channels: int,
        mode_count: int,
        base_channels: int = 32,
        depth: int = 4,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        encoder_layers = []
        channels = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            channels = out_ch
            if i < depth - 1:
                encoder_layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(depth - 2, -1, -1):
            out_ch = base_channels * (2 ** i)
            decoder_layers.append(UpConvBlock(channels, out_ch))
            channels = out_ch
        self.decoder = nn.Sequential(*decoder_layers)
        self.head = nn.Conv2d(channels, mode_count, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encoder(x)
        z = self.decoder(z)
        heatmap_logits = self.head(z)
        return {"heatmap_logits": heatmap_logits}

    def pick(self, logits: torch.Tensor, velocity_axis: torch.Tensor) -> torch.Tensor:
        return soft_argmax_velocities(logits, velocity_axis, self.temperature)
