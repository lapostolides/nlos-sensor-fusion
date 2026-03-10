"""
training/encoders.py - Per-sensor encoder modules.

Each encoder maps raw sensor data to a fixed-size embedding vector (B, embed_dim).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SPADEncoder(nn.Module):
    """
    Encodes SPAD histogram volumes into a fixed embedding.

    Input:  (B, H, W, bins) float32 — raw histogram counts.
    Output: (B, embed_dim) float32.

    Treats histogram bins as channels and spatial pixels (4×4 or 8×8)
    as the 2-D feature map.  AdaptiveAvgPool2d makes it resolution-agnostic.
    """

    def __init__(self, in_bins: int = 8, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_bins, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )

    def forward(self, spad: torch.Tensor) -> torch.Tensor:
        # (B, H, W, bins) → (B, bins, H, W)
        x = spad.permute(0, 3, 1, 2)
        return self.net(x)


class RGBDEncoder(nn.Module):
    """
    Encodes RGB-D images using a ResNet-18 backbone.

    Input:  rgb  (B, 3, H, W) float32 [0, 1]
            depth (B, H, W)   float32 metres
    Output: (B, embed_dim) float32.

    Depth is concatenated as a 4th channel.  When pretrained=True the first
    conv layer is expanded from 3→4 channels, with the depth channel
    initialised from the mean of the RGB weights.
    """

    def __init__(self, embed_dim: int = 128, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Expand first conv: 3 → 4 input channels.
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv

        # Remove the classification head; keep up to avgpool.
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        depth_ch = depth.unsqueeze(1)  # (B, 1, H, W)
        x = torch.cat([rgb, depth_ch], dim=1)  # (B, 4, H, W)
        features = self.backbone(x)  # (B, 512)
        return self.head(features)   # (B, embed_dim)


class UWBEncoder(nn.Module):
    """
    Encodes UWB CIR magnitude from multiple receivers into a fixed embedding.

    Input:  cir_mag   (B, n_receivers, n_samples) float32 — CIR magnitude
            fp_index  (B, n_receivers) float32 — normalised first-path index [0, 1]
    Output: (B, embed_dim) float32.

    Uses 1-D convolutions to process the CIR waveform (receivers as channels)
    and a small MLP to incorporate the first-path index as an auxiliary feature.
    """

    def __init__(
        self,
        n_receivers: int = 3,
        n_samples: int = 1016,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.cir_net = nn.Sequential(
            nn.Conv1d(n_receivers, 32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fp_proj = nn.Sequential(
            nn.Linear(n_receivers, 16),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(128 + 16, embed_dim),
            nn.ReLU(),
        )

    def forward(
        self, cir_mag: torch.Tensor, fp_index: torch.Tensor
    ) -> torch.Tensor:
        cir_feat = self.cir_net(cir_mag)     # (B, 128)
        fp_feat = self.fp_proj(fp_index)     # (B, 16)
        combined = torch.cat([cir_feat, fp_feat], dim=1)
        return self.combine(combined)        # (B, embed_dim)


class MmWaveEncoder(nn.Module):
    """Placeholder encoder for future mmWave 4-D point cloud data.

    Expected input: (B, N_points, 5) — x, y, z, velocity, intensity.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        raise NotImplementedError("mmWave encoder not yet implemented")

    def forward(self, mmwave_data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
