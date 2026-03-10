"""
training/heads.py - Task-specific prediction heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LocalizationHead(nn.Module):
    """
    Predicts normalised (cx, cy) person location in [0, 1].

    Input:  (B, in_dim) float32.
    Output: (B, 2) float32.
    """

    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PoseHead(nn.Module):
    """
    Classifies coarse pose into N classes (logits, no softmax).

    Input:  (B, in_dim) float32.
    Output: (B, n_classes) float32 logits.
    """

    def __init__(self, in_dim: int = 256, n_classes: int = 5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
