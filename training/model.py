"""
training/model.py - Abstract base class for all NLOS fusion models.

Concrete models subclass NLOSModel and implement forward(), loss(), and
metrics(). The Trainer calls only these three methods, so any architecture
can be dropped in without modifying the training loop.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """
    Standardised container returned by NLOSModel.forward().

    Attributes:
        prediction: The model's primary output tensor. Shape and semantics
                    are model-defined (e.g. (B, 2) for 2-D location,
                    (B, H, W) for a spatial heatmap).
        aux:        Optional dict of auxiliary tensors (intermediate features,
                    attention maps, etc.) for debugging or auxiliary losses.
    """

    prediction: torch.Tensor
    aux: dict[str, torch.Tensor] = field(default_factory=dict)


class NLOSModel(nn.Module):
    """
    Abstract base for all NLOS sensor-fusion models.

    Subclasses must implement:
        forward()  — produces a ModelOutput from a sample batch
        loss()     — scalar loss from output + batch
        metrics()  — evaluation metrics dict from output + batch

    The Trainer passes the full sample dict (keys: "spad", "rgb", "depth",
    "meta") so each model can use whichever modalities it needs.
    """

    @abstractmethod
    def forward(
        self,
        spad: torch.Tensor,   # (B, H, W, bins) float32 — raw histogram counts
        rgb: torch.Tensor,    # (B, 3, H, W)    float32 — RGB in [0, 1]
        depth: torch.Tensor,  # (B, H, W)       float32 — depth in metres
    ) -> ModelOutput:
        """Run the forward pass and return a ModelOutput."""
        ...

    @abstractmethod
    def loss(self, output: ModelOutput, batch: dict) -> torch.Tensor:
        """
        Compute a scalar loss.

        Args:
            output: The ModelOutput returned by forward().
            batch:  The full sample dict (includes ground-truth labels if any).

        Returns:
            Scalar loss tensor (differentiable).
        """
        ...

    @abstractmethod
    def metrics(self, output: ModelOutput, batch: dict) -> dict[str, float]:
        """
        Compute evaluation metrics (no gradients required).

        Args:
            output: ModelOutput with prediction already moved to CPU.
            batch:  Sample dict with tensors already moved to CPU.

        Returns:
            Dict mapping metric name -> scalar float,
            e.g. {"mae": 0.12, "accuracy": 0.95}.
        """
        ...

    # ── Convenience wrapper called by Trainer ──────────────────────────────

    def forward_batch(self, batch: dict) -> ModelOutput:
        """Unpack the sample dict and call forward()."""
        return self.forward(batch["spad"], batch["rgb"], batch["depth"])
