"""
training/fusion.py - Fusion module and the concrete NLOSFusionModel.

Combines per-sensor embeddings and drives localisation + pose prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .encoders import SPADEncoder, RGBDEncoder, UWBEncoder
from .heads import LocalizationHead, PoseHead
from .model import NLOSModel, ModelOutput


class FusionNeck(nn.Module):
    """
    Combines sensor embeddings via concatenation + projection.

    Input:  list of (B, embed_dim) tensors from active sensors.
    Output: (B, fused_dim) float32.
    """

    def __init__(self, embed_dim: int, n_sensors: int, fused_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * n_sensors, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
        )

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        fused = torch.cat(embeddings, dim=1)
        return self.proj(fused)


class NLOSFusionModel(NLOSModel):
    """
    Multi-sensor fusion model for NLOS human tracking.

    Builds only the encoders that are enabled in ModelConfig, fuses their
    embeddings, and runs localisation (+ optional pose) heads.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        embed_dim = config.embed_dim

        # Build active encoders.
        self.encoders = nn.ModuleDict()
        active_count = 0

        if config.use_spad:
            self.encoders["spad"] = SPADEncoder(
                in_bins=config.spad_bins,
                embed_dim=embed_dim,
            )
            active_count += 1

        if config.use_rgbd:
            self.encoders["rgbd"] = RGBDEncoder(
                embed_dim=embed_dim,
                pretrained=config.pretrained_backbone,
            )
            active_count += 1

        if config.use_uwb:
            self.encoders["uwb"] = UWBEncoder(
                n_receivers=config.uwb_n_receivers,
                n_samples=config.uwb_n_samples,
                embed_dim=embed_dim,
            )
            active_count += 1

        assert active_count > 0, "At least one sensor must be enabled in ModelConfig"

        # Fusion neck.
        fused_dim = config.fused_dim
        if active_count == 1:
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim, fused_dim),
                nn.ReLU(),
            )
        else:
            self.fusion = FusionNeck(embed_dim, active_count, fused_dim)

        # Task heads.
        self.loc_head = LocalizationHead(fused_dim)
        self.pose_head = (
            PoseHead(fused_dim, n_classes=config.n_pose_classes)
            if config.use_pose_head
            else None
        )

        # Loss weights.
        self.loc_weight = config.loc_loss_weight
        self.pose_weight = config.pose_loss_weight

    # ── Forward ───────────────────────────────────────────────────────────

    def forward_batch(self, batch: dict) -> ModelOutput:
        """Run all active encoders, fuse, and predict."""
        embeddings: list[torch.Tensor] = []

        if "spad" in self.encoders:
            embeddings.append(self.encoders["spad"](batch["spad"]))

        if "rgbd" in self.encoders:
            embeddings.append(self.encoders["rgbd"](batch["rgb"], batch["depth"]))

        if "uwb" in self.encoders:
            embeddings.append(
                self.encoders["uwb"](batch["uwb_cir"], batch["uwb_fp_index"])
            )

        # Fuse.
        if len(embeddings) == 1:
            fused = self.fusion(embeddings[0])
        else:
            fused = self.fusion(embeddings)

        # Predict.
        loc_pred = self.loc_head(fused)

        aux: dict[str, torch.Tensor] = {}
        if self.pose_head is not None:
            aux["pose_logits"] = self.pose_head(fused)

        return ModelOutput(prediction=loc_pred, aux=aux)

    def forward(self, spad, rgb, depth):
        """Legacy ABC interface — use forward_batch() instead."""
        raise NotImplementedError("Use forward_batch()")

    # ── Loss ──────────────────────────────────────────────────────────────

    def loss(self, output: ModelOutput, batch: dict) -> torch.Tensor:
        gt_loc = batch["gt_location"]
        loc_loss = F.smooth_l1_loss(output.prediction, gt_loc)
        total = self.loc_weight * loc_loss

        if self.pose_head is not None and "gt_pose" in batch:
            gt_pose = batch["gt_pose"]
            pose_logits = output.aux["pose_logits"]
            pose_loss = F.cross_entropy(pose_logits, gt_pose, ignore_index=-1)
            total = total + self.pose_weight * pose_loss

        return total

    # ── Metrics ───────────────────────────────────────────────────────────

    def metrics(self, output: ModelOutput, batch: dict) -> dict[str, float]:
        gt_loc = batch["gt_location"]
        pred_loc = output.prediction

        mae = (pred_loc - gt_loc).abs().mean().item()
        dist = ((pred_loc - gt_loc) ** 2).sum(dim=1).sqrt().mean().item()

        m: dict[str, float] = {
            "loc_mae": mae,
            "loc_dist": dist,
        }

        if self.pose_head is not None and "gt_pose" in batch:
            gt_pose = batch["gt_pose"]
            pose_logits = output.aux["pose_logits"]
            valid = gt_pose >= 0
            if valid.any():
                pred_cls = pose_logits[valid].argmax(dim=1)
                m["pose_acc"] = (pred_cls == gt_pose[valid]).float().mean().item()

        return m
