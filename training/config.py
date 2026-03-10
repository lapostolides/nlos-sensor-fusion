"""
training/config.py - Dataclass configs for the NLOS training pipeline.

All hyperparameters live here. Configs are serializable to/from JSON so they
can be saved alongside checkpoints and logged to W&B.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    # Sensor toggles (for ablation).
    use_spad: bool = True
    use_rgbd: bool = True
    use_uwb: bool = True
    use_mmwave: bool = False  # future

    # Architecture.
    embed_dim: int = 128       # per-sensor embedding size
    fused_dim: int = 256       # post-fusion size
    pretrained_backbone: bool = True  # ImageNet ResNet-18 for RGBD encoder

    # SPAD encoder.
    spad_bins: int = 8
    spad_spatial: int = 4      # 4 or 8

    # UWB encoder.
    uwb_n_receivers: int = 3
    uwb_n_samples: int = 1016

    # Task heads.
    use_pose_head: bool = True
    n_pose_classes: int = 5

    # Loss weights.
    loc_loss_weight: float = 1.0
    pose_loss_weight: float = 0.1


@dataclass
class DataConfig:
    # Paths to run directories (each must contain sync.json + sensor data).
    run_dirs: list[str] = field(default_factory=list)
    # Pairs with |dt| above this threshold are excluded from all splits.
    max_dt_ms: float = 100.0
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac is implied: 1 - train_frac - val_frac
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    # Only include pairs that have a ground-truth label.
    require_gt: bool = True
    # Camera used for ground-truth person detection.
    gt_camera: str = "overhead_cam"


@dataclass
class TrainConfig:
    max_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # "adamw" | "adam" | "sgd"
    optimizer: str = "adamw"
    # "cosine" | "step" | "none"
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1
    checkpoint_dir: str = "checkpoints/"
    # Path to a .pt checkpoint to resume from; None to train from scratch.
    resume_from: str | None = None


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "nlos-fusion"
    entity: str = ""
    run_name: str = ""
    tags: list[str] = field(default_factory=list)
    # Log a sample batch as images every N epochs (0 = disabled).
    log_media_every_n_epochs: int = 10


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        return cls(
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            wandb=WandbConfig(**d.get("wandb", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Config:
        with open(path) as f:
            return cls.from_dict(json.load(f))
