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
class DataConfig:
    # Paths to one or more _sync.json index files (supports multi-capture datasets).
    index_files: list[str] = field(default_factory=list)
    # Pairs with |dt| above this threshold are excluded from all splits.
    max_dt_ms: float = 100.0
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac is implied: 1 - train_frac - val_frac
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4


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
            train=TrainConfig(**d.get("train", {})),
            wandb=WandbConfig(**d.get("wandb", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Config:
        with open(path) as f:
            return cls.from_dict(json.load(f))
