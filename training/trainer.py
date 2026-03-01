"""
training/trainer.py - Training loop with W&B logging and checkpointing.

The Trainer owns the optimizer, scheduler, and checkpoint logic. It calls
model.forward_batch(), model.loss(), and model.metrics() — nothing else —
keeping it fully decoupled from any specific architecture.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import wandb

from .config import Config
from .datamodule import NLOSDataModule
from .model import NLOSModel, ModelOutput


class Trainer:
    """
    Drives the train / val loop, checkpointing, and W&B logging.

    Args:
        model:      An NLOSModel subclass.
        datamodule: Configured NLOSDataModule.
        config:     Full Config (train + wandb settings used here).
        run:        Active wandb.Run (from wandb.init()).
    """

    def __init__(
        self,
        model: NLOSModel,
        datamodule: NLOSDataModule,
        config: Config,
        run: wandb.sdk.wandb_run.Run,
    ):
        self.model = model
        self.datamodule = datamodule
        self.config = config
        self.run = run

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        self.model.to(self.device)

        self.ckpt_dir = Path(config.train.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Resume before watching so W&B sees restored weights.
        if config.train.resume_from:
            self.load_checkpoint(config.train.resume_from)

        run.watch(model, log="gradients", log_freq=config.train.log_every_n_steps)

    # ── Public API ─────────────────────────────────────────────────────────

    def fit(self):
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        cfg = self.config.train

        print(f"Training on {self.device}  |  {cfg.max_epochs} epochs\n")

        for epoch in range(self.epoch + 1, cfg.max_epochs + 1):
            self.epoch = epoch
            train_loss = self._train_epoch(train_loader)

            if epoch % cfg.val_every_n_epochs == 0:
                val_loss, val_metrics = self._val_epoch(val_loader)
                self.run.log(
                    {
                        "epoch": epoch,
                        "val/loss": val_loss,
                        **{f"val/{k}": v for k, v in val_metrics.items()},
                    }
                )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best.pt")
                    print(f"  -> New best val loss: {val_loss:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()

            self.run.log(
                {
                    "epoch": epoch,
                    "train/loss_epoch": train_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )
            self._save_checkpoint("last.pt")

        print("\nTraining complete.")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {path}  (epoch {self.epoch}, best_val={self.best_val_loss:.4f})")

    # ── Train / val loops ──────────────────────────────────────────────────

    def _train_epoch(self, loader) -> float:
        self.model.train()
        cfg = self.config.train
        total_loss = 0.0
        t0 = time.perf_counter()

        for batch in loader:
            batch = self._to_device(batch)
            output: ModelOutput = self.model.forward_batch(batch)
            loss = self.model.loss(output, batch)

            self.optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            if self.global_step % cfg.log_every_n_steps == 0:
                self.run.log(
                    {"train/loss": loss.item(), "step": self.global_step, "epoch": self.epoch}
                )

        mean_loss = total_loss / len(loader)
        elapsed = time.perf_counter() - t0
        print(f"Epoch {self.epoch:4d}  train_loss={mean_loss:.4f}  ({elapsed:.1f}s)")
        return mean_loss

    def _val_epoch(self, loader) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        accumulated: dict[str, list[float]] = {}

        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                output: ModelOutput = self.model.forward_batch(batch)
                loss = self.model.loss(output, batch)
                total_loss += loss.item()

                cpu_output = ModelOutput(
                    prediction=output.prediction.cpu(),
                    aux={k: v.cpu() for k, v in output.aux.items()},
                )
                cpu_batch = self._to_cpu(batch)
                for k, v in self.model.metrics(cpu_output, cpu_batch).items():
                    accumulated.setdefault(k, []).append(v)

        mean_loss = total_loss / len(loader)
        mean_metrics = {k: sum(v) / len(v) for k, v in accumulated.items()}
        print(f"           val_loss={mean_loss:.4f}  {mean_metrics}")
        return mean_loss, mean_metrics

    # ── Checkpointing ──────────────────────────────────────────────────────

    def _save_checkpoint(self, name: str):
        path = self.ckpt_dir / name
        torch.save(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "config": self.config.to_dict(),
            },
            path,
        )
        self.run.save(str(path))

    # ── Optimizer / scheduler builders ────────────────────────────────────

    def _build_optimizer(self):
        cfg = self.config.train
        params = self.model.parameters()
        match cfg.optimizer:
            case "adamw":
                return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
            case "adam":
                return torch.optim.Adam(params, lr=cfg.lr)
            case "sgd":
                return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
            case _:
                raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")

    def _build_scheduler(self):
        cfg = self.config.train
        match cfg.scheduler:
            case "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=cfg.max_epochs
                )
            case "step":
                return torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=max(1, cfg.max_epochs // 3), gamma=0.1
                )
            case "none":
                return None
            case _:
                raise ValueError(f"Unknown scheduler: {cfg.scheduler!r}")

    # ── Device helpers ─────────────────────────────────────────────────────

    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _to_cpu(self, batch: dict) -> dict:
        return {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
