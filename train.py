"""
train.py - Entry point for the NLOS fusion training pipeline.

Usage:
    # From a config file:
    python train.py --config path/to/config.json

    # Resume a run:
    python train.py --config path/to/config.json --resume checkpoints/last.pt

Plug in your model by importing it and passing it to train().
"""

import argparse

import wandb

from training import Config, DataConfig, TrainConfig, WandbConfig
from training import NLOSDataModule, NLOSModel, Trainer


def train(config: Config, model: NLOSModel):
    """
    Initialise W&B, build the datamodule and trainer, and run training.

    Args:
        config: Full pipeline config.
        model:  Concrete NLOSModel subclass to train.

    Returns:
        The Trainer instance (contains final model state).
    """
    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity or None,
        name=config.wandb.run_name or None,
        tags=config.wandb.tags,
        config=config.to_dict(),
        resume="allow" if config.train.resume_from else None,
    )

    datamodule = NLOSDataModule(config.data)
    datamodule.summary()

    trainer = Trainer(model, datamodule, config, run)
    trainer.fit()

    run.finish()
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train an NLOS fusion model.")
    parser.add_argument("--config", required=True, help="Path to config JSON.")
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CKPT",
        help="Path to a checkpoint to resume from (overrides config.train.resume_from).",
    )
    args = parser.parse_args()

    config = Config.from_json(args.config)
    if args.resume:
        config.train.resume_from = args.resume

    # ── Plug in your model here ────────────────────────────────────────────
    # from my_models import MyFusionModel
    # model = MyFusionModel(...)
    # train(config, model)
    # ──────────────────────────────────────────────────────────────────────
    raise NotImplementedError(
        "Instantiate your NLOSModel subclass in train.py before running."
    )


if __name__ == "__main__":
    main()
