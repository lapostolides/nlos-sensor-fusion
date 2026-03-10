"""
training/datamodule.py - DataModule: splits, DataLoaders, and dataset summary.

Accepts one or more run directories so captures from multiple sessions can
be combined into a single dataset. Pairs are split at the pair level (not
the capture level) after deterministic shuffling.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from torch.utils.data import DataLoader

from .config import DataConfig
from .dataset import NLOSDataset


class NLOSDataModule:
    """
    Loads sync indices from run directories, filters bad pairs, shuffles,
    splits into train / val / test, and vends DataLoaders.

    Args:
        config:    DataConfig with run_dirs, split fractions, and loader settings.
        transform: Optional transform applied to each sample (passed to NLOSDataset).
    """

    def __init__(self, config: DataConfig, transform=None):
        self.config = config
        self.transform = transform
        self._setup()

    def _setup(self):
        cfg = self.config
        all_pairs: list[dict] = []

        for run_dir_str in sorted(cfg.run_dirs):  # sorted for determinism
            rd = Path(run_dir_str)
            sync_path = rd / "sync.json"
            if not sync_path.exists():
                raise FileNotFoundError(
                    f"sync.json not found in {rd}. Run sync_data.py first."
                )

            with open(sync_path) as f:
                sync = json.load(f)

            # Load GT detection keys if we need to filter by GT availability.
            gt_keys: set[int] | None = None
            if cfg.require_gt:
                gt_path = rd / "gt.json"
                if gt_path.exists():
                    with open(gt_path) as f:
                        gt_payload = json.load(f)
                    gt_keys = {
                        int(k)
                        for k, v in gt_payload["detections"].items()
                        if v  # non-empty detection list
                    }
                else:
                    gt_keys = set()  # no GT file → no valid pairs from this run

            for entry in sync["pairs"]:
                # Check camera sync threshold.
                cam_key = "sensor_cam"
                if cam_key not in entry:
                    continue
                if entry[cam_key]["dt_ms"] > cfg.max_dt_ms:
                    continue

                # Build pair dict.
                pair: dict = {
                    "run_dir": str(rd),
                    "spad_idx": entry["spad_idx"],
                    "cam_idx": entry[cam_key]["idx"],
                    "dt_ms": entry[cam_key]["dt_ms"],
                }

                # Overhead camera index (for GT lookup).
                if "overhead_cam" in entry:
                    pair["overhead_cam_idx"] = entry["overhead_cam"]["idx"]

                # UWB receiver indices.
                for role in ("rx1", "rx2", "rx3"):
                    uwb_key = f"uwb_{role}"
                    if uwb_key in entry:
                        pair[uwb_key] = entry[uwb_key]["idx"]

                # Filter: require ground truth.
                if cfg.require_gt:
                    overhead_idx = pair.get("overhead_cam_idx")
                    if overhead_idx is None or gt_keys is None or overhead_idx not in gt_keys:
                        continue

                all_pairs.append(pair)

        # Deterministic shuffle and split.
        rng = random.Random(cfg.seed)
        rng.shuffle(all_pairs)

        n = len(all_pairs)
        n_train = int(n * cfg.train_frac)
        n_val = int(n * cfg.val_frac)

        self.train_pairs = all_pairs[:n_train]
        self.val_pairs = all_pairs[n_train : n_train + n_val]
        self.test_pairs = all_pairs[n_train + n_val :]

    # ── DataLoaders ────────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_pairs, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_pairs, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_pairs, shuffle=False)

    def _make_loader(self, pairs: list[dict], shuffle: bool) -> DataLoader:
        dataset = NLOSDataset(pairs, transform=self.transform)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    # ── Info ───────────────────────────────────────────────────────────────

    def summary(self):
        cfg = self.config
        total = len(self.train_pairs) + len(self.val_pairs) + len(self.test_pairs)
        all_p = self.train_pairs + self.val_pairs + self.test_pairs
        n_sources = len(set(p["run_dir"] for p in all_p)) if all_p else 0
        print(
            f"Dataset: {total} pairs from {n_sources} capture(s) "
            f"(max_dt={cfg.max_dt_ms:.0f} ms)\n"
            f"  train {len(self.train_pairs)} / "
            f"val {len(self.val_pairs)} / "
            f"test {len(self.test_pairs)}"
        )
