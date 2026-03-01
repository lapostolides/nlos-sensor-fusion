"""
training/datamodule.py - DataModule: splits, DataLoaders, and dataset summary.

Accepts one or more sync index files so captures from multiple sessions can
be combined into a single dataset. Pairs are split at the pair level (not
the capture level) after deterministic shuffling.
"""

import json
import random
from pathlib import Path

from torch.utils.data import DataLoader

from .config import DataConfig
from .dataset import NLOSDataset


class NLOSDataModule:
    """
    Loads sync indices, filters bad pairs, shuffles, splits into
    train / val / test, and vends DataLoaders.

    Args:
        config: DataConfig with index_files, split fractions, and loader settings.
        transform: Optional transform applied to each sample (passed to NLOSDataset).
    """

    def __init__(self, config: DataConfig, transform=None):
        self.config = config
        self.transform = transform
        self._setup()

    def _setup(self):
        all_pairs: list[dict] = []

        for index_path in sorted(self.config.index_files):  # sorted for determinism
            with open(index_path) as f:
                index = json.load(f)
            pkl_path = index["source_pkl"]
            for pair in index["pairs"]:
                if pair["dt_ms"] <= self.config.max_dt_ms:
                    all_pairs.append({**pair, "pkl_path": pkl_path})

        rng = random.Random(self.config.seed)
        rng.shuffle(all_pairs)

        n = len(all_pairs)
        n_train = int(n * self.config.train_frac)
        n_val = int(n * self.config.val_frac)

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
        n_sources = len(set(p["pkl_path"] for p in self.train_pairs + self.val_pairs + self.test_pairs))
        print(
            f"Dataset: {total} pairs from {n_sources} capture(s) "
            f"(max_dt={cfg.max_dt_ms:.0f} ms)\n"
            f"  train {len(self.train_pairs)} / "
            f"val {len(self.val_pairs)} / "
            f"test {len(self.test_pairs)}"
        )
