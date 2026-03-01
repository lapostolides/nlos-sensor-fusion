"""
training/dataset.py - PyTorch Dataset for synced SPAD + RGB-D pairs.

Reads directly from raw .pkl files using a pre-built byte-offset index so
the full pkl never needs to be loaded into RAM. Multiple pkl files are
supported (each pair carries the path to its source pkl).

Each __getitem__ returns a dict of plain float32 tensors — no enum keys,
no driver types — so the default DataLoader collate_fn works unchanged.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cloudpickle as pickle
except ImportError:
    import pickle


def _build_offset_index(pkl_path: Path) -> dict[int, int]:
    """
    Scan *pkl_path* once and return a mapping of record iter -> byte offset.
    Enables O(1) random access without loading all records into memory.
    """
    offsets: dict[int, int] = {}
    with open(pkl_path, "rb") as f:
        try:
            while True:
                pos = f.tell()
                record = pickle.load(f)
                if isinstance(record, dict) and "iter" in record:
                    offsets[record["iter"]] = pos
        except EOFError:
            pass
    return offsets


class NLOSDataset(Dataset):
    """
    Dataset of matched SPAD + camera pairs.

    Args:
        pairs: List of pair dicts from the sync index, each augmented with a
               "pkl_path" key pointing to its source .pkl file.
        transform: Optional callable applied to the sample dict before return.

    Returns per item (dict):
        spad:  float32 tensor (H, W, bins)  — SPAD histogram, raw counts
        rgb:   float32 tensor (3, H, W)     — aligned RGB in [0, 1], RGB order
        depth: float32 tensor (H, W)        — aligned depth in metres (0 = invalid)
        meta:  dict with dt_ms, spad_idx, cam_idx, pkl_path
    """

    def __init__(self, pairs: list[dict], transform=None):
        self.pairs = pairs
        self.transform = transform

        # Build one offset index per unique pkl file — scan each pkl once at init.
        unique_pkls = {p["pkl_path"] for p in pairs}
        self._offsets: dict[str, dict[int, int]] = {
            pkl: _build_offset_index(Path(pkl)) for pkl in unique_pkls
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        pkl = pair["pkl_path"]

        spad_record = self._load_record(pkl, pair["spad_idx"])
        cam_record = self._load_record(pkl, pair["cam_idx"])

        spad = self._extract_spad(spad_record)
        rgb = self._extract_rgb(cam_record)
        depth = self._extract_depth(cam_record)

        sample = {
            "spad": spad,
            "rgb": rgb,
            "depth": depth,
            "meta": {
                "dt_ms": pair["dt_ms"],
                "spad_idx": pair["spad_idx"],
                "cam_idx": pair["cam_idx"],
                "pkl_path": pkl,
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_record(self, pkl_path: str, iter_idx: int) -> dict:
        offset = self._offsets[pkl_path][iter_idx]
        with open(pkl_path, "rb") as f:
            f.seek(offset)
            return pickle.load(f)

    @staticmethod
    def _extract_spad(record: dict) -> torch.Tensor:
        # SPAD data is {SPADDataType.HISTOGRAM: np.ndarray (H, W, bins)}.
        # Extract the array without importing the driver enum.
        histogram: np.ndarray = next(iter(record["spad"].values()))
        return torch.from_numpy(histogram.astype(np.float32))  # (H, W, bins)

    @staticmethod
    def _extract_rgb(record: dict) -> torch.Tensor:
        bgr: np.ndarray = record["realsense"]["aligned_rgb"]  # uint8 (H, W, 3) BGR
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0      # float32 (H, W, 3) RGB
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)  # (3, H, W)

    @staticmethod
    def _extract_depth(record: dict) -> torch.Tensor:
        depth_mm: np.ndarray = record["realsense"]["aligned_depth"]  # uint16 (H, W)
        return torch.from_numpy(depth_mm.astype(np.float32) / 1000.0)  # metres (H, W)
