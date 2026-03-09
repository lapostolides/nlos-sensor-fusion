"""
training/dataset.py - PyTorch Dataset for synced SPAD + RGB-D pairs.

Reads from the new run directory format:
    <run_dir>/
    ├── manifest.json
    ├── spad.npz                    # histograms (N, H, W, bins)
    ├── sensor_cam/rgb/000000.jpg   # BGR uint8
    ├── sensor_cam/depth/000000.png # uint16 mm
    └── sync.json                   # frame pairs from sync_data.py

Each __getitem__ returns a dict of plain float32 tensors — no enum keys,
no driver types — so the default DataLoader collate_fn works unchanged.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class NLOSDataset(Dataset):
    """
    Dataset of matched SPAD + camera pairs.

    Args:
        run_dirs: List of run directory paths (each containing sync.json + data).
        cam_name: Camera to pair with SPAD ('sensor_cam' or 'overhead_cam').
        transform: Optional callable applied to the sample dict before return.

    Returns per item (dict):
        spad:  float32 tensor (H, W, bins)  — SPAD histogram, raw counts
        rgb:   float32 tensor (3, H, W)     — RGB in [0, 1]
        depth: float32 tensor (H, W)        — depth in metres (0 = invalid)
        meta:  dict with dt_ms, spad_idx, cam_idx, run_dir
    """

    def __init__(
        self,
        run_dirs: list[Path | str],
        cam_name: str = "sensor_cam",
        transform=None,
    ):
        self.cam_name = cam_name
        self.transform = transform
        self._pairs: list[dict] = []
        self._spad_cache: dict[str, np.ndarray] = {}

        for rd in run_dirs:
            rd = Path(rd)
            sync_path = rd / "sync.json"
            if not sync_path.exists():
                raise FileNotFoundError(
                    f"sync.json not found in {rd}. Run sync_data.py first."
                )

            import json
            with open(sync_path) as f:
                sync = json.load(f)

            # Load SPAD histograms once per run (memory-mapped for large files)
            spad_path = rd / "spad.npz"
            if spad_path.exists():
                self._spad_cache[str(rd)] = np.load(spad_path)["histograms"]

            for entry in sync["pairs"]:
                if cam_name not in entry:
                    continue
                self._pairs.append({
                    "run_dir": str(rd),
                    "spad_idx": entry["spad_idx"],
                    "cam_idx": entry[cam_name]["idx"],
                    "dt_ms": entry[cam_name]["dt_ms"],
                })

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self._pairs[idx]
        run_dir = Path(pair["run_dir"])
        spad_idx = pair["spad_idx"]
        cam_idx = pair["cam_idx"]

        # SPAD
        histograms = self._spad_cache[pair["run_dir"]]
        spad = torch.from_numpy(histograms[spad_idx].astype(np.float32))

        # RGB
        rgb_path = run_dir / self.cam_name / "rgb" / f"{cam_idx:06d}.jpg"
        if not rgb_path.exists():
            # overhead_cam stores images directly (no rgb/ subdir)
            rgb_path = run_dir / self.cam_name / f"{cam_idx:06d}.jpg"
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)

        # Depth (sensor_cam only)
        depth_path = run_dir / self.cam_name / "depth" / f"{cam_idx:06d}.png"
        if depth_path.exists():
            depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth = torch.from_numpy(depth_mm.astype(np.float32) / 1000.0)
        else:
            depth = torch.zeros(1)

        sample = {
            "spad": spad,
            "rgb": rgb,
            "depth": depth,
            "meta": {
                "dt_ms": pair["dt_ms"],
                "spad_idx": spad_idx,
                "cam_idx": cam_idx,
                "run_dir": pair["run_dir"],
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
