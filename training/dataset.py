"""
training/dataset.py - PyTorch Dataset for synced multi-sensor NLOS data.

Reads from run directories containing:
    <run_dir>/
    ├── manifest.json
    ├── spad.npz                    # histograms (N, H, W, bins)
    ├── sensor_cam/rgb/000000.jpg   # BGR uint8
    ├── sensor_cam/depth/000000.png # uint16 mm
    ├── overhead_cam/000000.jpg     # BGR uint8
    ├── rx1.npz, rx2.npz, rx3.npz  # UWB CIR per receiver
    ├── sync.json                   # frame pairs from sync_data.py
    └── gt.json                     # ground-truth person detections

Each __getitem__ returns a dict of plain float32 tensors — no enum keys,
no driver types — so the default DataLoader collate_fn works unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Pose string → class index mapping (matches pose_estimation.Pose enum values).
POSE_TO_IDX: dict[str, int] = {
    "still": 0,
    "locomotion": 1,
    "crouching": 2,
    "arms_raised": 3,
    "unknown": 4,
}
IDX_TO_POSE: dict[int, str] = {v: k for k, v in POSE_TO_IDX.items()}

# Number of UWB CIR samples per frame (fixed by DW1000 hardware).
UWB_N_SAMPLES = 1016


class NLOSDataset(Dataset):
    """
    Dataset of synced multi-sensor frames with ground-truth labels.

    Args:
        pairs:     Pre-filtered list of pair dicts (from NLOSDataModule).
                   Each dict has keys: run_dir, spad_idx, cam_idx, dt_ms,
                   and optionally overhead_cam_idx, uwb_rx1_idx, etc.
        cam_name:  Camera to load RGB/depth from ('sensor_cam' or 'overhead_cam').
        transform: Optional callable applied to the sample dict before return.

    Returns per item (dict):
        spad:          float32 (H, W, bins)  — SPAD histogram, raw counts
        rgb:           float32 (3, H, W)     — RGB in [0, 1]
        depth:         float32 (H, W)        — depth in metres (0 = invalid)
        uwb_cir:       float32 (3, 1016)     — CIR magnitude per receiver (0 if missing)
        uwb_fp_index:  float32 (3,)          — first-path index normalised to [0, 1]
        uwb_mask:      bool    (3,)          — which receivers have valid data
        gt_location:   float32 (2,)          — normalised (cx, cy) in [0, 1]
        gt_pose:       long    ()            — pose class index (-1 if missing)
        gt_valid:      bool    ()            — whether GT label exists
        meta:          dict                  — dt_ms, spad_idx, cam_idx, run_dir
    """

    def __init__(
        self,
        pairs: list[dict],
        cam_name: str = "sensor_cam",
        transform=None,
    ):
        self.cam_name = cam_name
        self.transform = transform
        self._pairs = pairs

        # Caches (populated lazily per run_dir).
        self._spad_cache: dict[str, np.ndarray] = {}
        self._uwb_cache: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        self._gt_cache: dict[str, dict[int, list[dict]]] = {}
        self._cam_resolution: dict[str, tuple[int, int]] = {}  # (width, height)

        # Discover unique run dirs and pre-load caches.
        run_dirs = sorted(set(p["run_dir"] for p in pairs))
        for rd_str in run_dirs:
            self._load_run(Path(rd_str))

    # ── Cache loading ─────────────────────────────────────────────────────

    def _load_run(self, rd: Path) -> None:
        rd_str = str(rd)

        # SPAD histograms.
        spad_path = rd / "spad.npz"
        if spad_path.exists():
            self._spad_cache[rd_str] = np.load(spad_path)["histograms"]

        # UWB CIR data per receiver.
        uwb: dict[str, dict[str, np.ndarray]] = {}
        for role in ("rx1", "rx2", "rx3"):
            npz_path = rd / f"{role}.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                uwb[role] = {
                    "cir_mag": data["cir_mag"],
                    "fp_index": data["fp_index"],
                }
        self._uwb_cache[rd_str] = uwb

        # Ground-truth detections.
        gt_path = rd / "gt.json"
        if gt_path.exists():
            with open(gt_path) as f:
                payload = json.load(f)
            # Map frame index (int) → list of detection dicts.
            self._gt_cache[rd_str] = {
                int(k): v for k, v in payload["detections"].items()
            }

        # Overhead camera resolution for GT normalisation.
        manifest_path = rd / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            cam_info = manifest.get("sensors", {}).get("overhead_cam", {})
            res = cam_info.get("resolution")
            if res and len(res) == 2:
                self._cam_resolution[rd_str] = tuple(res)  # (width, height)
        # Fallback default.
        if rd_str not in self._cam_resolution:
            self._cam_resolution[rd_str] = (1920, 1080)

    # ── Dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self._pairs[idx]
        run_dir = Path(pair["run_dir"])
        rd_str = pair["run_dir"]
        spad_idx = pair["spad_idx"]
        cam_idx = pair["cam_idx"]

        # ── SPAD ──
        histograms = self._spad_cache.get(rd_str)
        if histograms is not None:
            spad = torch.from_numpy(histograms[spad_idx].astype(np.float32))
        else:
            spad = torch.zeros(1)

        # ── RGB ──
        rgb_path = run_dir / self.cam_name / "rgb" / f"{cam_idx:06d}.jpg"
        if not rgb_path.exists():
            # overhead_cam stores images directly (no rgb/ subdir).
            rgb_path = run_dir / self.cam_name / f"{cam_idx:06d}.jpg"
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)

        # ── Depth (sensor_cam only) ──
        depth_path = run_dir / self.cam_name / "depth" / f"{cam_idx:06d}.png"
        if depth_path.exists():
            depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth = torch.from_numpy(depth_mm.astype(np.float32) / 1000.0)
        else:
            depth = torch.zeros(1)

        # ── UWB CIR ──
        uwb_cir = torch.zeros(3, UWB_N_SAMPLES, dtype=torch.float32)
        uwb_fp_index = torch.zeros(3, dtype=torch.float32)
        uwb_mask = torch.zeros(3, dtype=torch.bool)

        uwb_data = self._uwb_cache.get(rd_str, {})
        for i, role in enumerate(("rx1", "rx2", "rx3")):
            uwb_key = f"uwb_{role}"
            if uwb_key in pair and role in uwb_data:
                uwb_idx = pair[uwb_key]
                role_data = uwb_data[role]
                if uwb_idx < len(role_data["cir_mag"]):
                    uwb_cir[i] = torch.from_numpy(
                        role_data["cir_mag"][uwb_idx].astype(np.float32)
                    )
                    uwb_fp_index[i] = float(role_data["fp_index"][uwb_idx]) / UWB_N_SAMPLES
                    uwb_mask[i] = True

        # ── Ground truth ──
        gt_location = torch.zeros(2, dtype=torch.float32)
        gt_pose = torch.tensor(-1, dtype=torch.long)
        gt_valid = torch.tensor(False, dtype=torch.bool)

        overhead_idx = pair.get("overhead_cam_idx")
        gt_dets = self._gt_cache.get(rd_str, {})
        if overhead_idx is not None and overhead_idx in gt_dets:
            dets = gt_dets[overhead_idx]
            if dets:
                person = dets[0]  # take first (highest confidence) detection
                cx, cy = person["center"]
                w_res, h_res = self._cam_resolution[rd_str]
                gt_location[0] = cx / w_res
                gt_location[1] = cy / h_res
                gt_valid = torch.tensor(True, dtype=torch.bool)
                if "pose" in person and person["pose"] is not None:
                    gt_pose = torch.tensor(
                        POSE_TO_IDX.get(person["pose"], 4), dtype=torch.long
                    )

        sample = {
            "spad": spad,
            "rgb": rgb,
            "depth": depth,
            "uwb_cir": uwb_cir,
            "uwb_fp_index": uwb_fp_index,
            "uwb_mask": uwb_mask,
            "gt_location": gt_location,
            "gt_pose": gt_pose,
            "gt_valid": gt_valid,
            "meta": {
                "dt_ms": pair["dt_ms"],
                "spad_idx": spad_idx,
                "cam_idx": cam_idx,
                "run_dir": rd_str,
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
