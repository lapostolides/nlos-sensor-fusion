"""
run_writer.py — ML-friendly per-sensor file writer for full_capture.py.

Directory layout produced by RunWriter:

    <run_dir>/
    ├── manifest.json                  # metadata + per-sensor timestamps
    ├── spad.npz                       # histograms (N, H, W, bins), timestamps
    ├── sensor_cam/
    │   ├── rgb/000000.jpg  ...        # BGR uint8, JPEG q95
    │   └── depth/000000.png  ...      # uint16 mm, 16-bit PNG (lossless)
    ├── overhead_cam/
    │   └── 000000.jpg  ...            # BGR uint8, JPEG q95
    └── rx1.npz, rx2.npz, ...         # UWB (written by capture_uwb.py)
"""

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class RunWriter:
    """Streams per-sensor data to disk during capture.

    - SPAD histograms are accumulated in RAM (small) and written at finalize().
    - Camera frames are written to disk immediately as individual image files.
    - A manifest.json with metadata and all timestamps is written at finalize().
    """

    JPEG_QUALITY = 95

    def __init__(self, run_dir: Path, metadata: dict):
        self.run_dir = run_dir
        self._metadata = metadata

        # SPAD — small per frame, accumulate in memory
        self._spad_frames: list[np.ndarray] = []
        self._spad_timestamps: list[float] = []

        # sensor_cam — write images immediately
        self._sensor_cam_rgb_dir = run_dir / "sensor_cam" / "rgb"
        self._sensor_cam_depth_dir = run_dir / "sensor_cam" / "depth"
        self._sensor_cam_count = 0
        self._sensor_cam_timestamps: list[float] = []

        # overhead_cam — write images immediately
        self._overhead_cam_dir = run_dir / "overhead_cam"
        self._overhead_cam_count = 0
        self._overhead_cam_timestamps: list[float] = []

        self._finalized = False

    # ── Timestamp conversion ───────────────────────────────────────────

    @staticmethod
    def _ts_to_epoch(ts) -> float:
        """Convert ISO string or numeric timestamp to epoch seconds."""
        if isinstance(ts, (int, float)):
            return float(ts)
        return datetime.fromisoformat(str(ts)).timestamp()

    # ── Per-sensor write methods ───────────────────────────────────────

    def write_spad(self, data, timestamp):
        """Append one SPAD frame.  *data* is the dict from sensor.accumulate()."""
        if isinstance(data, dict):
            histogram = next(iter(data.values()))  # {SPADDataType: ndarray} → ndarray
        else:
            histogram = data  # already an ndarray
        self._spad_frames.append(np.array(histogram))
        self._spad_timestamps.append(self._ts_to_epoch(timestamp))
        if len(self._spad_frames) == 1:
            print(f"[RunWriter] First SPAD frame: shape={histogram.shape}, "
                  f"dtype={histogram.dtype}", flush=True)

    def write_sensor_cam(self, frame: dict, timestamp):
        """Write one sensor_cam frame (RGB JPEG + depth 16-bit PNG)."""
        if self._sensor_cam_count == 0:
            self._sensor_cam_rgb_dir.mkdir(parents=True, exist_ok=True)
            self._sensor_cam_depth_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{self._sensor_cam_count:06d}"

        rgb = frame.get("aligned_rgb", frame.get("raw_rgb"))
        if rgb is not None:
            cv2.imwrite(
                str(self._sensor_cam_rgb_dir / f"{fname}.jpg"),
                rgb,
                [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY],
            )

        depth = frame.get("aligned_depth")
        if depth is not None:
            cv2.imwrite(
                str(self._sensor_cam_depth_dir / f"{fname}.png"),
                depth,
            )

        self._sensor_cam_timestamps.append(self._ts_to_epoch(timestamp))
        self._sensor_cam_count += 1

    def write_overhead_cam(self, frame: dict, timestamp):
        """Write one overhead_cam frame (RGB JPEG)."""
        if self._overhead_cam_count == 0:
            self._overhead_cam_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{self._overhead_cam_count:06d}"

        rgb = frame.get("raw_rgb")
        if rgb is not None:
            cv2.imwrite(
                str(self._overhead_cam_dir / f"{fname}.jpg"),
                rgb,
                [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY],
            )

        self._overhead_cam_timestamps.append(self._ts_to_epoch(timestamp))
        self._overhead_cam_count += 1

    # ── Finalize ───────────────────────────────────────────────────────

    def finalize(self):
        """Write spad.npz and manifest.json.  Safe to call multiple times."""
        if self._finalized:
            return
        self._finalized = True
        print(f"[RunWriter] Finalizing ({self.counts})...", flush=True)

        # SPAD npz
        if self._spad_frames:
            try:
                np.savez_compressed(
                    self.run_dir / "spad.npz",
                    histograms=np.array(self._spad_frames),
                    timestamps=np.array(self._spad_timestamps, dtype=np.float64),
                )
                print(f"[RunWriter] Saved spad.npz ({len(self._spad_frames)} frames)",
                      flush=True)
            except Exception as e:
                print(f"[RunWriter] ERROR writing spad.npz: {e}", flush=True)

        # Manifest
        frames: dict = {}
        if self._spad_timestamps:
            frames["spad"] = {
                "count": len(self._spad_timestamps),
                "file": "spad.npz",
                "timestamps": self._spad_timestamps,
            }
        if self._sensor_cam_timestamps:
            frames["sensor_cam"] = {
                "count": self._sensor_cam_count,
                "rgb_dir": "sensor_cam/rgb",
                "depth_dir": "sensor_cam/depth",
                "timestamps": self._sensor_cam_timestamps,
            }
        if self._overhead_cam_timestamps:
            frames["overhead_cam"] = {
                "count": self._overhead_cam_count,
                "rgb_dir": "overhead_cam",
                "timestamps": self._overhead_cam_timestamps,
            }

        manifest = {
            "format_version": 1,
            **self._metadata,
            "frames": frames,
        }

        try:
            with open(self.run_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"[RunWriter] Saved manifest.json", flush=True)
        except Exception as e:
            print(f"[RunWriter] ERROR writing manifest.json: {e}", flush=True)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "spad": len(self._spad_frames),
            "sensor_cam": self._sensor_cam_count,
            "overhead_cam": self._overhead_cam_count,
        }
