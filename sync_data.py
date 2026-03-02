#!/usr/bin/env python3
"""
sync_data.py - Post-hoc temporal synchronization of SPAD and camera frames.

Loads a capture PKL file, extracts SPAD and per-camera timestamps, and
nearest-neighbour matches every SPAD frame to the closest frame from each
active camera independently.  Warns about pairs that exceed MAX_DT_MS ms.

Supported cameras (auto-detected from record keys):
  sensor_cam   — Intel RealSense RGB-D  (also accepts legacy key 'realsense')
  overhead_cam — eMeet C960 USB webcam

Timestamp layout (per-sensor keys take priority over shared 'timestamp'):
  {"iter":..., "spad":..., "spad_timestamp":"...",
               "sensor_cam":...,   "sensor_cam_timestamp":"...",
               "overhead_cam":..., "overhead_cam_timestamp":"..."}

Usage:
    python sync_data.py [path/to/capture.pkl] [--max-dt-ms 100]
"""

import argparse
import bisect
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import cloudpickle as pickle
except ImportError:
    import pickle

MAX_DT_MS = 100.0

# Maps record data-keys to a canonical camera name.
# Legacy 'realsense' is treated as 'sensor_cam'.
_CAM_DATA_KEYS: dict[str, str] = {
    "sensor_cam":   "sensor_cam",
    "realsense":    "sensor_cam",   # backwards-compat alias
    "overhead_cam": "overhead_cam",
}
# Corresponding per-sensor timestamp keys
_CAM_TS_KEYS: dict[str, str] = {
    "sensor_cam":   "sensor_cam_timestamp",
    "realsense":    "realsense_timestamp",
    "overhead_cam": "overhead_cam_timestamp",
}


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_records(path: Path) -> list:
    records = []
    with open(path, "rb") as f:
        try:
            while True:
                records.append(pickle.load(f))
        except EOFError:
            pass
    return records


def _ts_to_s(ts) -> float:
    """Convert an ISO timestamp string or a numeric value to seconds since epoch."""
    if isinstance(ts, (int, float)):
        return float(ts)
    return datetime.fromisoformat(str(ts)).timestamp()


# ── Frame extraction ──────────────────────────────────────────────────────────

def _extract_frames(
    records: list,
) -> tuple[list[float], dict[str, list[float]]]:
    """
    Return (spad_timestamps, cam_timestamps) where cam_timestamps maps each
    active camera's canonical name to its list of timestamps.

    Only cameras that appear in at least one record are included in the dict.
    """
    spad_ts: list[float] = []
    cam_ts: dict[str, list[float]] = {}

    for r in records:
        if not isinstance(r, dict) or "iter" not in r:
            continue

        shared = _ts_to_s(r["timestamp"]) if "timestamp" in r else None

        if "spad" in r:
            t = _ts_to_s(r["spad_timestamp"]) if "spad_timestamp" in r else shared
            if t is not None:
                spad_ts.append(t)

        for data_key, canonical in _CAM_DATA_KEYS.items():
            if data_key not in r:
                continue
            ts_key = _CAM_TS_KEYS[data_key]
            t = _ts_to_s(r[ts_key]) if ts_key in r else shared
            if t is not None:
                cam_ts.setdefault(canonical, []).append(t)

    return spad_ts, cam_ts


# ── Nearest-neighbour matching ────────────────────────────────────────────────

def _nearest_pool_idx(sorted_pool: list[float], q: float) -> int:
    """Return the index in *sorted_pool* of the value nearest to *q*."""
    i = bisect.bisect_left(sorted_pool, q)
    if i == 0:
        return 0
    if i == len(sorted_pool):
        return len(sorted_pool) - 1
    return i - 1 if (q - sorted_pool[i - 1]) <= (sorted_pool[i] - q) else i


def match_frames(
    spad_ts: list[float],
    cam_ts: dict[str, list[float]],
) -> dict[str, list[tuple[int, int, float]]]:
    """
    Match every SPAD frame to its nearest frame for each camera independently.

    Returns:
        Dict mapping camera name -> list of (spad_idx, cam_idx, dt_s) tuples,
        sorted by spad_idx.
    """
    all_matches: dict[str, list[tuple[int, int, float]]] = {}

    for cam_name, ts_list in cam_ts.items():
        sorted_order = sorted(range(len(ts_list)), key=lambda i: ts_list[i])
        sorted_ts = [ts_list[i] for i in sorted_order]

        matches = []
        for si, st in enumerate(spad_ts):
            pool_pos = _nearest_pool_idx(sorted_ts, st)
            ci = sorted_order[pool_pos]
            matches.append((si, ci, abs(st - ts_list[ci])))

        all_matches[cam_name] = matches

    return all_matches


# ── Report ────────────────────────────────────────────────────────────────────

def _camera_report(
    cam_name: str,
    n_spad: int,
    n_cam: int,
    matches: list[tuple[int, int, float]],
    max_dt_ms: float,
):
    used_cam = {ci for _, ci, _ in matches}
    n_unmatched = n_cam - len(used_cam)

    print(f"\n  [{cam_name}]  {n_cam} frames")
    print(f"    Matched SPAD frames   : {len(matches):>6}  (each -> nearest cam)")
    print(f"    Unique camera matched : {len(used_cam):>6}")
    print(f"    Unmatched camera      : {n_unmatched:>6}")

    if not matches:
        return

    dts_ms = [dt * 1000 for _, _, dt in matches]
    mean_dt = sum(dts_ms) / len(dts_ms)
    n_over = sum(1 for dt_ms in dts_ms if dt_ms > max_dt_ms)

    print(f"    Avg |dt|: {mean_dt:.2f} ms  "
          f"(min {min(dts_ms):.2f}, max {max(dts_ms):.2f} ms)")
    if n_over:
        print(f"    WARNING: {n_over} pair(s) exceed {max_dt_ms:.0f} ms threshold.")
    else:
        print(f"    All pairs within {max_dt_ms:.0f} ms threshold.")


def report(
    n_spad: int,
    cam_ts: dict[str, list[float]],
    all_matches: dict[str, list[tuple[int, int, float]]],
    spad_ts: list[float],
    max_dt_ms: float,
):
    sep = "-" * 52
    print(sep)
    print(f"  SPAD frames : {n_spad}")
    for cam_name, matches in all_matches.items():
        _camera_report(cam_name, n_spad, len(cam_ts[cam_name]), matches, max_dt_ms)

    # Overall SPAD cadence
    if n_spad > 1:
        intervals_ms = [(spad_ts[i+1] - spad_ts[i]) * 1000
                        for i in range(len(spad_ts) - 1)]
        mean_iv = sum(intervals_ms) / len(intervals_ms)
        print(f"\n  SPAD avg interval : {mean_iv:.2f} ms  ({1000/mean_iv:.1f} fps)")

    print(sep)


# ── Index export ──────────────────────────────────────────────────────────────

def save_index(
    pkl_path: Path,
    all_matches: dict[str, list[tuple[int, int, float]]],
    spad_ts: list[float],
    cam_ts: dict[str, list[float]],
    max_dt_ms: float,
) -> Path:
    """
    Write a JSON sync index alongside *pkl_path*.

    Each entry in 'pairs' corresponds to one SPAD frame and includes the
    nearest-matched index from every active camera.

    Returns the path of the written file.
    """
    index_path = pkl_path.with_name(pkl_path.stem + "_sync.json")

    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts).isoformat()

    n_spad = len(spad_ts)

    # Build per-camera lookup: spad_idx -> (cam_idx, dt_ms, cam_ts)
    cam_lookup: dict[str, dict[int, dict]] = {}
    for cam_name, matches in all_matches.items():
        cam_lookup[cam_name] = {
            si: {"idx": ci, "dt_ms": round(dt * 1000, 3),
                 "ts": _iso(cam_ts[cam_name][ci])}
            for si, ci, dt in matches
        }

    pairs = []
    for si in range(n_spad):
        entry: dict = {"spad_idx": si, "spad_ts": _iso(spad_ts[si])}
        for cam_name, lookup in cam_lookup.items():
            if si in lookup:
                entry[cam_name] = lookup[si]
        pairs.append(entry)

    # Summary stats per camera
    camera_stats = {}
    for cam_name, matches in all_matches.items():
        dts_ms = [dt * 1000 for _, _, dt in matches]
        camera_stats[cam_name] = {
            "n_frames": len(cam_ts[cam_name]),
            "n_matched": len(matches),
            "n_over_threshold": sum(1 for dt_ms in dts_ms if dt_ms > max_dt_ms),
            "mean_dt_ms": round(sum(dts_ms) / len(dts_ms), 3) if dts_ms else None,
        }

    index = {
        "source_pkl": str(pkl_path.resolve()),
        "created": datetime.now().isoformat(),
        "max_dt_ms": max_dt_ms,
        "n_spad": n_spad,
        "cameras": camera_stats,
        "pairs": pairs,
    }

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return index_path


# ── Public API ────────────────────────────────────────────────────────────────

def sync(
    pkl_path: Path,
    max_dt_ms: float = MAX_DT_MS,
    save: bool = True,
) -> dict[str, list[tuple[int, int, float]]]:
    """
    Load *pkl_path*, match SPAD frames to each camera by timestamp, print a
    summary report, and (if *save* is True) write a JSON index alongside the pkl.

    Returns:
        Dict mapping camera name -> list of (spad_idx, cam_idx, dt_s) pairs.
    """
    print(f"\nFile: {pkl_path}")
    records = load_records(pkl_path)

    n_data = sum(1 for r in records if isinstance(r, dict) and "iter" in r)
    n_meta = len(records) - n_data
    print(f"Loaded {len(records)} records  ({n_data} data, {n_meta} metadata)\n")

    spad_ts, cam_ts = _extract_frames(records)

    if not spad_ts:
        print("No SPAD frames found.")
        return {}
    if not cam_ts:
        print(f"No camera frames found; {len(spad_ts)} SPAD frames available.")
        return {}

    all_matches = match_frames(spad_ts, cam_ts)
    report(len(spad_ts), cam_ts, all_matches, spad_ts, max_dt_ms)

    if save:
        index_path = save_index(pkl_path, all_matches, spad_ts, cam_ts, max_dt_ms)
        print(f"Index saved: {index_path}")

    return all_matches


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Synchronise SPAD and camera frames by timestamp."
    )
    parser.add_argument("pkl", nargs="?", help="Path to capture PKL file.")
    parser.add_argument(
        "--max-dt-ms",
        type=float,
        default=MAX_DT_MS,
        metavar="MS",
        help=f"Warn threshold for |dt| in ms (default: {MAX_DT_MS:.0f})",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip writing the JSON sync index.",
    )
    args = parser.parse_args()

    if args.pkl:
        path = Path(args.pkl)
        if not path.exists():
            print(f"Error: {path} does not exist.")
            sys.exit(1)
    else:
        logs = sorted(Path("data/logs").rglob("*.pkl"), key=lambda p: p.stat().st_mtime)
        if not logs:
            print("No PKL file found in data/logs/. Pass a path or capture data first.")
            sys.exit(1)
        path = logs[-1]
        print(f"Auto-detected: {path}")

    sync(path, max_dt_ms=args.max_dt_ms, save=not args.no_index)


if __name__ == "__main__":
    main()
