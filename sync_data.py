#!/usr/bin/env python3
"""
sync_data.py - Post-hoc temporal synchronization of SPAD and camera frames.

Loads a capture PKL file, extracts SPAD and camera frames each with their own
timestamps, and nearest-neighbour matches every SPAD frame to the closest
camera frame. Extra camera frames remain unmatched. Warns about pairs that
exceed MAX_DT_MS milliseconds.

Supports two timestamp layouts:
  - Shared:     {"iter":..., "timestamp":"...", "spad":..., "realsense":...}
                (current full_capture.py format; both sensors share one timestamp)
  - Per-sensor: {"iter":..., "spad":..., "spad_timestamp":"...",
                              "realsense":..., "realsense_timestamp":"..."}
                (future format; each sensor records its own capture time)
  - Separate records are also handled: a record with only "spad" or only
    "realsense" key uses its own "timestamp".

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


# ── Frame extraction ───────────────────────────────────────────────────────────

def _extract_frames(records: list) -> tuple[list[float], list[float]]:
    """
    Return (spad_timestamps, cam_timestamps) in record order.

    Per-sensor timestamp keys ('spad_timestamp' / 'realsense_timestamp') take
    priority. Falls back to the shared 'timestamp' key when they are absent.
    """
    spad_ts: list[float] = []
    cam_ts: list[float] = []

    for r in records:
        if not isinstance(r, dict) or "iter" not in r:
            continue  # skip metadata or malformed entries

        shared = _ts_to_s(r["timestamp"]) if "timestamp" in r else None

        if "spad" in r:
            t = _ts_to_s(r["spad_timestamp"]) if "spad_timestamp" in r else shared
            if t is not None:
                spad_ts.append(t)

        if "realsense" in r:
            t = _ts_to_s(r["realsense_timestamp"]) if "realsense_timestamp" in r else shared
            if t is not None:
                cam_ts.append(t)

    return spad_ts, cam_ts


# ── Nearest-neighbour matching ─────────────────────────────────────────────────

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
    cam_ts: list[float],
) -> list[tuple[int, int, float]]:
    """
    Match every SPAD frame to its nearest camera frame.

    Returns:
        List of (spad_idx, cam_idx, dt_s) sorted by spad_idx, where dt_s is
        the absolute time difference in seconds.
    """
    # Build a sorted view of camera timestamps for binary search.
    sorted_cam_order = sorted(range(len(cam_ts)), key=lambda i: cam_ts[i])
    sorted_cam_ts = [cam_ts[i] for i in sorted_cam_order]

    matches = []
    for si, st in enumerate(spad_ts):
        pool_pos = _nearest_pool_idx(sorted_cam_ts, st)
        ci = sorted_cam_order[pool_pos]
        matches.append((si, ci, abs(st - cam_ts[ci])))

    return matches


# ── Report ─────────────────────────────────────────────────────────────────────

def report(
    n_spad: int,
    n_cam: int,
    matches: list[tuple[int, int, float]],
    spad_ts: list[float],
    max_dt_ms: float,
):
    used_cam = {ci for _, ci, _ in matches}
    n_unmatched_cam = n_cam - len(used_cam)

    sep = "-" * 52
    print(sep)
    print(f"  SPAD frames:            {n_spad:>6}")
    print(f"  Camera frames:          {n_cam:>6}")
    print(sep)
    print(f"  Matched SPAD frames:    {len(matches):>6}  (each -> nearest cam)")
    print(f"  Unique camera matched:  {len(used_cam):>6}")
    print(f"  Unmatched camera:       {n_unmatched_cam:>6}")

    if not matches:
        print(sep)
        return

    dts_ms = [dt * 1000 for _, _, dt in matches]
    mean_dt = sum(dts_ms) / len(dts_ms)

    print(f"\n  Per-pair time differences (ms):")
    for si, ci, dt_ms in [(si, ci, dt * 1000) for si, ci, dt in matches]:
        flag = "  ***" if dt_ms > max_dt_ms else ""
        print(f"    SPAD[{si:4d}] <-> cam[{ci:4d}]  dt = {dt_ms:8.2f}{flag}")
    print(f"\n  Average |dt|: {mean_dt:.2f} ms  (min {min(dts_ms):.2f}, max {max(dts_ms):.2f})")

    # Average interval between consecutive matched pairs (SPAD-side timestamps).
    matched_ts = [spad_ts[si] for si, _, _ in matches]
    if len(matched_ts) > 1:
        intervals_ms = [(matched_ts[i+1] - matched_ts[i]) * 1000
                        for i in range(len(matched_ts) - 1)]
        mean_interval = sum(intervals_ms) / len(intervals_ms)
        print(f"  Avg interval between matched frames: {mean_interval:.2f} ms"
              f"  ({1000/mean_interval:.1f} fps)")

    n_over = sum(1 for dt_ms in dts_ms if dt_ms > max_dt_ms)
    if n_over:
        print(f"\n  WARNING: {n_over} pair(s) marked *** exceed {max_dt_ms:.0f} ms threshold.")
    else:
        print(f"\n  All pairs within {max_dt_ms:.0f} ms threshold.")

    print(sep)


# ── Index export ───────────────────────────────────────────────────────────────

def save_index(
    pkl_path: Path,
    matches: list[tuple[int, int, float]],
    spad_ts: list[float],
    cam_ts: list[float],
    max_dt_ms: float,
) -> Path:
    """
    Write a JSON sync index alongside *pkl_path*.

    The index contains only indices and timing metadata — no sensor data — so
    the training pipeline can load matched pairs on demand from the raw pkl
    without duplicating arrays.

    Returns the path of the written file.
    """
    index_path = pkl_path.with_name(pkl_path.stem + "_sync.json")

    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts).isoformat()

    dts_ms = [dt * 1000 for _, _, dt in matches]
    pairs = [
        {
            "spad_idx": si,
            "cam_idx": ci,
            "dt_ms": round(dt_ms, 3),
            "spad_ts": _iso(spad_ts[si]),
            "cam_ts": _iso(cam_ts[ci]),
        }
        for (si, ci, _), dt_ms in zip(matches, dts_ms)
    ]

    index = {
        "source_pkl": str(pkl_path.resolve()),
        "created": datetime.now().isoformat(),
        "max_dt_ms": max_dt_ms,
        "n_spad": len(spad_ts),
        "n_cam": len(cam_ts),
        "n_pairs": len(pairs),
        "n_pairs_over_threshold": sum(1 for dt_ms in dts_ms if dt_ms > max_dt_ms),
        "pairs": pairs,
    }

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return index_path


# ── Public API ─────────────────────────────────────────────────────────────────

def sync(pkl_path: Path, max_dt_ms: float = MAX_DT_MS, save: bool = True) -> list[tuple[int, int, float]]:
    """
    Load *pkl_path*, match SPAD and camera frames by timestamp, print a
    summary report, and (if *save* is True) write a JSON index alongside the
    pkl.

    Returns:
        List of (spad_idx, cam_idx, dt_s) matched pairs.
    """
    print(f"\nFile: {pkl_path}")
    records = load_records(pkl_path)

    n_data = sum(1 for r in records if isinstance(r, dict) and "iter" in r)
    n_meta = len(records) - n_data
    print(f"Loaded {len(records)} records  ({n_data} data, {n_meta} metadata)\n")

    spad_ts, cam_ts = _extract_frames(records)
    n_spad, n_cam = len(spad_ts), len(cam_ts)

    if n_spad == 0 and n_cam == 0:
        print("No sensor frames found.")
        return []
    if n_spad == 0:
        print(f"No SPAD frames found; {n_cam} camera frames available.")
        return []
    if n_cam == 0:
        print(f"No camera frames found; {n_spad} SPAD frames available.")
        return []

    matches = match_frames(spad_ts, cam_ts)
    report(n_spad, n_cam, matches, spad_ts, max_dt_ms)

    if save:
        index_path = save_index(pkl_path, matches, spad_ts, cam_ts, max_dt_ms)
        print(f"Index saved: {index_path}")

    return matches


# ── CLI ────────────────────────────────────────────────────────────────────────

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
