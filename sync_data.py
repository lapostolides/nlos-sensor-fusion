#!/usr/bin/env python3
"""
sync_data.py - Post-hoc temporal synchronization of SPAD, camera, and UWB frames.

Supports two input formats:

  New (manifest.json):
    python sync_data.py data/logs/my-run/
    python sync_data.py                              # auto-detect latest run dir

  Legacy (PKL):
    python sync_data.py data/logs/my-run/data.pkl    # explicit PKL path
    python sync_data.py --uwb-dir data/logs/my-run/  # separate UWB dir

Matches every SPAD frame to the nearest frame from each camera and UWB RX
board independently.  Writes a sync.json index in the run directory.
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

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

MAX_DT_MS = 100.0

_CAM_DATA_KEYS: dict[str, str] = {
    "sensor_cam":   "sensor_cam",
    "realsense":    "sensor_cam",
    "overhead_cam": "overhead_cam",
}
_CAM_TS_KEYS: dict[str, str] = {
    "sensor_cam":   "sensor_cam_timestamp",
    "realsense":    "realsense_timestamp",
    "overhead_cam": "overhead_cam_timestamp",
}

_UWB_RX_ROLES = ("rx1", "rx2", "rx3")
_UWB_TX_ROLE  = "tx"


# ── I/O — new manifest format ────────────────────────────────────────────────

def _load_manifest_timestamps(
    run_dir: Path,
) -> tuple[list[float], dict[str, list[float]]]:
    """Load SPAD and camera timestamps from manifest.json."""
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    frames = manifest.get("frames", {})
    spad_ts: list[float] = frames.get("spad", {}).get("timestamps", [])

    cam_ts: dict[str, list[float]] = {}
    for cam_name in ("sensor_cam", "overhead_cam"):
        ts = frames.get(cam_name, {}).get("timestamps")
        if ts:
            cam_ts[cam_name] = ts

    return spad_ts, cam_ts


# ── I/O — legacy PKL format ─────────────────────────────────────────────────

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
    if isinstance(ts, (int, float)):
        return float(ts)
    return datetime.fromisoformat(str(ts)).timestamp()


def _extract_frames_pkl(
    records: list,
) -> tuple[list[float], dict[str, list[float]]]:
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


# ── UWB loading ──────────────────────────────────────────────────────────────

def load_uwb_timestamps(
    uwb_dir: Path,
) -> tuple[dict[str, list[float]], list[float] | None]:
    if not _NUMPY:
        print("  [uwb] numpy not available — skipping UWB sync.")
        return {}, None

    rx_ts: dict[str, list[float]] = {}
    tx_ts: list[float] | None = None

    for role in (_UWB_TX_ROLE, *_UWB_RX_ROLES):
        path = uwb_dir / f"{role}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        if "timestamp_wall_start" not in data.files:
            print(
                f"  [uwb/{role}] missing timestamp_wall_start — "
                "re-capture with the updated capture_uwb.py to enable UWB sync"
            )
            continue
        wall_start = float(data["timestamp_wall_start"])
        abs_ts = [wall_start + float(t) for t in data["timestamp"]]
        if role == _UWB_TX_ROLE:
            tx_ts = abs_ts
        else:
            rx_ts[role] = abs_ts
        print(f"  [uwb/{role}] {len(abs_ts)} frames loaded from {path.name}")

    return rx_ts, tx_ts


# ── Nearest-neighbour matching ───────────────────────────────────────────────

def _nearest_pool_idx(sorted_pool: list[float], q: float) -> int:
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


def match_uwb_tx_rx(
    tx_ts: list[float],
    rx_ts: dict[str, list[float]],
) -> dict[str, list[tuple[int, int, float]]]:
    results: dict[str, list[tuple[int, int, float]]] = {}
    sorted_tx = sorted(range(len(tx_ts)), key=lambda i: tx_ts[i])
    sorted_tx_vals = [tx_ts[i] for i in sorted_tx]

    for role, rts in rx_ts.items():
        matches = []
        for ri, rt in enumerate(rts):
            pool_pos = _nearest_pool_idx(sorted_tx_vals, rt)
            ti = sorted_tx[pool_pos]
            matches.append((ri, ti, abs(rt - tx_ts[ti])))
        results[role] = matches

    return results


# ── Report ───────────────────────────────────────────────────────────────────

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


def _uwb_tx_rx_report(
    tx_rx_matches: dict[str, list[tuple[int, int, float]]],
    rx_ts: dict[str, list[float]],
    tx_n: int,
):
    print(f"\n  [uwb TX->RX]  {tx_n} TX frames")
    for role, matches in tx_rx_matches.items():
        n_rx = len(rx_ts[role])
        dts_ms = [dt * 1000 for _, _, dt in matches]
        mean_dt = sum(dts_ms) / len(dts_ms) if dts_ms else 0.0
        print(f"    {role}: {n_rx} RX frames -> {len(matches)} matched "
              f"| avg |dt|: {mean_dt:.2f} ms  "
              f"(max {max(dts_ms):.2f} ms)" if dts_ms else f"    {role}: no matches")


def report(
    n_spad: int,
    cam_ts: dict[str, list[float]],
    all_matches: dict[str, list[tuple[int, int, float]]],
    spad_ts: list[float],
    max_dt_ms: float,
    uwb_rx_ts: dict[str, list[float]] | None = None,
    uwb_tx_ts: list[float] | None = None,
    uwb_tx_rx_matches: dict[str, list[tuple[int, int, float]]] | None = None,
):
    sep = "-" * 52
    print(sep)
    print(f"  SPAD frames : {n_spad}")

    for cam_name, matches in all_matches.items():
        n_frames = len(uwb_rx_ts[cam_name.replace("uwb_", "")]) \
            if (uwb_rx_ts and cam_name.startswith("uwb_")) \
            else len(cam_ts.get(cam_name, []))
        _camera_report(cam_name, n_spad, n_frames, matches, max_dt_ms)

    if uwb_tx_rx_matches and uwb_rx_ts and uwb_tx_ts:
        _uwb_tx_rx_report(uwb_tx_rx_matches, uwb_rx_ts, len(uwb_tx_ts))

    if n_spad > 1:
        intervals_ms = [(spad_ts[i+1] - spad_ts[i]) * 1000
                        for i in range(len(spad_ts) - 1)]
        mean_iv = sum(intervals_ms) / len(intervals_ms)
        print(f"\n  SPAD avg interval : {mean_iv:.2f} ms  ({1000/mean_iv:.1f} fps)")

    print(sep)


# ── Index export ─────────────────────────────────────────────────────────────

def save_index(
    run_dir: Path,
    all_matches: dict[str, list[tuple[int, int, float]]],
    spad_ts: list[float],
    cam_ts: dict[str, list[float]],
    max_dt_ms: float,
    uwb_rx_ts: dict[str, list[float]] | None = None,
    uwb_tx_ts: list[float] | None = None,
    uwb_tx_rx_matches: dict[str, list[tuple[int, int, float]]] | None = None,
) -> Path:
    """Write sync.json in *run_dir*."""
    index_path = run_dir / "sync.json"

    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts).isoformat()

    n_spad = len(spad_ts)

    source_lookup: dict[str, dict[int, dict]] = {}
    for name, matches in all_matches.items():
        if name.startswith("uwb_"):
            role = name.replace("uwb_", "")
            ts_list = (uwb_rx_ts or {}).get(role, [])
        else:
            ts_list = cam_ts.get(name, [])
        source_lookup[name] = {
            si: {"idx": ci, "dt_ms": round(dt * 1000, 3),
                 "ts": _iso(ts_list[ci]) if ci < len(ts_list) else None}
            for si, ci, dt in matches
        }

    pairs = []
    for si in range(n_spad):
        entry: dict = {"spad_idx": si, "spad_ts": _iso(spad_ts[si])}
        for name, lookup in source_lookup.items():
            if si in lookup:
                entry[name] = lookup[si]
        pairs.append(entry)

    camera_stats = {}
    for cam_name, matches in all_matches.items():
        if cam_name.startswith("uwb_"):
            continue
        dts_ms = [dt * 1000 for _, _, dt in matches]
        camera_stats[cam_name] = {
            "n_frames": len(cam_ts.get(cam_name, [])),
            "n_matched": len(matches),
            "n_over_threshold": sum(1 for dt_ms in dts_ms if dt_ms > max_dt_ms),
            "mean_dt_ms": round(sum(dts_ms) / len(dts_ms), 3) if dts_ms else None,
        }

    uwb_rx_stats = {}
    for cam_name, matches in all_matches.items():
        if not cam_name.startswith("uwb_"):
            continue
        role = cam_name.replace("uwb_", "")
        dts_ms = [dt * 1000 for _, _, dt in matches]
        used = {ci for _, ci, _ in matches}
        uwb_rx_stats[role] = {
            "n_frames": len((uwb_rx_ts or {}).get(role, [])),
            "n_spad_matched": len(matches),
            "unique_uwb_matched": len(used),
            "n_over_threshold": sum(1 for dt_ms in dts_ms if dt_ms > max_dt_ms),
            "mean_dt_ms": round(sum(dts_ms) / len(dts_ms), 3) if dts_ms else None,
        }

    uwb_tx_rx_stats = {}
    if uwb_tx_rx_matches and uwb_tx_ts:
        for role, matches in uwb_tx_rx_matches.items():
            dts_ms = [dt * 1000 for _, _, dt in matches]
            uwb_tx_rx_stats[role] = {
                "n_tx_frames": len(uwb_tx_ts),
                "n_rx_frames": len((uwb_rx_ts or {}).get(role, [])),
                "n_matched": len(matches),
                "mean_dt_ms": round(sum(dts_ms) / len(dts_ms), 3) if dts_ms else None,
                "pairs": [
                    {"rx_idx": ri, "tx_idx": ti, "dt_ms": round(dt * 1000, 3)}
                    for ri, ti, dt in matches
                ],
            }

    index = {
        "source": str(run_dir.resolve()),
        "created": datetime.now().isoformat(),
        "max_dt_ms": max_dt_ms,
        "n_spad": n_spad,
        "cameras": camera_stats,
        "uwb_rx": uwb_rx_stats,
        "uwb_tx_rx": uwb_tx_rx_stats,
        "pairs": pairs,
    }

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return index_path


# ── Public API ───────────────────────────────────────────────────────────────

def sync(
    run_dir: Path,
    max_dt_ms: float = MAX_DT_MS,
    save: bool = True,
    uwb_dir: Path | None = None,
) -> dict[str, list[tuple[int, int, float]]]:
    """
    Synchronise sensor data from a run directory (manifest.json) or
    legacy PKL file.

    When *run_dir* points to a directory with manifest.json, timestamps are
    read directly from the manifest.  UWB data is auto-detected in the same
    directory.

    When *run_dir* points to a .pkl file, falls back to legacy PKL parsing.
    """
    # Detect format
    is_pkl = run_dir.is_file() and run_dir.suffix == ".pkl"

    if is_pkl:
        print(f"\nFile: {run_dir} (legacy PKL)")
        records = load_records(run_dir)
        n_data = sum(1 for r in records if isinstance(r, dict) and "iter" in r)
        print(f"Loaded {len(records)} records  ({n_data} data)\n")
        spad_ts, cam_ts = _extract_frames_pkl(records)
        effective_dir = run_dir.parent
    else:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"Error: {manifest_path} not found.")
            return {}
        print(f"\nRun: {run_dir} (manifest.json)")
        spad_ts, cam_ts = _load_manifest_timestamps(run_dir)
        print(f"  SPAD: {len(spad_ts)} frames")
        for name, ts in cam_ts.items():
            print(f"  {name}: {len(ts)} frames")
        effective_dir = run_dir

    # UWB
    uwb_rx_ts: dict[str, list[float]] = {}
    uwb_tx_ts: list[float] | None = None
    uwb_tx_rx_matches: dict[str, list[tuple[int, int, float]]] = {}

    if uwb_dir is None and any(effective_dir.glob("rx*.npz")):
        uwb_dir = effective_dir
        print(f"  Auto-detected UWB data in {uwb_dir}")

    if uwb_dir is not None:
        uwb_rx_ts, uwb_tx_ts = load_uwb_timestamps(uwb_dir)
        for role, ts_list in uwb_rx_ts.items():
            cam_ts[f"uwb_{role}"] = ts_list
        if uwb_tx_ts and uwb_rx_ts:
            uwb_tx_rx_matches = match_uwb_tx_rx(uwb_tx_ts, uwb_rx_ts)

    if not spad_ts:
        print("No SPAD frames found.")
        return {}
    if not cam_ts:
        print(f"No camera or UWB frames found; {len(spad_ts)} SPAD frames available.")
        return {}

    all_matches = match_frames(spad_ts, cam_ts)
    report(
        len(spad_ts), cam_ts, all_matches, spad_ts, max_dt_ms,
        uwb_rx_ts=uwb_rx_ts,
        uwb_tx_ts=uwb_tx_ts,
        uwb_tx_rx_matches=uwb_tx_rx_matches or None,
    )

    if save:
        index_path = save_index(
            effective_dir, all_matches, spad_ts, cam_ts, max_dt_ms,
            uwb_rx_ts=uwb_rx_ts,
            uwb_tx_ts=uwb_tx_ts,
            uwb_tx_rx_matches=uwb_tx_rx_matches or None,
        )
        print(f"Index saved: {index_path}")

    return all_matches


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Synchronise SPAD, camera, and UWB frames by timestamp."
    )
    parser.add_argument(
        "path", nargs="?",
        help="Run directory (with manifest.json) or legacy .pkl file.",
    )
    parser.add_argument(
        "--uwb-dir",
        metavar="DIR",
        help="Directory containing UWB .npz files. Auto-detected if omitted.",
    )
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

    if args.path:
        path = Path(args.path)
        if not path.exists():
            print(f"Error: {path} does not exist.")
            sys.exit(1)
    else:
        # Auto-detect: find the most recent run across both formats.
        # manifest.json dirs → use parent dir; PKL files → use the file itself.
        candidates: list[Path] = []
        for mf in Path("data/logs").rglob("manifest.json"):
            candidates.append(mf.parent)
        for pk in Path("data/logs").rglob("*.pkl"):
            candidates.append(pk)
        if not candidates:
            print("No run directories or PKL files found in data/logs/.")
            sys.exit(1)
        # Sort by modification time, pick the most recent
        path = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"Auto-detected: {path}")

    uwb_dir = Path(args.uwb_dir) if args.uwb_dir else None
    if uwb_dir and not uwb_dir.exists():
        print(f"Error: --uwb-dir {uwb_dir} does not exist.")
        sys.exit(1)

    sync(path, max_dt_ms=args.max_dt_ms, save=not args.no_index, uwb_dir=uwb_dir)


if __name__ == "__main__":
    main()
