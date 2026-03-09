#!/usr/bin/env python3
"""
playback.py - Play back a recorded capture run.

Supports two formats:
  New:    python playback.py data/logs/my-run/          # directory with manifest.json
  Legacy: python playback.py data/logs/my-run/data.pkl  # PKL file

Keyboard controls:
    Space       pause / resume
    ,  /  .     step one frame backward / forward  (while paused)
    [  /  ]     0.5x / 2x playback speed
    q  or Esc   quit

Usage:
    python playback.py [path] [--speed 1.0] [--gt]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import cloudpickle as pickle
except ImportError:
    import pickle


# ── Load — new manifest format ───────────────────────────────────────────────

def _load_manifest_records(run_dir: Path) -> list[dict]:
    """Build playback-compatible records from the new directory layout."""
    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)

    frames = manifest.get("frames", {})
    spad_count = frames.get("spad", {}).get("count", 0)
    sc_count = frames.get("sensor_cam", {}).get("count", 0)
    ov_count = frames.get("overhead_cam", {}).get("count", 0)
    n = max(spad_count, sc_count, ov_count)

    sc_ts = frames.get("sensor_cam", {}).get("timestamps", [])
    ov_ts = frames.get("overhead_cam", {}).get("timestamps", [])
    spad_ts = frames.get("spad", {}).get("timestamps", [])

    sc_rgb_dir = run_dir / "sensor_cam" / "rgb"
    sc_depth_dir = run_dir / "sensor_cam" / "depth"
    ov_dir = run_dir / "overhead_cam"

    records = []
    for i in range(n):
        rec: dict = {"iter": i}

        # Timestamp (use first available for pacing)
        if i < len(sc_ts):
            rec["sensor_cam_timestamp"] = datetime.fromtimestamp(sc_ts[i]).isoformat()
        elif i < len(ov_ts):
            rec["overhead_cam_timestamp"] = datetime.fromtimestamp(ov_ts[i]).isoformat()
        elif i < len(spad_ts):
            rec["spad_timestamp"] = datetime.fromtimestamp(spad_ts[i]).isoformat()

        # sensor_cam
        fname = f"{i:06d}"
        rgb_path = sc_rgb_dir / f"{fname}.jpg"
        depth_path = sc_depth_dir / f"{fname}.png"
        if rgb_path.exists():
            sc: dict = {"raw_rgb": cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)}
            if depth_path.exists():
                sc["aligned_depth"] = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            rec["sensor_cam"] = sc

        # overhead_cam
        ov_path = ov_dir / f"{fname}.jpg"
        if ov_path.exists():
            rec["overhead_cam"] = {
                "raw_rgb": cv2.imread(str(ov_path), cv2.IMREAD_COLOR)
            }

        records.append(rec)

    return records


# ── Load — legacy PKL format ────────────────────────────────────────────────

def _load_pkl_records(path: Path) -> list[dict]:
    records = []
    with open(path, "rb") as f:
        try:
            while True:
                records.append(pickle.load(f))
        except EOFError:
            pass
    return [r for r in records if isinstance(r, dict) and "iter" in r]


def load_records(path: Path) -> list[dict]:
    """Load records from a run directory or legacy PKL file."""
    if path.is_dir() and (path / "manifest.json").exists():
        return _load_manifest_records(path)
    if path.is_file() and path.suffix == ".pkl":
        return _load_pkl_records(path)
    # Try as directory even without manifest (maybe only partial data)
    if path.is_dir():
        print(f"Warning: no manifest.json in {path}, trying PKL fallback...")
        pkls = sorted(path.glob("*.pkl"))
        if pkls:
            return _load_pkl_records(pkls[0])
    print(f"Error: cannot load {path}")
    sys.exit(1)


# ── Rendering ────────────────────────────────────────────────────────────────

def _depth_colormap(depth: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=0.03),
        cv2.COLORMAP_JET,
    )


def _overlay_text(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    y = 22
    for line in lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return out


def build_sensor_cam_frame(rec: dict) -> np.ndarray | None:
    sc = rec.get("sensor_cam")
    if sc is None:
        return None
    rgb   = sc.get("raw_rgb")
    depth = sc.get("aligned_depth")
    if rgb is None:
        return None
    if depth is not None:
        return np.hstack([rgb, _depth_colormap(depth)])
    return rgb


def build_overhead_cam_frame(rec: dict, display_w: int, display_h: int) -> np.ndarray | None:
    ov = rec.get("overhead_cam")
    if ov is None:
        return None
    rgb = ov.get("raw_rgb")
    if rgb is None:
        return None
    return cv2.resize(rgb, (display_w, display_h))


# ── Ground-truth I/O and overlay ─────────────────────────────────────────────

def load_gt_for_playback(path: Path) -> dict[int, list] | None:
    from ground_truth import load_gt
    result = load_gt(path)
    if result is None:
        print(
            f"\nNo GT file found for {path.name}.\n"
            f"Run first:  python ground_truth.py {path}\n"
        )
    return result


def _draw_detections(
    img: np.ndarray,
    locations: list,
    src_w: int,
    src_h: int,
) -> np.ndarray:
    out = img.copy()
    dw, dh = out.shape[1], out.shape[0]
    sx, sy = dw / src_w, dh / src_h

    for loc in locations:
        bx, by, bw, bh = loc.bbox
        cx, cy = loc.center

        x1 = int(bx * sx)
        y1 = int(by * sy)
        x2 = int((bx + bw) * sx)
        y2 = int((by + bh) * sy)
        pcx = int(cx * sx)
        pcy = int(cy * sy)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        r = 6
        cv2.line(out, (pcx - r, pcy), (pcx + r, pcy), (0, 255, 0), 2)
        cv2.line(out, (pcx, pcy - r), (pcx, pcy + r), (0, 255, 0), 2)

        tid = f"ID:{loc.track_id}" if loc.track_id is not None else "ID:?"
        label = f"{tid}  {loc.confidence:.0%}"
        lx, ly = x1, max(y1 - 6, 14)
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)

        pose = getattr(loc, "pose", None)
        if pose is not None:
            pose_conf = getattr(loc, "pose_conf", 0.0)
            pose_label = f"{pose}  {pose_conf:.0%}"
            ply = min(y1 + 18, y2 - 4)
            cv2.putText(out, pose_label, (x1 + 4, ply), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, pose_label, (x1 + 4, ply), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (0, 220, 255), 1, cv2.LINE_AA)

    return out


# ── Playback ─────────────────────────────────────────────────────────────────

SENSOR_WIN   = "sensor_cam  [space=pause  , . = step  [ ] = speed  q=quit]"
OVERHEAD_WIN = "overhead_cam"


def play(
    records: list[dict],
    speed: float,
    detections: dict[int, list] | None = None,
    ov_src_w: int = 1920,
    ov_src_h: int = 1080,
):
    if not records:
        print("No data records found.")
        return

    n = len(records)

    def _ts(r: dict) -> float | None:
        for key in ("spad_timestamp", "sensor_cam_timestamp", "overhead_cam_timestamp"):
            if key in r:
                return datetime.fromisoformat(r[key]).timestamp()
        return None

    timestamps = [_ts(r) for r in records]
    intervals: list[float] = [
        timestamps[i+1] - timestamps[i]
        for i in range(n - 1)
        if timestamps[i] is not None and timestamps[i+1] is not None
    ]
    base_interval = (sum(intervals) / len(intervals)) if intervals else 0.033

    _ov_sample = next(
        (r["overhead_cam"]["raw_rgb"] for r in records if "overhead_cam" in r), None
    )
    if _ov_sample is not None:
        oh, ow = _ov_sample.shape[:2]
        ov_disp_w, ov_disp_h = ow // 2, oh // 2
    else:
        ov_disp_w, ov_disp_h = 960, 540

    has_sensor   = any("sensor_cam"   in r for r in records)
    has_overhead = any("overhead_cam" in r for r in records)

    sc_h, sc_w = 480, 848
    if has_sensor:
        sc0 = next(r for r in records if "sensor_cam" in r)
        sc_h = sc0["sensor_cam"]["raw_rgb"].shape[0]
        sc_w = sc0["sensor_cam"]["raw_rgb"].shape[1]
        has_depth = "aligned_depth" in sc0["sensor_cam"]
        cv2.namedWindow(SENSOR_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(SENSOR_WIN, sc_w * (2 if has_depth else 1), sc_h)
        cv2.moveWindow(SENSOR_WIN, 10, 10)

    if has_overhead:
        cv2.namedWindow(OVERHEAD_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(OVERHEAD_WIN, ov_disp_w, ov_disp_h)
        cv2.moveWindow(OVERHEAD_WIN, 10, 10 + sc_h + 40)

    idx    = 0
    paused = False

    print(f"Playing {n} frames  |  base interval {base_interval*1000:.1f} ms"
          f"  ({1/base_interval:.1f} fps)  |  speed x{speed:.1f}")
    print("  Space=pause  ,/.=step  [/]=speed  q/Esc=quit\n")

    while True:
        rec = records[idx]

        t_render = time.perf_counter()
        info_lines = [
            f"frame {idx+1}/{n}",
            f"iter {rec.get('iter','?')}",
            f"speed x{speed:.1f}",
            "PAUSED" if paused else "",
        ]

        if has_sensor:
            frame_sc = build_sensor_cam_frame(rec)
            if frame_sc is None:
                frame_sc = np.zeros((480, 848, 3), dtype=np.uint8)
            cv2.imshow(SENSOR_WIN, _overlay_text(frame_sc, info_lines))

        if has_overhead:
            frame_ov = build_overhead_cam_frame(rec, ov_disp_w, ov_disp_h)
            if frame_ov is None:
                frame_ov = np.zeros((ov_disp_h, ov_disp_w, 3), dtype=np.uint8)
            if detections is not None:
                locs = detections.get(rec.get("iter", -1), [])
                if locs:
                    frame_ov = _draw_detections(frame_ov, locs, ov_src_w, ov_src_h)
            cv2.imshow(OVERHEAD_WIN, frame_ov)

        render_ms   = (time.perf_counter() - t_render) * 1000
        interval_ms = max(1, int(base_interval / speed * 1000) - int(render_ms))
        key = cv2.waitKey(interval_ms if not paused else 30) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord(',') and paused:
            idx = max(0, idx - 1)
            continue
        elif key == ord('.') and paused:
            idx = min(n - 1, idx + 1)
            continue
        elif key == ord('['):
            speed = max(0.125, speed * 0.5)
            print(f"Speed: x{speed:.3f}")
        elif key == ord(']'):
            speed = min(16.0, speed * 2.0)
            print(f"Speed: x{speed:.3f}")

        if not paused:
            idx += 1
            if idx >= n:
                print("End of recording.")
                break

    cv2.destroyAllWindows()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Play back a capture run.")
    parser.add_argument(
        "path", nargs="?",
        help="Run directory (with manifest.json) or legacy .pkl file.",
    )
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0).")
    parser.add_argument("--gt", action="store_true",
                        help="Overlay ground-truth detections.")
    args = parser.parse_args()

    if args.path:
        path = Path(args.path)
        if not path.exists():
            print(f"Error: {path} does not exist.")
            sys.exit(1)
    else:
        # Auto-detect: most recent run across both formats
        candidates: list[Path] = []
        for mf in Path("data/logs").rglob("manifest.json"):
            candidates.append(mf.parent)
        for pk in Path("data/logs").rglob("*.pkl"):
            candidates.append(pk)
        if not candidates:
            print("No runs found in data/logs/.")
            sys.exit(1)
        path = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"Auto-detected: {path}")

    print(f"Loading {path} ...")
    records = load_records(path)
    print(f"Loaded {len(records)} data records.")

    detections: dict[int, list] | None = None
    ov_src_w, ov_src_h = 1920, 1080
    if args.gt:
        ov_sample = next(
            (r["overhead_cam"]["raw_rgb"] for r in records if "overhead_cam" in r), None
        )
        if ov_sample is not None:
            ov_src_h, ov_src_w = ov_sample.shape[:2]
        detections = load_gt_for_playback(path)

    play(records, speed=args.speed, detections=detections,
         ov_src_w=ov_src_w, ov_src_h=ov_src_h)


if __name__ == "__main__":
    main()
