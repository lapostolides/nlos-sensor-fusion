#!/usr/bin/env python3
"""
playback.py - Play back a recorded capture PKL file.

Shows sensor_cam (RGB + depth) and overhead_cam at the original capture
rate.  With --gt, YOLO detections are pre-computed and overlaid on the
overhead_cam window.

Keyboard controls:

    Space       pause / resume
    ,  /  .     step one frame backward / forward  (while paused)
    [  /  ]     0.5× / 2× playback speed
    q  or Esc   quit

Usage:
    python playback.py [path/to/capture.pkl] [--speed 1.0] [--gt]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import cloudpickle as pickle  # type: ignore[import-untyped]
except ImportError:
    import pickle


# ── Load ──────────────────────────────────────────────────────────────────────

def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, "rb") as f:
        try:
            while True:
                records.append(pickle.load(f))
        except EOFError:
            pass
    # Drop header / non-data records
    return [r for r in records if isinstance(r, dict) and "iter" in r]


# ── Rendering ─────────────────────────────────────────────────────────────────

def _depth_colormap(depth: np.ndarray) -> np.ndarray:
    """Convert uint16 depth (mm) to a JET colormap BGR image."""
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


# ── Ground-truth I/O and overlay ──────────────────────────────────────────────

def load_gt_for_playback(pkl_path: Path) -> dict[int, list] | None:
    """
    Load the sidecar _gt.json for *pkl_path*.  Returns None (with a hint)
    if the file doesn't exist yet.
    """
    from ground_truth import load_gt
    result = load_gt(pkl_path)
    if result is None:
        print(
            f"\nNo GT file found for {pkl_path.name}.\n"
            f"Run first:  python ground_truth.py {pkl_path}\n"
        )
    return result


def _draw_detections(
    img: np.ndarray,
    locations: list,
    src_w: int,
    src_h: int,
) -> np.ndarray:
    """
    Draw YOLO person detections onto *img*.

    Pixel coordinates from the detector (src_w × src_h space) are scaled to
    match the display image size, so this works regardless of resize factor.
    """
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

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Centre crosshair
        r = 6
        cv2.line(out, (pcx - r, pcy), (pcx + r, pcy), (0, 255, 0), 2)
        cv2.line(out, (pcx, pcy - r), (pcx, pcy + r), (0, 255, 0), 2)

        # Label
        tid = f"ID:{loc.track_id}" if loc.track_id is not None else "ID:?"
        label = f"{tid}  {loc.confidence:.0%}"
        lx, ly = x1, max(y1 - 6, 14)
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)

    return out




# ── Playback ──────────────────────────────────────────────────────────────────

SENSOR_WIN  = "sensor_cam  [space=pause  , . = step  [ ] = speed  q=quit]"
OVERHEAD_WIN = "overhead_cam"


def play(
    records: list[dict],
    speed: float,
    detections: dict[int, list] | None = None,  # iter -> list[PersonLocation]
    ov_src_w: int = 1920,
    ov_src_h: int = 1080,
):
    if not records:
        print("No data records found.")
        return

    n = len(records)

    # Infer per-frame interval from timestamps (fall back to 33 ms)
    def _ts(r: dict) -> float | None:
        for key in ("spad_timestamp", "sensor_cam_timestamp", "overhead_cam_timestamp"):
            if key in r:
                return datetime.fromisoformat(r[key]).timestamp()
        return None

    timestamps = [_ts(r) for r in records]
    intervals: list[float] = [
        timestamps[i+1] - timestamps[i]   # type: ignore[operator]
        for i in range(n - 1)
        if timestamps[i] is not None and timestamps[i+1] is not None
    ]
    base_interval = (sum(intervals) / len(intervals)) if intervals else 0.033

    # Determine overhead display size (half of native to fit on screen)
    _ov_sample = next(
        (r["overhead_cam"]["raw_rgb"] for r in records if "overhead_cam" in r), None
    )
    if _ov_sample is not None:
        oh, ow = _ov_sample.shape[:2]
        ov_disp_w, ov_disp_h = ow // 2, oh // 2
    else:
        ov_disp_w, ov_disp_h = 960, 540

    # Create windows
    has_sensor   = any("sensor_cam"   in r for r in records)
    has_overhead = any("overhead_cam" in r for r in records)

    sc_h, sc_w, has_depth = 480, 848, False
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
          f"  ({1/base_interval:.1f} fps)  |  speed ×{speed:.1f}")
    print("  Space=pause  ,/.=step  [/]=speed  q/Esc=quit\n")

    while True:
        rec = records[idx]

        # ── Build frames ──
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

        # ── Keyboard ──
        interval_ms = max(1, int(base_interval / speed * 1000))
        key = cv2.waitKey(interval_ms if not paused else 30) & 0xFF

        if key in (ord('q'), 27):           # q or Esc
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord(',') and paused:    # step back
            idx = max(0, idx - 1)
            continue
        elif key == ord('.') and paused:    # step forward
            idx = min(n - 1, idx + 1)
            continue
        elif key == ord('['):               # slower
            speed = max(0.125, speed * 0.5)
            print(f"Speed: ×{speed:.3f}")
        elif key == ord(']'):               # faster
            speed = min(16.0, speed * 2.0)
            print(f"Speed: ×{speed:.3f}")

        if not paused:
            idx += 1
            if idx >= n:
                print("End of recording.")
                break

    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Play back a capture PKL file.")
    parser.add_argument("pkl", nargs="?", help="Path to capture PKL file.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0).")
    parser.add_argument("--gt", action="store_true",
                        help="Overlay ground-truth detections from the _gt.json sidecar.")
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
