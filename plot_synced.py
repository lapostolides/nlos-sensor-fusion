#!/usr/bin/env python3
"""
plot_synced.py — Step through synced multi-sensor frames interactively.

Usage:
    python plot_synced.py                          # auto-detect latest run
    python plot_synced.py data/logs/test5/         # explicit run dir
    python plot_synced.py --start 50               # start at pair index 50

Controls:
    Right / d / Space  — next frame
    Left  / a          — previous frame
    q / Esc            — quit
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


# ── Data loading ────────────────────────────────────────────────────────────

def find_latest_run() -> Path:
    candidates: list[Path] = []
    log_root = Path("data/logs")
    if not log_root.exists():
        print("No data/logs/ directory found.")
        sys.exit(1)
    for mf in log_root.rglob("manifest.json"):
        candidates.append(mf.parent)
    if not candidates:
        print("No run directories found in data/logs/.")
        sys.exit(1)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_run(run_dir: Path):
    """Load all data sources for a run. Returns a dict of arrays/metadata."""
    sync_path = run_dir / "sync.json"
    if not sync_path.exists():
        print(f"Error: {sync_path} not found. Run sync_data.py first.")
        sys.exit(1)

    with open(sync_path) as f:
        sync = json.load(f)

    data = {"run_dir": run_dir, "sync": sync, "pairs": sync["pairs"]}

    # SPAD
    spad_path = run_dir / "spad.npz"
    if spad_path.exists():
        spad = np.load(spad_path)
        data["spad_histograms"] = spad["histograms"]
        data["spad_timestamps"] = spad["timestamps"]
        print(f"  SPAD: {data['spad_histograms'].shape}")
    else:
        data["spad_histograms"] = None
        print("  SPAD: not found")

    # sensor_cam
    data["sensor_cam_rgb_dir"] = run_dir / "sensor_cam" / "rgb"
    data["sensor_cam_depth_dir"] = run_dir / "sensor_cam" / "depth"
    n_rgb = len(list(data["sensor_cam_rgb_dir"].glob("*.jpg"))) if data["sensor_cam_rgb_dir"].exists() else 0
    print(f"  sensor_cam: {n_rgb} frames")

    # overhead_cam
    data["overhead_cam_dir"] = run_dir / "overhead_cam"
    n_oh = len(list(data["overhead_cam_dir"].glob("*.jpg"))) if data["overhead_cam_dir"].exists() else 0
    print(f"  overhead_cam: {n_oh} frames")

    # UWB RX
    for role in ("rx1", "rx2", "rx3"):
        npz_path = run_dir / f"{role}.npz"
        if npz_path.exists():
            rx = np.load(npz_path)
            data[f"{role}_cir_mag"] = rx["cir_mag"] if "cir_mag" in rx.files else None
            data[f"{role}_fp_index"] = rx["fp_index"] if "fp_index" in rx.files else None
            n = rx["cir_mag"].shape[0] if "cir_mag" in rx.files else 0
            print(f"  {role}: {n} frames, CIR len={rx['cir_mag'].shape[1] if 'cir_mag' in rx.files else '?'}")
        else:
            data[f"{role}_cir_mag"] = None
            data[f"{role}_fp_index"] = None
            print(f"  {role}: not found")

    return data


# ── Image loading helpers ───────────────────────────────────────────────────

def load_image(directory: Path, idx: int) -> np.ndarray | None:
    path = directory / f"{idx:06d}.jpg"
    if not path.exists():
        path = directory / f"{idx:06d}.png"
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


# ── Plotting ────────────────────────────────────────────────────────────────

class SyncedViewer:
    def __init__(self, data: dict, start: int = 0):
        self.data = data
        self.pairs = data["pairs"]
        self.idx = max(0, min(start, len(self.pairs) - 1))

        # Single figure with GridSpec layout:
        #   Rows 0-3, Cols 0-3 : SPAD 4x4 histograms
        #   Rows 0-1, Cols 4-5 : sensor_cam RGB
        #   Rows 0-1, Cols 6-7 : sensor_cam depth
        #   Rows 2-3, Cols 4-7 : overhead_cam
        #   Rows 4-6, Cols 0-7 : UWB CIR (3 stacked)
        self.fig = plt.figure(figsize=(20, 14))
        gs = self.fig.add_gridspec(7, 8, hspace=0.45, wspace=0.35)

        # SPAD 4x4
        self.axes_spad = [[self.fig.add_subplot(gs[r, c]) for c in range(4)]
                          for r in range(4)]

        # Camera image axes
        self.ax_rgb = self.fig.add_subplot(gs[0:2, 4:6])
        self.ax_depth = self.fig.add_subplot(gs[0:2, 6:8])
        self.ax_overhead = self.fig.add_subplot(gs[2:4, 4:8])

        # UWB CIR (shared x-axis)
        ax0 = self.fig.add_subplot(gs[4, :])
        self.axes_uwb = [ax0,
                         self.fig.add_subplot(gs[5, :], sharex=ax0),
                         self.fig.add_subplot(gs[6, :], sharex=ax0)]

        self.fig.canvas.manager.set_window_title("Synced Multi-Sensor Viewer")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._draw()
        plt.show()

    def _on_key(self, event):
        if event.key in ("right", "d", " "):
            self.idx = min(self.idx + 1, len(self.pairs) - 1)
            self._draw()
        elif event.key in ("left", "a"):
            self.idx = max(self.idx - 1, 0)
            self._draw()
        elif event.key in ("q", "escape"):
            plt.close("all")

    @staticmethod
    def _clear_image_ax(ax, text="N/A"):
        ax.clear()
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    def _draw(self):
        pair = self.pairs[self.idx]
        spad_idx = pair["spad_idx"]
        title = (f"Pair {self.idx}/{len(self.pairs)-1}  |  "
                 f"SPAD #{spad_idx}  |  ts: {pair['spad_ts']}")

        # ── SPAD histograms ─────────────────────────────────────────────
        if self.data["spad_histograms"] is not None and spad_idx < len(self.data["spad_histograms"]):
            hist = self.data["spad_histograms"][spad_idx]  # (4, 4, bins)
            for r in range(4):
                for c in range(4):
                    ax = self.axes_spad[r][c]
                    ax.clear()
                    ax.bar(range(hist.shape[-1]), hist[r, c], width=1.0, color="steelblue")
                    ax.set_ylim(bottom=0)
                    ax.tick_params(labelsize=6)
                    if r < 3:
                        ax.set_xticklabels([])
        else:
            for r in range(4):
                for c in range(4):
                    self.axes_spad[r][c].clear()
                    self.axes_spad[r][c].text(0.5, 0.5, "N/A", ha="center", va="center",
                                               transform=self.axes_spad[r][c].transAxes)

        # ── Camera frames ───────────────────────────────────────────────
        sc = pair.get("sensor_cam")
        if sc is not None:
            rgb = load_image(self.data["sensor_cam_rgb_dir"], sc["idx"])
            if rgb is not None:
                self.ax_rgb.clear()
                self.ax_rgb.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), aspect="auto")
                self.ax_rgb.set_title(f"sensor_cam #{sc['idx']}  dt={sc['dt_ms']:.1f}ms", fontsize=8)
                self.ax_rgb.set_xticks([])
                self.ax_rgb.set_yticks([])
            else:
                self._clear_image_ax(self.ax_rgb, "RGB: no file")

            depth = load_image(self.data["sensor_cam_depth_dir"], sc["idx"])
            if depth is not None:
                self.ax_depth.clear()
                if depth.dtype == np.uint16:
                    disp = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    disp = cv2.applyColorMap(disp, cv2.COLORMAP_TURBO)
                    disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                else:
                    disp = depth if depth.ndim == 2 else cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
                self.ax_depth.imshow(disp, aspect="auto")
                self.ax_depth.set_title(f"depth #{sc['idx']}", fontsize=8)
                self.ax_depth.set_xticks([])
                self.ax_depth.set_yticks([])
            else:
                self._clear_image_ax(self.ax_depth, "Depth: no file")
        else:
            self._clear_image_ax(self.ax_rgb, "sensor_cam: N/A")
            self._clear_image_ax(self.ax_depth, "depth: N/A")

        oc = pair.get("overhead_cam")
        if oc is not None:
            oh = load_image(self.data["overhead_cam_dir"], oc["idx"])
            if oh is not None:
                self.ax_overhead.clear()
                self.ax_overhead.imshow(cv2.cvtColor(oh, cv2.COLOR_BGR2RGB), aspect="auto")
                self.ax_overhead.set_title(f"overhead_cam #{oc['idx']}  dt={oc['dt_ms']:.1f}ms", fontsize=8)
                self.ax_overhead.set_xticks([])
                self.ax_overhead.set_yticks([])
            else:
                self._clear_image_ax(self.ax_overhead, "overhead: no file")
        else:
            self._clear_image_ax(self.ax_overhead, "overhead_cam: N/A")

        # ── UWB CIR ────────────────────────────────────────────────────
        for i, role in enumerate(("rx1", "rx2", "rx3")):
            ax = self.axes_uwb[i]
            ax.clear()

            uwb_key = f"uwb_{role}"
            cir_mag = self.data.get(f"{role}_cir_mag")
            fp_index = self.data.get(f"{role}_fp_index")

            if uwb_key in pair and cir_mag is not None:
                uwb_idx = pair[uwb_key]["idx"]
                dt_ms = pair[uwb_key]["dt_ms"]
                if uwb_idx < len(cir_mag):
                    mag = cir_mag[uwb_idx]
                    ax.plot(mag, linewidth=0.7, color="tab:blue")
                    if fp_index is not None and uwb_idx < len(fp_index):
                        fp = int(fp_index[uwb_idx])
                        ax.axvline(fp, color="red", linewidth=0.8, linestyle="--", label=f"FP={fp}")
                        ax.legend(fontsize=7, loc="upper right")
                    ax.set_title(f"{role}  idx={uwb_idx}  dt={dt_ms:.1f}ms", fontsize=9)
                else:
                    ax.text(0.5, 0.5, f"{role}: idx {uwb_idx} out of range",
                            ha="center", va="center", transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f"{role}: no match", ha="center", va="center",
                        transform=ax.transAxes)

            ax.set_ylabel("Magnitude", fontsize=8)
        self.axes_uwb[-1].set_xlabel("CIR Sample Index", fontsize=8)

        self.fig.suptitle(title, fontsize=12)
        self.fig.canvas.draw_idle()


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step through synced multi-sensor frames interactively."
    )
    parser.add_argument(
        "path", nargs="?",
        help="Run directory (with sync.json). Auto-detects latest if omitted.",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Starting pair index (default: 0).",
    )
    args = parser.parse_args()

    if args.path:
        run_dir = Path(args.path)
        if not run_dir.exists():
            print(f"Error: {run_dir} does not exist.")
            sys.exit(1)
    else:
        run_dir = find_latest_run()
        print(f"Auto-detected: {run_dir}")

    print(f"Loading {run_dir} ...")
    data = load_run(run_dir)
    print(f"\n{len(data['pairs'])} synced pairs. Use arrow keys to navigate, q to quit.\n")

    SyncedViewer(data, start=args.start)


if __name__ == "__main__":
    main()
