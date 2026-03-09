"""
Offline CIR (Channel Impulse Response) plotter for captured .npz logs.

Modes:
  step      (default)  Step through frames one at a time.
  waterfall            Full waterfall heatmap of the entire log.
  compare              Overlay rx1/rx2/rx3 from one run directory on one plot.

Usage:
  python uwb_cir_plotter.py                              # auto-detect latest log, step mode
  python uwb_cir_plotter.py path/to/cir.npz              # explicit log, step mode
  python uwb_cir_plotter.py --waterfall                  # waterfall of latest log
  python uwb_cir_plotter.py path/to/cir.npz --waterfall
  python uwb_cir_plotter.py --compare                    # compare rx1/rx2/rx3, latest run
  python uwb_cir_plotter.py --compare path/to/run/dir/  # compare explicit run directory
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_SAMPLES  = 1016
LOGDIR     = Path("data/uwb/logs")
RX_ROLES   = ("rx1", "rx2", "rx3")
RX_COLORS  = ("steelblue", "tomato", "seagreen")


# ── Load ───────────────────────────────────────────────────────────────────

def find_latest_log() -> Path:
    # Prefer resp.npz in run subdirs (data/uwb/logs/<run>/resp.npz)
    resp_logs = sorted(LOGDIR.glob("*/resp.npz"), key=lambda p: p.stat().st_mtime)
    # Fallback: legacy cir_*.npz in LOGDIR root
    legacy_logs = sorted(LOGDIR.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    logs = resp_logs or legacy_logs
    if not logs:
        print(f"No .npz files found in {LOGDIR}. Run capture_uwb.py first.")
        sys.exit(1)
    path = logs[-1]
    print(f"Auto-detected: {path}")
    return path


def find_latest_run() -> Path:
    runs = sorted(
        [p for p in LOGDIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        print(f"No run directories found in {LOGDIR}.")
        sys.exit(1)
    path = runs[-1]
    print(f"Auto-detected run: {path}")
    return path


def load_log(path: Path) -> dict:
    data = np.load(path)
    n = data["cir"].shape[0]
    cir = data["cir"]

    # ── Zero I/Q diagnostics ─────────────────────────
    zero_pairs = np.sum((cir.real == 0) & (cir.imag == 0), axis=1)

    print(f"Loaded {n} frames from {path}")
    print(f"Zero I/Q pairs per frame (min/mean/max): "
          f"{zero_pairs.min()} / {zero_pairs.mean():.1f} / {zero_pairs.max()} out of {N_SAMPLES}")

    for i in range(min(5, n)):
        print(f"  frame {i}: {zero_pairs[i]} zero taps")

    # cir_mag if available (new logs), else derive from cir (legacy)
    cir_mag = data["cir_mag"] if "cir_mag" in data else np.abs(cir)

    return {
        "cir":       cir,
        "cir_mag":   cir_mag,
        "seq":       data["seq"],
        "fp_index":  data["fp_index"],
        "rxpacc":    data["rxpacc"],
        "timestamp": data["timestamp"],
    }


# ── Step-through plotter ─────────────────────────────────────────────────────

def plot_step(log: dict):
    cir_mag_all = log["cir_mag"]
    seq_all     = log["seq"]
    fp_all     = log["fp_index"]
    rxpacc_all = log["rxpacc"]
    ts_all     = log["timestamp"]
    n = cir_mag_all.shape[0]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("CIR Log — Step Through (magnitude)", fontsize=13)

    sample_axis = np.arange(N_SAMPLES)
    mag = np.roll(cir_mag_all[0], -int(fp_all[0]))

    (line,)   = ax.plot(sample_axis, mag, lw=0.8, color="steelblue")
    fp_line   = ax.axvline(x=0, color="red",    lw=1.5, ls="--", label="First path (index 0)")
    peak_line = ax.axvline(x=0, color="orange", lw=1.5, ls=":",  label="Peak")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, N_SAMPLES - 1)
    ax.set_xlabel("Sample index relative to first path")
    ax.set_ylabel("Normalised magnitude")
    ax.grid(True, alpha=0.3)

    info_text = ax.text(
        0.01, 0.95, "", transform=ax.transAxes,
        fontsize=8, va="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    state = {"idx": 0}

    def draw_frame(i):
        fp  = int(fp_all[i])
        mag = np.roll(cir_mag_all[i], -fp)
        peak_idx = int(np.argmax(mag))
        fp_peak_ratio = mag[0] / (mag[peak_idx] + 1e-9)

        line.set_ydata(mag)
        ax.set_ylim(0, mag.max() * 1.15 + 1e-6)
        fp_line.set_xdata([0, 0])
        peak_line.set_xdata([peak_idx, peak_idx])

        info_text.set_text(
            f"frame {i}/{n-1}  seq={seq_all[i]}  t={ts_all[i]:.3f}s\n"
            f"fp={fp}  peak={peak_idx}  rxpacc={rxpacc_all[i]}\n"
            f"FP/peak={fp_peak_ratio:.3f}"
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("enter", "right", " ", "."):
            state["idx"] = min(state["idx"] + 1, n - 1)
        elif event.key in ("left", ","):
            state["idx"] = max(state["idx"] - 1, 0)
        elif event.key == "home":
            state["idx"] = 0
        elif event.key == "end":
            state["idx"] = n - 1
        elif event.key in ("q", "escape"):
            plt.close(fig)
            return
        else:
            return
        draw_frame(state["idx"])

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_frame(0)

    print(f"\n  Right/Enter/Space = next    Left = prev")
    print(f"  Home = first    End = last    q/Esc = quit\n")

    plt.tight_layout()
    plt.show()


# ── Waterfall plotter ────────────────────────────────────────────────────────

def plot_waterfall(log: dict):
    cir_mag_all = log["cir_mag"]
    fp_all      = log["fp_index"]
    ts_all = log["timestamp"]
    n = cir_mag_all.shape[0]

    mag_all = np.array([np.roll(cir_mag_all[i], -int(fp_all[i])) for i in range(n)])

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(f"CIR Waterfall — {n} frames", fontsize=13)

    img = ax.imshow(
        mag_all, aspect="auto", origin="upper",
        extent=[0, N_SAMPLES - 1, n, 0],
        cmap="inferno", interpolation="nearest",
    )
    fig.colorbar(img, ax=ax, label="Normalised magnitude")

    ax.axvline(x=0, color="cyan", lw=0.8, alpha=0.7, label="First path (index 0)")
    ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Sample index relative to first path")
    ax.set_ylabel("Frame")
    ax.grid(False)

    plt.tight_layout()
    plt.show()


# ── Multi-RX compare plotter ─────────────────────────────────────────────────

def plot_compare(run_dir: Path):
    """Overlay CIR magnitude from rx1/rx2/rx3 for the same frame index."""
    logs = {}
    for role in RX_ROLES:
        path = run_dir / f"{role}.npz"
        if not path.exists():
            print(f"  {role}.npz not found — skipping")
            continue
        data = np.load(path)
        logs[role] = {
            "cir_mag":  data["cir_mag"] if "cir_mag" in data else np.abs(data["cir"]),
            "fp_index": data["fp_index"],
            "seq":      data["seq"],
            "timestamp": data["timestamp"],
        }
        print(f"  {role}: {logs[role]['cir_mag'].shape[0]} frames")

    if not logs:
        print("No RX npz files found.")
        sys.exit(1)

    n_frames = min(v["cir_mag"].shape[0] for v in logs.values())
    sample_axis = np.arange(N_SAMPLES)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(f"CIR Compare — {', '.join(logs.keys())} — {run_dir.name}", fontsize=12)

    lines = {}
    fp_lines = {}
    for role, color in zip(RX_ROLES, RX_COLORS):
        if role not in logs:
            continue
        mag = np.roll(logs[role]["cir_mag"][0], -int(logs[role]["fp_index"][0]))
        (line,) = ax.plot(sample_axis, mag, lw=0.9, color=color, label=role, alpha=0.85)
        fp_line = ax.axvline(x=0, color=color, lw=1.2, ls="--", alpha=0.5)
        lines[role] = line
        fp_lines[role] = fp_line

    ax.set_xlim(0, N_SAMPLES - 1)
    ax.set_xlabel("Sample index relative to first path")
    ax.set_ylabel("Magnitude")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    info_text = ax.text(
        0.01, 0.95, "", transform=ax.transAxes,
        fontsize=8, va="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    state = {"idx": 0}

    def draw_frame(i):
        parts = [f"frame {i}/{n_frames - 1}"]
        y_max = 0.0
        for role in logs:
            fp  = int(logs[role]["fp_index"][i])
            mag = np.roll(logs[role]["cir_mag"][i], -fp)
            lines[role].set_ydata(mag)
            fp_lines[role].set_xdata([0, 0])
            y_max = max(y_max, mag.max())
            parts.append(f"{role}: fp={fp}  seq={logs[role]['seq'][i]}  "
                         f"t={logs[role]['timestamp'][i]:.3f}s")
        ax.set_ylim(0, y_max * 1.15 + 1e-6)
        info_text.set_text("\n".join(parts))
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("enter", "right", " ", "."):
            state["idx"] = min(state["idx"] + 1, n_frames - 1)
        elif event.key in ("left", ","):
            state["idx"] = max(state["idx"] - 1, 0)
        elif event.key == "home":
            state["idx"] = 0
        elif event.key == "end":
            state["idx"] = n_frames - 1
        elif event.key in ("q", "escape"):
            plt.close(fig)
            return
        else:
            return
        draw_frame(state["idx"])

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_frame(0)

    print(f"\n  Right/Enter/Space = next    Left = prev")
    print(f"  Home = first    End = last    q/Esc = quit\n")

    plt.tight_layout()
    plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot CIR data from a captured .npz log")
    parser.add_argument("path", nargs="?",
                        help="Path to .npz log or run directory (auto-detects latest if omitted)")
    parser.add_argument("--waterfall", action="store_true",
                        help="Show full waterfall instead of step-through")
    parser.add_argument("--compare", action="store_true",
                        help="Overlay rx1/rx2/rx3 from a run directory on one plot")
    args = parser.parse_args()

    if args.compare:
        if args.path:
            run_dir = Path(args.path)
            if not run_dir.is_dir():
                print(f"Error: {run_dir} is not a directory.")
                sys.exit(1)
        else:
            run_dir = find_latest_run()
        plot_compare(run_dir)
        return

    if args.path:
        path = Path(args.path)
        if not path.exists():
            print(f"Error: {path} does not exist.")
            sys.exit(1)
    else:
        path = find_latest_log()

    log = load_log(path)

    if args.waterfall:
        plot_waterfall(log)
    else:
        plot_step(log)


if __name__ == "__main__":
    main()
