"""
Offline CIR (Channel Impulse Response) plotter for captured .npz logs.

Modes:
  step      (default)  Step through frames one at a time.
  waterfall            Full waterfall heatmap of the entire log.

Usage:
  python uwb_cir_plotter.py                          # auto-detect latest log, step mode
  python uwb_cir_plotter.py path/to/cir.npz          # explicit log, step mode
  python uwb_cir_plotter.py --waterfall               # waterfall of latest log
  python uwb_cir_plotter.py path/to/cir.npz --waterfall
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_SAMPLES = 1016
LOGDIR = Path("data/uwb/logs")


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


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot CIR data from a captured .npz log")
    parser.add_argument("npz", nargs="?", help="Path to .npz log (auto-detects latest if omitted)")
    parser.add_argument("--waterfall", action="store_true", help="Show full waterfall instead of step-through")
    args = parser.parse_args()

    if args.npz:
        path = Path(args.npz)
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
