"""
Offline CIR (Channel Impulse Response) plotter for captured .npz logs.

Modes:
  step      (default)  Step through frames one at a time (all RX subplots).
  waterfall            Full waterfall heatmap (all RX subplots).

If given a run directory (or none — auto-detects latest run), all available
rx1/rx2/rx3 channels are loaded and shown side-by-side.  If given a single
.npz file, only that channel is shown.

Usage:
  python uwb_cir_plotter.py                              # latest run, step mode
  python uwb_cir_plotter.py path/to/run_dir/             # explicit run dir
  python uwb_cir_plotter.py path/to/rx1.npz              # single file
  python uwb_cir_plotter.py --waterfall                  # latest run, waterfall
  python uwb_cir_plotter.py path/to/run_dir/ --waterfall
  python uwb_cir_plotter.py --compare path/to/run_dir/   # (legacy alias for step)
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
          f"{zero_pairs.min()} / {zero_pairs.mean():.1f} / {zero_pairs.max()} "
          f"out of {N_SAMPLES}")

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


def load_run(run_dir: Path) -> "dict[str, dict]":
    """Load all available RX logs from a run directory."""
    logs = {}
    for role in RX_ROLES:
        path = run_dir / f"{role}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        logs[role] = {
            "cir_mag":   data["cir_mag"] if "cir_mag" in data else np.abs(data["cir"]),
            "fp_index":  data["fp_index"],
            "seq":       data["seq"],
            "rxpacc":    data["rxpacc"] if "rxpacc" in data else np.zeros(data["cir"].shape[0], dtype=np.uint16),
            "timestamp": data["timestamp"],
        }
        print(f"  {role}: {logs[role]['cir_mag'].shape[0]} frames")
    return logs


# ── Step-through plotter (multi-RX) ──────────────────────────────────────────

def plot_step(logs: "dict[str, dict]", title: str = ""):
    """Step through frames with one subplot per RX channel."""
    roles = [r for r in RX_ROLES if r in logs]
    n_rx = len(roles)
    n_frames = min(logs[r]["cir_mag"].shape[0] for r in roles)
    sample_axis = np.arange(N_SAMPLES)

    fig, axes = plt.subplots(n_rx, 1, figsize=(14, 4 * n_rx), sharex=True)
    if n_rx == 1:
        axes = [axes]
    fig.suptitle(title or "CIR Step Through", fontsize=13)

    lines = {}
    fp_lines = {}
    peak_lines = {}
    info_texts = {}

    for ax, role, color in zip(axes, roles, RX_COLORS):
        mag = logs[role]["cir_mag"][0]
        fp = int(logs[role]["fp_index"][0])
        peak = int(np.argmax(mag))

        (line,) = ax.plot(sample_axis, mag, lw=0.8, color=color, label=role)
        fp_line = ax.axvline(x=fp, color=color, lw=1.2, ls="--", alpha=0.6,
                             label="First path")
        peak_line = ax.axvline(x=peak, color="orange", lw=1.2, ls=":",
                               alpha=0.6, label="Peak")
        ax.set_ylabel(role, fontsize=11, fontweight="bold")
        ax.set_xlim(0, N_SAMPLES - 1)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)

        info = ax.text(
            0.99, 0.95, "", transform=ax.transAxes,
            fontsize=8, va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        lines[role] = line
        fp_lines[role] = fp_line
        peak_lines[role] = peak_line
        info_texts[role] = info

    axes[-1].set_xlabel("Sample index")

    state = {"idx": 0}

    def draw_frame(i):
        for ax, role in zip(axes, roles):
            log = logs[role]
            mag = log["cir_mag"][i]
            fp = int(log["fp_index"][i])
            peak = int(np.argmax(mag))
            fp_peak_ratio = mag[fp] / (mag[peak] + 1e-9)

            lines[role].set_ydata(mag)
            ax.set_ylim(0, float(mag.max()) * 1.15 + 1e-6)
            fp_lines[role].set_xdata([fp, fp])
            peak_lines[role].set_xdata([peak, peak])
            info_texts[role].set_text(
                f"frame {i}/{n_frames-1}  seq={log['seq'][i]}  "
                f"t={log['timestamp'][i]:.3f}s\n"
                f"fp={fp}  peak={peak}  rxpacc={log['rxpacc'][i]}  "
                f"FP/peak={fp_peak_ratio:.3f}"
            )
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


# ── Waterfall plotter (multi-RX) ────────────────────────────────────────────

def plot_waterfall(logs: "dict[str, dict]", title: str = ""):
    """Waterfall heatmap with one subplot per RX channel."""
    roles = [r for r in RX_ROLES if r in logs]
    n_rx = len(roles)

    fig, axes = plt.subplots(n_rx, 1, figsize=(14, 5 * n_rx), sharex=True)
    if n_rx == 1:
        axes = [axes]
    fig.suptitle(title or "CIR Waterfall", fontsize=13)

    for ax, role, color in zip(axes, roles, RX_COLORS):
        log = logs[role]
        cir_mag = log["cir_mag"]
        fp_all = log["fp_index"]
        n = cir_mag.shape[0]

        fp_line = ax.axvline(x=0, color="cyan", lw=0.8, ls="--", alpha=0.0)

        img = ax.imshow(
            cir_mag, aspect="auto", origin="upper",
            extent=[0, N_SAMPLES - 1, n, 0],
            cmap="inferno", interpolation="nearest",
        )
        fig.colorbar(img, ax=ax, label="Magnitude", shrink=0.8)

        # Overlay FP trace as a scatter/line
        ax.plot(fp_all, np.arange(n), color="cyan", lw=0.6, alpha=0.7,
                label="First path")

        ax.set_ylabel(f"{role}\nFrame", fontsize=10, fontweight="bold")
        ax.set_xlim(0, N_SAMPLES - 1)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(False)

        ax.text(0.99, 0.02, f"{n} frames", transform=ax.transAxes,
                fontsize=8, ha="right", va="bottom", color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))

    axes[-1].set_xlabel("Sample index")

    plt.tight_layout()
    plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot CIR data from captured .npz logs")
    parser.add_argument("path", nargs="?",
                        help="Run directory or single .npz file "
                             "(auto-detects latest run if omitted)")
    parser.add_argument("--waterfall", action="store_true",
                        help="Show waterfall heatmap instead of step-through")
    parser.add_argument("--compare", action="store_true",
                        help="(Legacy) same as default step-through on a run dir")
    args = parser.parse_args()

    # ── Resolve path → either a run dir with multiple RX, or a single file ──
    if args.path:
        path = Path(args.path)
        if not path.exists():
            print(f"Error: {path} does not exist.")
            sys.exit(1)
        if path.is_dir():
            run_dir = path
        else:
            run_dir = None  # single file
    else:
        run_dir = find_latest_run()
        path = None

    # ── Load ──
    if run_dir is not None:
        print(f"Loading from {run_dir}:")
        logs = load_run(run_dir)
        if not logs:
            print("No RX .npz files found in directory.")
            sys.exit(1)
        title_suffix = f" — {run_dir.name}"
    else:
        # Single .npz file
        log = load_log(path)
        role = path.stem if path.stem in RX_ROLES else "rx"
        logs = {role: log}
        title_suffix = f" — {path.name}"

    # ── Plot ──
    if args.waterfall:
        plot_waterfall(logs, title=f"CIR Waterfall{title_suffix}")
    else:
        plot_step(logs, title=f"CIR Step Through{title_suffix}")


if __name__ == "__main__":
    main()
