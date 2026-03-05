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
    print(f"Loaded {n} frames from {path}")
    return {
        "cir":       data["cir"],
        "seq":       data["seq"],
        "fp_index":  data["fp_index"],
        "rxpacc":    data["rxpacc"],
        "timestamp": data["timestamp"],
    }

def plot_debug(log: dict, frame: int = 0):
    cir_all = log["cir"]
    mag = np.abs(cir_all[frame])
    sample_axis = np.arange(N_SAMPLES)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sample_axis, mag, lw=0.8, color="steelblue")
    ax.set_xlim(0, N_SAMPLES - 1)
    ax.set_xlabel("Tap index (0..1015)")
    ax.set_ylabel("Raw magnitude")
    ax.set_title(f"CIR magnitude — frame {frame}/{cir_all.shape[0] - 1}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot raw CIR magnitude vs tap index")
    parser.add_argument("npz", nargs="?", help="Path to .npz log (auto-detects latest if omitted)")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to plot (default 0)")
    args = parser.parse_args()
    if args.npz:
        path = Path(args.npz)
        if not path.exists():
            print(f"Error: {path} does not exist.")
            sys.exit(1)
    else:
        path = find_latest_log()
    log = load_log(path)
    plot_debug(log, frame=args.frame)

if __name__ == "__main__":
    main()