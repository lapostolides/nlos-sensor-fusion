#!/usr/bin/env python3
"""
test_uwb_rx_diff.py - Verify that rx1, rx2, rx3 CIR data are distinct.

Usage:
    python test_uwb_rx_diff.py                        # auto-detects latest run
    python test_uwb_rx_diff.py data/uwb/logs/my-run/
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

LOGDIR = Path("data/uwb/logs")
ROLES  = ("rx1", "rx2", "rx3")


def find_latest_run() -> Path:
    runs = sorted(
        [p for p in LOGDIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        print(f"No run directories found in {LOGDIR}.")
        sys.exit(1)
    return runs[-1]


def load_cir(run_dir: Path, role: str) -> "np.ndarray | None":
    path = run_dir / f"{role}.npz"
    if not path.exists():
        return None
    return np.load(path)["cir_mag"]  # (n_frames, 1016) float32


def check_pair(a_mag: np.ndarray, b_mag: np.ndarray, label: str):
    """Compare two CIR magnitude arrays (use min frame count)."""
    n = min(len(a_mag), len(b_mag))
    a, b = a_mag[:n], b_mag[:n]

    identical   = np.array_equal(a, b)
    mean_absdif = float(np.mean(np.abs(a - b)))
    # Pearson correlation per frame, then average
    corrs = []
    for i in range(n):
        ai, bi = a[i], b[i]
        if ai.std() > 0 and bi.std() > 0:
            corrs.append(float(np.corrcoef(ai, bi)[0, 1]))
    mean_corr = float(np.mean(corrs)) if corrs else float("nan")

    status = "IDENTICAL (FAIL)" if identical else "DIFFERENT (OK)"
    print(f"  {label}: {status}")
    print(f"    frames compared : {n}")
    print(f"    mean |delta|    : {mean_absdif:.4f}")
    print(f"    mean correlation: {mean_corr:.4f}  (1.0 = identical, 0.0 = uncorrelated)")
    return not identical


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_run()
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist.")
        sys.exit(1)

    print(f"\nRun: {run_dir}\n")

    logs = {}
    for role in ROLES:
        mag = load_cir(run_dir, role)
        if mag is None:
            print(f"  {role}.npz not found — skipping")
        else:
            print(f"  {role}: {len(mag)} frames, shape {mag.shape}")
            logs[role] = mag

    if len(logs) < 2:
        print("\nNeed at least 2 RX files to compare.")
        sys.exit(1)

    print()
    all_ok = True
    for r_a, r_b in combinations(logs.keys(), 2):
        ok = check_pair(logs[r_a], logs[r_b], f"{r_a} vs {r_b}")
        all_ok = all_ok and ok
        print()

    print("-" * 40)
    if all_ok:
        print("PASS — all RX channels have distinct CIR data.")
    else:
        print("FAIL — one or more RX channels have identical data.")
    print("-" * 40)


if __name__ == "__main__":
    main()
