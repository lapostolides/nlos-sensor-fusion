#!/usr/bin/env python3
"""
Live UWB CIR plotter — connects to 1 TX + up to 3 RX DWM1001-DEV boards
and displays real-time CIR magnitude for each RX channel.

Optionally saves all received frames to .npz logs (same format as
capture_uwb.py) for later reference.

Usage:
  python plot_uwb_live.py                       # live view, no save
  python plot_uwb_live.py --name my-test         # live view + save to data/uwb/logs/my-test/
  python plot_uwb_live.py --name my-test --baud 115200
"""

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from capture_uwb import (
    INIT_FRAME,
    N_SAMPLES,
    RESP_FRAME,
    SYNC,
    arm_board,
    discover_roles,
    parse_init_frame,
    parse_resp_frame,
)

LOGDIR = Path("data/uwb/logs")

RX_ROLES = ("rx1", "rx2", "rx3")
RX_COLORS = ("steelblue", "tomato", "seagreen")

UPDATE_INTERVAL_MS = 50   # matplotlib timer interval
WATERFALL_ROWS     = 200  # rolling window depth for waterfall


# ── Thread-safe single-slot buffer ───────────────────────────────────────────

class LatestCIR:
    """Single-slot buffer for live display + optional frame accumulator for saving."""

    def __init__(self, save: bool = False):
        self._lock = threading.Lock()
        self._mag = None
        self._fp = 0
        self._seq = 0
        self._count = 0
        # Accumulator (only when saving)
        self._save = save
        self._cirs: list[np.ndarray] = []
        self._seqs: list[int] = []
        self._fps: list[int] = []
        self._rxpaccs: list[int] = []
        self._rx_tss: list[np.uint64] = []
        self._timestamps: list[float] = []

    def put(self, mag, fp, seq, cir=None, rxpacc=0, rx_ts=np.uint64(0),
            timestamp=0.0):
        with self._lock:
            self._mag = mag
            self._fp = fp
            self._seq = seq
            self._count += 1
            if self._save and cir is not None:
                self._cirs.append(cir)
                self._seqs.append(seq)
                self._fps.append(fp)
                self._rxpaccs.append(rxpacc)
                self._rx_tss.append(rx_ts)
                self._timestamps.append(timestamp)

    def get(self):
        with self._lock:
            return self._mag, self._fp, self._seq, self._count

    def save_npz(self, path: Path):
        """Write accumulated frames to .npz (same format as capture_uwb.py)."""
        with self._lock:
            n = len(self._cirs)
            if n == 0:
                print(f"  {path.name}: no frames to save")
                return 0
            cir_arr = np.array(self._cirs, dtype=np.complex64)
            ts_arr = np.array(self._timestamps, dtype=np.float64)
            wall_start = np.float64(
                time.time() - (time.monotonic() - self._timestamps[0]))
            ts_arr -= ts_arr[0]

        cir_i = cir_arr.real.astype(np.float32)
        cir_q = cir_arr.imag.astype(np.float32)
        cir_mag = np.abs(cir_arr).astype(np.float32)
        cir_phase = np.angle(cir_arr).astype(np.float32)

        np.savez_compressed(
            path,
            cir=cir_arr, cir_i=cir_i, cir_q=cir_q,
            cir_mag=cir_mag, cir_phase=cir_phase,
            seq=np.array(self._seqs, dtype=np.uint16),
            fp_index=np.array(self._fps, dtype=np.uint16),
            rxpacc=np.array(self._rxpaccs, dtype=np.uint16),
            rx_ts=np.array(self._rx_tss, dtype=np.uint64),
            timestamp=ts_arr,
            timestamp_wall_start=wall_start,
        )
        print(f"  {path.name}: {n} frames saved")
        return n


# ── RX streaming thread ─────────────────────────────────────────────────────

def rx_stream(tag, port, baud, buf: LatestCIR, stop: threading.Event):
    import serial

    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        print(f"[{tag}] Opened {port}")
    except Exception as e:
        print(f"[{tag}] Failed to open {port}: {e}")
        return

    if not arm_board(ser, tag):
        ser.close()
        return

    raw_buf = bytearray()
    frames = 0
    t0 = time.monotonic()

    while not stop.is_set():
        chunk = ser.read(512)
        if not chunk:
            continue
        raw_buf.extend(chunk)

        while len(raw_buf) >= RESP_FRAME:
            idx = raw_buf.find(SYNC)
            if idx == -1:
                raw_buf = raw_buf[-1:]
                break
            if idx > 0:
                raw_buf = raw_buf[idx:]
            if len(raw_buf) < RESP_FRAME:
                break

            raw = bytes(raw_buf[:RESP_FRAME])
            result = parse_resp_frame(raw)
            if result is not None:
                seq, fp_index, rxpacc, rx_ts, cir = result
                t_now = time.monotonic()
                mag = np.abs(cir).astype(np.float32)
                buf.put(mag, fp_index, seq,
                        cir=cir, rxpacc=rxpacc, rx_ts=rx_ts,
                        timestamp=t_now)
                frames += 1
                raw_buf = raw_buf[RESP_FRAME:]
                if frames == 1:
                    print(f"[{tag}] First frame after {time.monotonic() - t0:.1f}s")
                if frames % 200 == 0:
                    fps = frames / (time.monotonic() - t0)
                    print(f"[{tag}] {frames} frames, {fps:.1f} fps")
            else:
                raw_buf = raw_buf[1:]

    try:
        ser.write(b"STOP\n")
    except Exception:
        pass
    ser.close()
    elapsed = time.monotonic() - t0
    print(f"[{tag}] Done — {frames} frames in {elapsed:.1f}s")


# ── TX keepalive thread ──────────────────────────────────────────────────────

def tx_stream(tag, port, baud, stop: threading.Event):
    """Arm the initiator and drain its frames so it keeps transmitting."""
    import serial

    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        print(f"[{tag}] Opened {port}")
    except Exception as e:
        print(f"[{tag}] Failed to open {port}: {e}")
        return

    if not arm_board(ser, tag):
        ser.close()
        return

    raw_buf = bytearray()
    frames = 0
    t0 = time.monotonic()

    while not stop.is_set():
        chunk = ser.read(512)
        if not chunk:
            continue
        raw_buf.extend(chunk)

        # Drain TX timestamp frames to keep the buffer from backing up
        while len(raw_buf) >= INIT_FRAME:
            idx = raw_buf.find(SYNC)
            if idx == -1:
                raw_buf = raw_buf[-1:]
                break
            if idx > 0:
                raw_buf = raw_buf[idx:]
            if len(raw_buf) < INIT_FRAME:
                break

            raw = bytes(raw_buf[:INIT_FRAME])
            result = parse_init_frame(raw)
            if result is not None:
                frames += 1
                raw_buf = raw_buf[INIT_FRAME:]
                if frames == 1:
                    print(f"[{tag}] TX active — first pulse after "
                          f"{time.monotonic() - t0:.1f}s")
            else:
                raw_buf = raw_buf[1:]

    try:
        ser.write(b"STOP\n")
    except Exception:
        pass
    ser.close()
    elapsed = time.monotonic() - t0
    print(f"[{tag}] Done — {frames} TX pulses in {elapsed:.1f}s")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live UWB CIR plotter (3 RX channels)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--name",
                        help="Run name — saves .npz logs to data/uwb/logs/<name>/. "
                             "Omit to view without saving.")
    args = parser.parse_args()

    saving = args.name is not None
    if saving:
        run_dir = LOGDIR / args.name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving to: {run_dir}")
    else:
        run_dir = None
        print("View-only mode (use --name to save)")

    # Discover boards
    role_to_port = discover_roles(args.baud)
    if not role_to_port:
        print("No UWB boards found.")
        return

    rx_ports = {r: p for r, p in role_to_port.items() if r.startswith("rx")}
    tx_port = role_to_port.get("tx")

    if not rx_ports:
        print("No RX boards found — nothing to plot.")
        return
    if not tx_port:
        print("WARNING: No TX board found — RX boards won't receive any data!")

    active = sorted(rx_ports)
    if tx_port:
        active = ["tx"] + active
    print(f"\nActive boards: {', '.join(active)}")

    # Buffers + threads
    stop = threading.Event()
    bufs: dict[str, LatestCIR] = {}
    threads = []

    # Start TX first so it's already pulsing when RX boards arm
    if tx_port:
        tx_thread = threading.Thread(
            target=tx_stream,
            args=("tx", tx_port, args.baud, stop),
            daemon=True,
        )
        tx_thread.start()
        threads.append(tx_thread)

    for role, port in sorted(rx_ports.items()):
        buf = LatestCIR(save=saving)
        bufs[role] = buf
        t = threading.Thread(
            target=rx_stream,
            args=(role, port, args.baud, buf, stop),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Set up matplotlib figure — line plots (left) + waterfalls (right)
    n_rx = len(rx_ports)
    sorted_rx = sorted(rx_ports)
    fig, all_axes = plt.subplots(
        n_rx, 2, figsize=(16, 3 * n_rx),
        gridspec_kw={"width_ratios": [3, 2]},
    )
    if n_rx == 1:
        all_axes = all_axes.reshape(1, 2)
    fig.suptitle("Live UWB CIR  (close window to stop)", fontsize=12)

    sample_axis = np.arange(N_SAMPLES)
    lines = {}
    fp_lines = {}
    info_texts = {}
    wf_imgs = {}
    wf_data = {}
    wf_rows = {}       # how many rows written so far
    wf_fp_lines = {}
    prev_counts = {}

    for i, (role, color) in enumerate(zip(sorted_rx, RX_COLORS)):
        # ── Left: line plot ──
        ax_line = all_axes[i, 0]
        (line,) = ax_line.plot(sample_axis, np.zeros(N_SAMPLES), lw=0.8,
                               color=color)
        fp_line = ax_line.axvline(x=0, color=color, lw=1.2, ls="--", alpha=0.5)
        ax_line.set_ylabel(role, fontsize=11, fontweight="bold")
        ax_line.set_xlim(0, N_SAMPLES - 1)
        ax_line.grid(True, alpha=0.3)
        info = ax_line.text(0.99, 0.95, "", transform=ax_line.transAxes,
                            fontsize=8, va="top", ha="right", family="monospace",
                            bbox=dict(boxstyle="round", facecolor="wheat",
                                      alpha=0.5))
        lines[role] = line
        fp_lines[role] = fp_line
        info_texts[role] = info
        prev_counts[role] = 0

        # ── Right: waterfall ──
        ax_wf = all_axes[i, 1]
        data = np.zeros((WATERFALL_ROWS, N_SAMPLES), dtype=np.float32)
        img = ax_wf.imshow(
            data, aspect="auto", origin="upper",
            extent=[0, N_SAMPLES - 1, WATERFALL_ROWS, 0],
            cmap="inferno", interpolation="nearest",
            vmin=0, vmax=1,
        )
        wf_fp = ax_wf.axvline(x=0, color="cyan", lw=0.8, ls="--", alpha=0.6)
        ax_wf.set_ylabel(role, fontsize=9)
        ax_wf.set_xlim(0, N_SAMPLES - 1)
        ax_wf.grid(False)
        wf_imgs[role] = img
        wf_data[role] = data
        wf_rows[role] = 0
        wf_fp_lines[role] = wf_fp

    all_axes[-1, 0].set_xlabel("Sample index")
    all_axes[-1, 1].set_xlabel("Sample index")
    fig.tight_layout()

    # Timer-driven update
    def update(_frame=None):
        for i, role in enumerate(sorted_rx):
            mag, fp, seq, count = bufs[role].get()
            if mag is None or count == prev_counts[role]:
                continue
            prev_counts[role] = count
            fp = int(fp)
            peak = int(np.argmax(mag))

            # Line plot
            lines[role].set_ydata(mag)
            all_axes[i, 0].set_ylim(0, float(mag.max()) * 1.15 + 1e-6)
            fp_lines[role].set_xdata([fp, fp])
            info_texts[role].set_text(f"seq={seq}  fp={fp}  peak={peak}")

            # Waterfall — scroll up, newest row at bottom
            wf = wf_data[role]
            wf[:-1] = wf[1:]
            wf[-1] = mag
            wf_rows[role] += 1
            wf_imgs[role].set_data(wf)
            wf_imgs[role].set_clim(vmin=0, vmax=float(wf.max()) + 1e-6)
            wf_fp_lines[role].set_xdata([fp, fp])

        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=UPDATE_INTERVAL_MS)
    timer.add_callback(update)
    timer.start()

    print("\n=== Live CIR — close the plot window to stop ===\n")

    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    stop.set()
    for t in threads:
        t.join(timeout=3)

    # Save accumulated data
    if saving:
        print(f"\nSaving to {run_dir}:")
        total = 0
        for role in sorted(bufs):
            n = bufs[role].save_npz(run_dir / f"{role}.npz")
            total += n
        if total > 0:
            print(f"Done — {total} total frames across {len(bufs)} channel(s).")
            print(f"Replay with: python uwb_cir_plotter.py --compare {run_dir}")
        else:
            print("No frames captured.")
    else:
        print("Done (no data saved).")


if __name__ == "__main__":
    main()
