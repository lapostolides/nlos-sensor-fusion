"""
Real-time CIR (Channel Impulse Response) reader and plotter for DWM1001-Dev.

Expected binary frame format from custom CIR firmware:
  [0xBC][0xAD]       2B  sync
  [seq]              2B  uint16 LE frame counter
  [fp_index]         2B  uint16 LE first path index in CIR
  [rxpacc]           2B  uint16 LE preamble accumulation count
  [cir]           4064B  1016 x (I: int16 LE, Q: int16 LE)
  [xor]              1B  XOR checksum of bytes[2:-1]
  Total: 4073 bytes/frame

Usage:
  python uwb_cir_plotter.py
  python uwb_cir_plotter.py --port COM5
  python uwb_cir_plotter.py --port COM5 --waterfall 100
"""

import argparse
import queue
import struct
import threading
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import serial
import serial.tools.list_ports

# ── Frame format constants ──────────────────────────────────────────────────
SYNC         = bytes([0xBC, 0xAD])
HEADER_FMT   = "<HHH"          # seq, fp_index, rxpacc  (3 × uint16)
HEADER_SIZE  = struct.calcsize(HEADER_FMT)   # 6 bytes
N_SAMPLES    = 1016
CIR_SIZE     = N_SAMPLES * 4   # 1016 × (int16 I + int16 Q)
FRAME_SIZE   = 2 + HEADER_SIZE + CIR_SIZE + 1  # sync + header + cir + xor = 4073

# ── Globals shared between reader thread and plot ───────────────────────────
frame_queue: queue.Queue = queue.Queue(maxsize=8)
stop_event = threading.Event()


# ───────────────────────────────────────────────────────────────────────────
# Serial / frame parsing
# ───────────────────────────────────────────────────────────────────────────

def find_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").upper()
        if any(k in desc for k in ("JLINK", "J-LINK", "DWM")):
            print(f"[uwb] Auto-detected: {p.device} ({p.description})")
            return p.device
    if ports:
        print("[uwb] Available ports:")
        for i, p in enumerate(ports):
            print(f"  [{i}] {p.device} — {p.description}")
        return ports[int(input("Select index: "))].device
    raise RuntimeError("No serial ports found.")


def xor_checksum(data: bytes) -> int:
    result = 0
    for b in data:
        result ^= b
    return result


def parse_frame(raw: bytes):
    """
    Parse a complete raw frame (FRAME_SIZE bytes starting with SYNC).
    Returns (seq, fp_index, rxpacc, cir_complex) or None on checksum error.
    cir_complex: np.ndarray of shape (N_SAMPLES,), dtype complex64
    """
    # Checksum covers bytes[2:-1] (everything between sync and checksum byte)
    payload = raw[2:-1]
    expected_xor = xor_checksum(payload)
    actual_xor   = raw[-1]
    if expected_xor != actual_xor:
        return None

    header_bytes = payload[:HEADER_SIZE]
    cir_bytes    = payload[HEADER_SIZE:]

    seq, fp_index_raw, rxpacc = struct.unpack(HEADER_FMT, header_bytes)
    fp_index = min(fp_index_raw >> 6, N_SAMPLES - 1)  # 10.6 fixed-point, clamped

    # Unpack 1016 pairs of int16 (I, Q)
    iq = np.frombuffer(cir_bytes, dtype="<i2").reshape(N_SAMPLES, 2).astype(np.float32)
    cir = iq[:, 0] + 1j * iq[:, 1]

    # Normalize by RXPACC (preamble accumulation count)
    if rxpacc > 0:
        cir /= rxpacc

    return seq, fp_index, rxpacc, cir


def reader_thread(port: str, baud: int):
    """
    Background thread: reads serial bytes, syncs to frame boundaries,
    parses frames, pushes to frame_queue.
    """
    buf = bytearray()
    stats = {
        "frames": 0, "errors": 0, "bytes": 0,
        "syncs": 0, "t0": time.monotonic(), "last_diag": time.monotonic(),
    }

    try:
        ser = serial.Serial(port, baud, timeout=0.05)
        print(f"[uwb] Opened {port} at {baud} baud")
    except Exception as e:
        print(f"[uwb] Failed to open port: {e}")
        stop_event.set()
        return

    with ser:
        while not stop_event.is_set():
            chunk = ser.read(512)
            if not chunk:
                now = time.monotonic()
                if now - stats["last_diag"] >= 3.0:
                    elapsed = now - stats["t0"]
                    print(f"[uwb] {elapsed:.0f}s | {stats['bytes']} bytes rx | "
                          f"{stats['syncs']} syncs | {stats['frames']} frames | "
                          f"{stats['errors']} checksum errors")
                    if stats["bytes"] == 0:
                        print("[uwb]   ** No bytes received — check that both boards are "
                              "powered and this port is the RESPONDER board **")
                    elif stats["syncs"] == 0:
                        print("[uwb]   ** Bytes received but no sync (0xBC 0xAD) found — "
                              "possible baud rate mismatch **")
                    elif stats["frames"] == 0:
                        print("[uwb]   ** Syncs found but all checksums fail — "
                              "possible frame format mismatch **")
                    stats["last_diag"] = now
                continue
            stats["bytes"] += len(chunk)

            # Detect ASCII diagnostic messages from firmware (e.g. "CIR:BOOT", "CIR:DW_FAIL")
            if b"CIR:" in chunk:
                try:
                    text = chunk.decode("ascii", errors="replace").strip()
                    for line in text.splitlines():
                        if "CIR:" in line:
                            print(f"[uwb] FIRMWARE: {line.strip()}")
                except Exception:
                    pass

            buf.extend(chunk)

            # Slide through buffer looking for sync bytes
            while len(buf) >= FRAME_SIZE:
                idx = buf.find(SYNC)
                if idx == -1:
                    buf = buf[-1:]
                    break
                if idx > 0:
                    buf = buf[idx:]
                if len(buf) < FRAME_SIZE:
                    break

                stats["syncs"] += 1
                raw = bytes(buf[:FRAME_SIZE])
                result = parse_frame(raw)

                if result is not None:
                    stats["frames"] += 1
                    if stats["frames"] == 1:
                        print(f"[uwb] First valid frame received after "
                              f"{time.monotonic() - stats['t0']:.1f}s")
                    try:
                        frame_queue.put_nowait(result)
                    except queue.Full:
                        frame_queue.get_nowait()
                        frame_queue.put_nowait(result)
                    buf = buf[FRAME_SIZE:]
                else:
                    stats["errors"] += 1
                    if stats["errors"] <= 5:
                        print(f"[uwb] Checksum mismatch #{stats['errors']} "
                              f"(total syncs: {stats['syncs']})")
                    buf = buf[1:]

                if stats["frames"] % 100 == 0 and stats["frames"] > 0:
                    elapsed = time.monotonic() - stats["t0"]
                    fps = stats["frames"] / elapsed
                    err_rate = stats["errors"] / max(stats["frames"] + stats["errors"], 1)
                    print(f"[uwb] {stats['frames']} frames | {fps:.1f} fps | "
                          f"error rate {err_rate:.1%}")


# ───────────────────────────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────────────────────────

def make_plot(n_waterfall: int):
    fig, (ax_cir, ax_wf) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.suptitle("DWM1001 CIR — Live", fontsize=13)

    sample_axis = np.arange(N_SAMPLES)

    # ── Top: CIR magnitude ──────────────────────────────────────────────────
    ax_cir.set_xlim(0, N_SAMPLES - 1)
    ax_cir.set_xlabel("Sample index")
    ax_cir.set_ylabel("Normalised magnitude")
    ax_cir.set_title("CIR magnitude (current frame)")
    ax_cir.grid(True, alpha=0.3)

    (line_cir,) = ax_cir.plot(sample_axis, np.zeros(N_SAMPLES), lw=0.8, color="steelblue")
    fp_line     = ax_cir.axvline(x=0, color="red",    lw=1.5, ls="--", label="First path")
    peak_line   = ax_cir.axvline(x=0, color="orange", lw=1.5, ls=":",  label="Peak")
    ax_cir.legend(loc="upper right", fontsize=8)

    info_text = ax_cir.text(
        0.01, 0.95, "", transform=ax_cir.transAxes,
        fontsize=8, va="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    # ── Bottom: Waterfall ───────────────────────────────────────────────────
    waterfall = np.zeros((n_waterfall, N_SAMPLES), dtype=np.float32)
    img = ax_wf.imshow(
        waterfall, aspect="auto", origin="upper",
        extent=[0, N_SAMPLES - 1, n_waterfall, 0],
        cmap="inferno", interpolation="nearest"
    )
    fig.colorbar(img, ax=ax_wf, label="Normalised magnitude")
    ax_wf.set_xlabel("Sample index")
    ax_wf.set_ylabel("Frame (newest at top)")
    ax_wf.set_title("Waterfall (recent frames)")

    plt.tight_layout()

    # ── State shared into animation callback ────────────────────────────────
    state = {
        "waterfall": waterfall,
        "frame_count": 0,
        "t_start": time.monotonic(),
        "last_fp": 0,
    }

    def update(_):
        # Drain all pending frames, keep only the latest
        latest = None
        while True:
            try:
                latest = frame_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            return line_cir, fp_line, peak_line, img, info_text

        seq, fp_index, rxpacc, cir = latest
        mag = np.abs(cir)

        # Update CIR line
        line_cir.set_ydata(mag)
        ax_cir.set_ylim(0, mag.max() * 1.15 + 1e-6)

        # Marker lines
        peak_idx = int(np.argmax(mag))
        fp_line.set_xdata([fp_index, fp_index])
        peak_line.set_xdata([peak_idx, peak_idx])

        # Waterfall: roll up, insert new row at top
        state["waterfall"] = np.roll(state["waterfall"], 1, axis=0)
        state["waterfall"][0] = mag
        img.set_data(state["waterfall"])
        img.set_clim(0, state["waterfall"].max() + 1e-6)

        # Info text
        state["frame_count"] += 1
        elapsed = time.monotonic() - state["t_start"]
        fps = state["frame_count"] / elapsed
        fp_peak_ratio = mag[fp_index] / (mag[peak_idx] + 1e-9)
        info_text.set_text(
            f"seq={seq}  fp={fp_index}  peak={peak_idx}  rxpacc={rxpacc}\n"
            f"FP/peak={fp_peak_ratio:.3f}  fps={fps:.1f}"
        )

        return line_cir, fp_line, peak_line, img, info_text

    return fig, update


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DWM1001 CIR live plotter")
    parser.add_argument("--port",      help="Serial port (auto-detected if omitted)")
    parser.add_argument("--baud",      type=int, default=115200, help="Baud rate (default 115200)")
    parser.add_argument("--waterfall", type=int, default=60,     help="Waterfall history rows (default 60)")
    args = parser.parse_args()

    port = args.port or find_port()

    # Start reader thread
    t = threading.Thread(target=reader_thread, args=(port, args.baud), daemon=True)
    t.start()

    # Build and run plot
    fig, update_fn = make_plot(args.waterfall)
    ani = animation.FuncAnimation(
        fig, update_fn,
        interval=50,   # ms between animation frames (~20 Hz refresh)
        blit=True,
        cache_frame_data=False,
    )

    try:
        plt.show()
    finally:
        stop_event.set()
        t.join(timeout=2)


if __name__ == "__main__":
    main()
