#!/usr/bin/env python3
"""
capture_uwb.py - Capture UWB CIR frames to disk.

Reads CIR binary frames from a DWM1001-Dev over UART and saves them as a
single .npz file.  Press Ctrl+C to stop and flush to disk.

Output arrays in the .npz:
  cir        (N, 1016) complex64  — normalised complex CIR samples
  seq        (N,)      uint16     — firmware frame counter
  fp_index   (N,)      uint16     — first-path index (after 10.6 conversion)
  rxpacc     (N,)      uint16     — preamble accumulation count
  timestamp  (N,)      float64    — host-side time.monotonic() at parse

Usage:
  python capture_uwb.py
  python capture_uwb.py --port COM5
  python capture_uwb.py --port COM5 --baud 115200
"""

import argparse
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports

# ── Frame format constants ───────────────────────────────────────────────────
SYNC        = bytes([0xBC, 0xAD])
HEADER_FMT  = "<HHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
N_SAMPLES   = 1016
CIR_SIZE    = N_SAMPLES * 4
FRAME_SIZE  = 2 + HEADER_SIZE + CIR_SIZE + 1  # 4073

LOGDIR = Path("data/uwb/logs")


# ── Serial helpers ───────────────────────────────────────────────────────────

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
    """Return (seq, fp_index, rxpacc, cir_complex) or None on checksum error."""
    payload = raw[2:-1]
    if xor_checksum(payload) != raw[-1]:
        return None

    header_bytes = payload[:HEADER_SIZE]
    cir_bytes    = payload[HEADER_SIZE:]

    seq, fp_index_raw, rxpacc = struct.unpack(HEADER_FMT, header_bytes)
    fp_index = min(fp_index_raw >> 6, N_SAMPLES - 1)

    iq = np.frombuffer(cir_bytes, dtype="<i2").reshape(N_SAMPLES, 2).astype(np.float32)
    cir = iq[:, 0] + 1j * iq[:, 1]
    if rxpacc > 0:
        cir /= rxpacc

    return seq, fp_index, rxpacc, cir


# ── Capture loop ─────────────────────────────────────────────────────────────

def capture(port: str, baud: int):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    outpath = LOGDIR / f"cir_{now.strftime('%Y-%m-%d_%H-%M-%S')}.npz"

    seqs      = []
    fp_idxs   = []
    rxpaccs   = []
    cirs      = []
    timestamps = []

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
        return

    print(f"[uwb] Recording to {outpath}")
    print("[uwb] Press Ctrl+C to stop.\n")

    try:
        with ser:
            while True:
                chunk = ser.read(512)
                if not chunk:
                    now_t = time.monotonic()
                    if now_t - stats["last_diag"] >= 3.0:
                        elapsed = now_t - stats["t0"]
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
                        stats["last_diag"] = now_t
                    continue
                stats["bytes"] += len(chunk)

                if b"CIR:" in chunk:
                    try:
                        text = chunk.decode("ascii", errors="replace").strip()
                        for line in text.splitlines():
                            if "CIR:" in line:
                                print(f"[uwb] FIRMWARE: {line.strip()}")
                    except Exception:
                        pass

                buf.extend(chunk)

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
                        t_recv = time.monotonic()
                        seq, fp_index, rxpacc, cir = result
                        stats["frames"] += 1

                        seqs.append(seq)
                        fp_idxs.append(fp_index)
                        rxpaccs.append(rxpacc)
                        cirs.append(cir)
                        timestamps.append(t_recv)

                        if stats["frames"] == 1:
                            print(f"[uwb] First valid frame after "
                                  f"{t_recv - stats['t0']:.1f}s")
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

    except KeyboardInterrupt:
        pass

    n = stats["frames"]
    if n == 0:
        print("\n[uwb] No frames captured, nothing to save.")
        return

    elapsed = time.monotonic() - stats["t0"]
    fps = n / elapsed
    err_rate = stats["errors"] / max(n + stats["errors"], 1)

    ts_arr = np.array(timestamps, dtype=np.float64)
    ts_arr -= ts_arr[0]

    np.savez_compressed(
        outpath,
        cir=np.array(cirs, dtype=np.complex64),
        seq=np.array(seqs, dtype=np.uint16),
        fp_index=np.array(fp_idxs, dtype=np.uint16),
        rxpacc=np.array(rxpaccs, dtype=np.uint16),
        timestamp=ts_arr,
    )

    print(f"\n[uwb] Saved {n} frames to {outpath}")
    print(f"[uwb] {elapsed:.1f}s | {fps:.1f} fps | error rate {err_rate:.1%}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Capture UWB CIR frames to disk")
    parser.add_argument("--port", help="Serial port (auto-detected if omitted)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default 115200)")
    args = parser.parse_args()

    port = args.port or find_port()
    capture(port, args.baud)


if __name__ == "__main__":
    main()
