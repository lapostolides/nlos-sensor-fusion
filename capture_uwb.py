#!/usr/bin/env python3
"""
capture_uwb.py - Capture UWB CIR + timestamps from responder and initiator.

Reads binary frames from two DWM1001-Dev boards over UART:
  - Responder (rx): CIR accumulator + RX timestamp  (4078-byte frames)
  - Initiator (tx): TX timestamp only                (10-byte frames)

Press Ctrl+C to stop and flush to disk.

Responder .npz arrays:
  cir        (N, 1016) complex64  — normalised complex CIR samples
  seq        (N,)      uint16     — firmware frame counter
  fp_index   (N,)      uint16     — first-path index (10.6 → integer)
  rxpacc     (N,)      uint16     — preamble accumulation count
  rx_ts      (N,)      uint64     — DW1000 40-bit RX timestamp (~15.65 ps/tick)
  timestamp  (N,)      float64    — host-side time.monotonic() at parse

Initiator .npz arrays:
  seq        (N,)      uint16     — firmware frame counter
  tx_ts      (N,)      uint64     — DW1000 40-bit TX timestamp (~15.65 ps/tick)
  timestamp  (N,)      float64    — host-side time.monotonic() at parse

Usage:
  python capture_uwb.py                                        # auto-detect
  python capture_uwb.py --resp-port COM5                       # responder only
  python capture_uwb.py --resp-port COM5 --init-port COM6
  python capture_uwb.py --name my-experiment
"""

import argparse
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports

# ── Responder frame format (fixed length) ───────────────────────────────────────
#   1. Find 0xBC 0xAD
#   2. Read exactly 4076 more bytes (payload + xor)
#   3. Verify XOR, only then decode CIR
#   sync(2) + seq(2) + fp_index(2) + rxpacc(2) + rx_ts(5) + cir(4064) + xor(1)
SYNC                 = bytes([0xBC, 0xAD])
RESP_BYTES_AFTER_SYNC = 4076   # payload(4075) + xor(1)
RESP_FRAME            = 2 + RESP_BYTES_AFTER_SYNC    # 4078
RESP_HDR_FMT          = "<HHH"
RESP_HDR_SIZE         = struct.calcsize(RESP_HDR_FMT) # 6
RX_TS_SIZE            = 5
N_SAMPLES             = 1016
CIR_SIZE              = N_SAMPLES * 4                 # 4064

# ── Initiator frame format (10 bytes) ────────────────────────────────────────
#   sync(2) + seq(2) + tx_ts(5) + xor(1)
INIT_HDR_FMT  = "<H"
INIT_HDR_SIZE = struct.calcsize(INIT_HDR_FMT)         # 2
TX_TS_SIZE    = 5
INIT_FRAME    = 2 + INIT_HDR_SIZE + TX_TS_SIZE + 1    # 10

LOGDIR = Path("data/uwb/logs")


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_jlink_ports() -> list:
    """Return JLink/DWM serial port info objects, sorted by device name."""
    result = []
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").upper()
        if any(k in desc for k in ("JLINK", "J-LINK", "DWM")):
            result.append(p)
    result.sort(key=lambda p: p.device)
    return result


def xor_checksum(data: bytes) -> int:
    result = 0
    for b in data:
        result ^= b
    return result


def ts40_to_uint64(raw: bytes) -> np.uint64:
    """Convert a 5-byte little-endian DW1000 timestamp to uint64."""
    return np.uint64(int.from_bytes(raw, "little"))


def parse_resp_frame(raw: bytes):
    """Fixed-length parse: 1) verify XOR, 2) only then decode CIR. Returns None on failure."""
    if len(raw) != RESP_FRAME:
        return None

    payload = raw[2:-1]  # bytes after sync, excluding xor
    if xor_checksum(payload) != raw[-1]:
        return None

    # Decode only after XOR verified
    hdr_end = RESP_HDR_SIZE
    ts_end  = hdr_end + RX_TS_SIZE

    seq, fp_raw, rxpacc = struct.unpack(RESP_HDR_FMT, payload[:hdr_end])
    rx_ts    = ts40_to_uint64(payload[hdr_end:ts_end])
    fp_index = min(fp_raw >> 6, N_SAMPLES - 1)

    iq  = np.frombuffer(payload[ts_end:], dtype="<i2").reshape(N_SAMPLES, 2).astype(np.float32)
    cir = iq[:, 0] + 1j * iq[:, 1]
    # No rxpacc normalization for now (debugging)

    return seq, fp_index, rxpacc, rx_ts, cir


def parse_init_frame(raw: bytes):
    """Parse initiator frame → (seq, tx_ts) or None."""
    payload = raw[2:-1]
    if xor_checksum(payload) != raw[-1]:
        return None

    hdr_end = INIT_HDR_SIZE
    ts_end  = hdr_end + TX_TS_SIZE

    (seq,)  = struct.unpack(INIT_HDR_FMT, payload[:hdr_end])
    tx_ts   = ts40_to_uint64(payload[hdr_end:ts_end])

    return seq, tx_ts


# ── Generic binary-frame capture loop ────────────────────────────────────────

def _stream_loop(tag, ser, frame_size, parse_fn, on_frame, stop):
    """Find 0xBC 0xAD, read exactly (frame_size - 2) more bytes, verify XOR, decode."""
    buf = bytearray()
    stats = {
        "frames": 0, "errors": 0, "bytes": 0,
        "syncs": 0, "t0": time.monotonic(), "last_diag": time.monotonic(),
    }

    while not stop.is_set():
        chunk = ser.read(512)
        if not chunk:
            now_t = time.monotonic()
            if now_t - stats["last_diag"] >= 5.0:
                elapsed = now_t - stats["t0"]
                print(f"[{tag}] {elapsed:.0f}s | {stats['bytes']}B rx | "
                      f"{stats['syncs']} syncs | {stats['frames']} frames | "
                      f"{stats['errors']} chk errors")
                if stats["bytes"] == 0:
                    print(f"[{tag}]   ** No bytes received — check board power **")
                stats["last_diag"] = now_t
            continue

        stats["bytes"] += len(chunk)
        buf.extend(chunk)

        while len(buf) >= frame_size:
            idx = buf.find(SYNC)
            if idx == -1:
                buf = buf[-1:]
                break
            if idx > 0:
                buf = buf[idx:]
            if len(buf) < frame_size:
                break

            stats["syncs"] += 1
            raw = bytes(buf[:frame_size])
            result = parse_fn(raw)

            if result is not None:
                stats["frames"] += 1
                on_frame(result, time.monotonic())
                if stats["frames"] == 1:
                    print(f"[{tag}] First valid frame after "
                          f"{time.monotonic() - stats['t0']:.1f}s")
                buf = buf[frame_size:]
            else:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    print(f"[{tag}] Checksum mismatch #{stats['errors']}")
                buf = buf[1:]

            if stats["frames"] > 0 and stats["frames"] % 100 == 0:
                elapsed = time.monotonic() - stats["t0"]
                fps = stats["frames"] / elapsed
                err_rate = stats["errors"] / max(stats["frames"] + stats["errors"], 1)
                print(f"[{tag}] {stats['frames']} frames | {fps:.1f} fps | "
                      f"error rate {err_rate:.1%}")

    return stats


# ── Responder capture thread ─────────────────────────────────────────────────

def capture_responder(port, baud, outpath, stop):
    tag = "resp"
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        print(f"[{tag}] Opened {port} at {baud} baud")
    except Exception as e:
        print(f"[{tag}] Failed to open {port}: {e}")
        return

    print(f"[{tag}] Recording to {outpath}")

    seqs, fp_idxs, rxpaccs, rx_tss, cirs, timestamps = [], [], [], [], [], []

    def on_frame(result, t):
        seq, fp_index, rxpacc, rx_ts, cir = result
        seqs.append(seq)
        fp_idxs.append(fp_index)
        rxpaccs.append(rxpacc)
        rx_tss.append(rx_ts)
        cirs.append(cir)
        timestamps.append(t)

    with ser:
        stats = _stream_loop(tag, ser, RESP_FRAME, parse_resp_frame, on_frame, stop)

    n = stats["frames"]
    if n == 0:
        print(f"[{tag}] No frames captured.")
        return

    ts_arr = np.array(timestamps, dtype=np.float64)
    ts_arr -= ts_arr[0]

    np.savez_compressed(
        outpath,
        cir=np.array(cirs, dtype=np.complex64),
        seq=np.array(seqs, dtype=np.uint16),
        fp_index=np.array(fp_idxs, dtype=np.uint16),
        rxpacc=np.array(rxpaccs, dtype=np.uint16),
        rx_ts=np.array(rx_tss, dtype=np.uint64),
        timestamp=ts_arr,
    )

    elapsed = time.monotonic() - stats["t0"]
    print(f"[{tag}] Saved {n} frames to {outpath}  "
          f"({elapsed:.1f}s, {n / elapsed:.1f} fps)")


# ── Initiator capture thread ─────────────────────────────────────────────────

def capture_initiator(port, baud, outpath, stop):
    tag = "init"
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        print(f"[{tag}] Opened {port} at {baud} baud")
    except Exception as e:
        print(f"[{tag}] Failed to open {port}: {e}")
        return

    print(f"[{tag}] Recording to {outpath}")

    seqs, tx_tss, timestamps = [], [], []

    def on_frame(result, t):
        seq, tx_ts = result
        seqs.append(seq)
        tx_tss.append(tx_ts)
        timestamps.append(t)

    with ser:
        stats = _stream_loop(tag, ser, INIT_FRAME, parse_init_frame,
                             on_frame, stop)

    n = stats["frames"]
    if n == 0:
        print(f"[{tag}] No frames captured.")
        return

    ts_arr = np.array(timestamps, dtype=np.float64)
    ts_arr -= ts_arr[0]

    np.savez_compressed(
        outpath,
        seq=np.array(seqs, dtype=np.uint16),
        tx_ts=np.array(tx_tss, dtype=np.uint64),
        timestamp=ts_arr,
    )

    elapsed = time.monotonic() - stats["t0"]
    print(f"[{tag}] Saved {n} frames to {outpath}  "
          f"({elapsed:.1f}s, {n / elapsed:.1f} fps)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Capture UWB CIR + timestamps")
    parser.add_argument("--resp-port", help="Responder serial port (auto-detect if omitted)")
    parser.add_argument("--init-port", help="Initiator serial port (auto-detect if omitted)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--name", help="Run name (default: timestamp)")
    args = parser.parse_args()

    # ── Auto-detect JLink ports ──────────────────────────────────────────
    resp_port = args.resp_port
    init_port = args.init_port

    if not resp_port or not init_port:
        jlinks = find_jlink_ports()

        if not resp_port and not init_port:
            if len(jlinks) == 0:
                print("[uwb] No JLink ports found. Use --resp-port / --init-port.")
                return
            elif len(jlinks) == 1:
                resp_port = jlinks[0].device
                print(f"[uwb] Auto-detected responder: {resp_port} "
                      f"({jlinks[0].description})")
                print("[uwb] Only one JLink port found — skipping initiator.")
            else:
                resp_port = jlinks[0].device
                init_port = jlinks[1].device
                print(f"[uwb] Auto-detected responder: {resp_port} "
                      f"({jlinks[0].description})")
                print(f"[uwb] Auto-detected initiator: {init_port} "
                      f"({jlinks[1].description})")
                if len(jlinks) > 2:
                    print(f"[uwb] Note: {len(jlinks)} JLink ports found, "
                          f"using first two. Override with --resp-port / --init-port.")
        elif not resp_port:
            candidates = [p for p in jlinks if p.device != init_port]
            if candidates:
                resp_port = candidates[0].device
                print(f"[uwb] Auto-detected responder: {resp_port} "
                      f"({candidates[0].description})")
        elif not init_port:
            candidates = [p for p in jlinks if p.device != resp_port]
            if candidates:
                init_port = candidates[0].device
                print(f"[uwb] Auto-detected initiator: {init_port} "
                      f"({candidates[0].description})")

    if not resp_port:
        print("[uwb] No responder port specified or detected.")
        return

    # ── Run directory ────────────────────────────────────────────────────
    default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.name:
        run_name = args.name
    else:
        try:
            run_name = input(f"[uwb] Run name [{default_name}]: ").strip() or default_name
        except EOFError:
            run_name = default_name
    run_dir  = LOGDIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[uwb] Run: {run_name}")

    # ── Launch capture threads ───────────────────────────────────────────
    stop    = threading.Event()
    threads = []

    resp_thread = threading.Thread(
        target=capture_responder,
        args=(resp_port, args.baud, run_dir / "resp.npz", stop),
        daemon=True,
    )
    threads.append(resp_thread)

    if init_port:
        init_thread = threading.Thread(
            target=capture_initiator,
            args=(init_port, args.baud, run_dir / "init.npz", stop),
            daemon=True,
        )
        threads.append(init_thread)

    for t in threads:
        t.start()

    print("[uwb] Press Ctrl+C to stop.\n")

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    print("\n[uwb] Stopping...")
    stop.set()
    for t in threads:
        t.join(timeout=5)

    print("[uwb] Done.")


if __name__ == "__main__":
    main()
