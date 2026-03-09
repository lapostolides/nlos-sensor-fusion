#!/usr/bin/env python3
"""
capture_uwb.py - Capture UWB CIR + timestamps from 1 TX + up to 3 RX boards.

Boards are identified without hardcoding ports: Python sends ID\\n, the board
replies with its unique nRF52 FICR hardware ID, and a device_roles.json file
maps IDs to roles (tx, rx1, rx2, rx3).

Protocol (Python → board, before streaming):
  ID\\n    → board replies  ID:<16-hex-chars>\\r\\n
  ARM\\n   → board replies  OK\\r\\n  and starts streaming
  STOP\\n  → board deletes its FreeRTOS task (stops streaming)

First-time setup — run once to discover IDs:
  python capture_uwb.py --scan

Then create device_roles.json:
  { "A1B2C3D4E5F60718": "tx", "...": "rx1", "...": "rx2", "...": "rx3" }

Then capture:
  python capture_uwb.py
  python capture_uwb.py --name my-experiment

Output (.npz per board in data/uwb/logs/<run-name>/):
  tx.npz   — seq, tx_ts, timestamp
  rx1.npz  — cir, cir_i, cir_q, cir_mag, cir_phase, seq, fp_index, rxpacc, rx_ts, timestamp
  rx2.npz  — same as rx1
  rx3.npz  — same as rx1
"""

import argparse
import json
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports

# ── Frame format constants ────────────────────────────────────────────────────

SYNC                  = bytes([0xBC, 0xAD])

# Responder (rx*): CIR frame — 4078 bytes total
RESP_BYTES_AFTER_SYNC = 4076   # payload(4075) + xor(1)
RESP_FRAME            = 2 + RESP_BYTES_AFTER_SYNC
RESP_HDR_FMT          = "<HHH"
RESP_HDR_SIZE         = struct.calcsize(RESP_HDR_FMT)  # 6
RX_TS_SIZE            = 5
N_SAMPLES             = 1016
CIR_SIZE              = N_SAMPLES * 4                  # 4064

# Initiator (tx): TX timestamp frame — 10 bytes total
INIT_HDR_FMT  = "<H"
INIT_HDR_SIZE = struct.calcsize(INIT_HDR_FMT)          # 2
TX_TS_SIZE    = 5
INIT_FRAME    = 2 + INIT_HDR_SIZE + TX_TS_SIZE + 1     # 10

LOGDIR            = Path("data/uwb/logs")
DEVICE_ROLES_FILE = Path("device_roles.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_jlink_ports() -> list:
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
    return np.uint64(int.from_bytes(raw, "little"))


def parse_resp_frame(raw: bytes):
    if len(raw) != RESP_FRAME:
        return None
    payload = raw[2:-1]
    if xor_checksum(payload) != raw[-1]:
        return None
    hdr_end = RESP_HDR_SIZE
    ts_end  = hdr_end + RX_TS_SIZE
    seq, fp_raw, rxpacc = struct.unpack(RESP_HDR_FMT, payload[:hdr_end])
    rx_ts    = ts40_to_uint64(payload[hdr_end:ts_end])
    fp_index = min(fp_raw >> 6, N_SAMPLES - 1)
    iq  = np.frombuffer(payload[ts_end:], dtype="<i2").reshape(N_SAMPLES, 2).astype(np.float32)
    cir = iq[:, 0] + 1j * iq[:, 1]
    return seq, fp_index, rxpacc, rx_ts, cir


def parse_init_frame(raw: bytes):
    payload = raw[2:-1]
    if xor_checksum(payload) != raw[-1]:
        return None
    hdr_end = INIT_HDR_SIZE
    ts_end  = hdr_end + TX_TS_SIZE
    (seq,)  = struct.unpack(INIT_HDR_FMT, payload[:hdr_end])
    tx_ts   = ts40_to_uint64(payload[hdr_end:ts_end])
    return seq, tx_ts


# ── Port identification ───────────────────────────────────────────────────────

def probe_board_id(port: str, baud: int, timeout: float = 2.0) -> "str | None":
    """Open port, send ID\\n, return the 16-char hex ID string or None."""
    print(f"  {port}: requesting hardware ID...", end="", flush=True)
    try:
        ser = serial.Serial(port, baud, timeout=0.3)
    except Exception as e:
        print(f"  failed to open ({e})")
        return None
    try:
        ser.reset_input_buffer()
        ser.write(b"ID\n")
        deadline = time.monotonic() + timeout
        buf = b""
        while time.monotonic() < deadline:
            chunk = ser.read(ser.in_waiting or 1)
            buf += chunk
            if b"\n" in buf:
                break
        for line in buf.decode(errors="replace").splitlines():
            line = line.strip()
            if line.startswith("ID:"):
                hwid = line[3:].strip()
                print(f" {hwid}")
                return hwid
        print("  no response (wrong firmware or board not idle)")
        return None
    finally:
        ser.close()


def _scan_ports(baud: int) -> "dict[str, str]":
    """Probe all JLink ports and return {port: hwid} for responding boards."""
    jlinks = find_jlink_ports()
    if not jlinks:
        print("[uwb] No JLink ports found.")
        return {}
    print(f"\n[uwb] === Scanning: found {len(jlinks)} JLink port(s) ===")
    result = {}
    for p in jlinks:
        hwid = probe_board_id(p.device, baud)
        if hwid:
            result[p.device] = hwid
    return result


def discover_roles(baud: int) -> "dict[str, str]":
    """Probe all JLink ports, map hardware IDs to roles via device_roles.json."""
    port_to_id = _scan_ports(baud)
    if not port_to_id:
        return {}

    if not DEVICE_ROLES_FILE.exists():
        print(f"\n[uwb] {DEVICE_ROLES_FILE} not found.")
        print("[uwb] Create it with the following content, then re-run:\n")
        roles = ["tx", "rx1", "rx2", "rx3"]
        template = {hwid: roles[i] if i < len(roles) else f"rx{i}"
                    for i, hwid in enumerate(port_to_id.values())}
        print(f"  {json.dumps(template, indent=2)}\n")
        return {}

    with open(DEVICE_ROLES_FILE) as f:
        id_to_role = json.load(f)

    print(f"\n[uwb] === Mapping roles from {DEVICE_ROLES_FILE} ===")
    role_to_port = {}
    for port, hwid in port_to_id.items():
        role = id_to_role.get(hwid)
        if role is None:
            print(f"  {port}: {hwid} → unknown (add to {DEVICE_ROLES_FILE})")
            continue
        role_to_port[role] = port
        print(f"  {port}: {hwid} → {role}")

    return role_to_port


# ── ARM / STOP helpers ────────────────────────────────────────────────────────

def arm_board(ser: serial.Serial, tag: str, timeout: float = 3.0) -> bool:
    """Send ARM\\n, wait for OK\\r\\n. Returns True on success."""
    print(f"[{tag}] Arming ({ser.port})...", end="", flush=True)
    ser.reset_input_buffer()
    ser.write(b"ARM\n")
    deadline = time.monotonic() + timeout
    buf = b""
    while time.monotonic() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        buf += chunk
        if b"OK" in buf:
            print(" OK")
            return True
    print(f" no response — board may not be running ID firmware")
    return False


# ── Generic binary-frame stream loop ─────────────────────────────────────────

def _stream_loop(tag, ser, frame_size, parse_fn, on_frame, stop):
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
                    print(f"[{tag}]   ** No bytes received — check board power and ARM **")
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


# ── Capture threads ───────────────────────────────────────────────────────────

def capture_responder(tag, port, baud, outpath, stop):
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        print(f"[{tag}] Opened {port} at {baud} baud")
    except Exception as e:
        print(f"[{tag}] Failed to open {port}: {e}")
        return

    if not arm_board(ser, tag):
        ser.close()
        return

    print(f"[{tag}] Armed — recording to {outpath}")

    seqs, fp_idxs, rxpaccs, rx_tss, cirs, timestamps = [], [], [], [], [], []

    def on_frame(result, t):
        seq, fp_index, rxpacc, rx_ts, cir = result
        seqs.append(seq)
        fp_idxs.append(fp_index)
        rxpaccs.append(rxpacc)
        rx_tss.append(rx_ts)
        cirs.append(cir)
        timestamps.append(t)

    try:
        stats = _stream_loop(tag, ser, RESP_FRAME, parse_resp_frame, on_frame, stop)
    finally:
        try:
            ser.write(b"STOP\n")
        except Exception:
            pass
        ser.close()

    n = stats["frames"]
    if n == 0:
        print(f"[{tag}] No frames captured.")
        return

    ts_arr    = np.array(timestamps, dtype=np.float64)
    # Wall-clock epoch seconds at the moment of the first frame (used by sync_data.py).
    wall_start = np.float64(time.time() - (time.monotonic() - timestamps[0]))
    ts_arr   -= ts_arr[0]
    cir_arr   = np.array(cirs, dtype=np.complex64)
    cir_i     = cir_arr.real.astype(np.float32)
    cir_q     = cir_arr.imag.astype(np.float32)
    cir_mag   = np.abs(cir_arr).astype(np.float32)
    cir_phase = np.angle(cir_arr).astype(np.float32)

    np.savez_compressed(
        outpath,
        cir=cir_arr, cir_i=cir_i, cir_q=cir_q,
        cir_mag=cir_mag, cir_phase=cir_phase,
        seq=np.array(seqs, dtype=np.uint16),
        fp_index=np.array(fp_idxs, dtype=np.uint16),
        rxpacc=np.array(rxpaccs, dtype=np.uint16),
        rx_ts=np.array(rx_tss, dtype=np.uint64),
        timestamp=ts_arr,
        timestamp_wall_start=wall_start,
    )
    elapsed = time.monotonic() - stats["t0"]
    print(f"[{tag}] Saved {n} frames to {outpath}  ({elapsed:.1f}s, {n/elapsed:.1f} fps)")


def capture_initiator(tag, port, baud, outpath, stop):
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        print(f"[{tag}] Opened {port} at {baud} baud")
    except Exception as e:
        print(f"[{tag}] Failed to open {port}: {e}")
        return

    if not arm_board(ser, tag):
        ser.close()
        return

    print(f"[{tag}] Armed — recording to {outpath}")

    seqs, tx_tss, timestamps = [], [], []

    def on_frame(result, t):
        seq, tx_ts = result
        seqs.append(seq)
        tx_tss.append(tx_ts)
        timestamps.append(t)

    try:
        stats = _stream_loop(tag, ser, INIT_FRAME, parse_init_frame, on_frame, stop)
    finally:
        try:
            ser.write(b"STOP\n")
        except Exception:
            pass
        ser.close()

    n = stats["frames"]
    if n == 0:
        print(f"[{tag}] No frames captured.")
        return

    ts_arr  = np.array(timestamps, dtype=np.float64)
    wall_start = np.float64(time.time() - (time.monotonic() - timestamps[0]))
    ts_arr -= ts_arr[0]

    np.savez_compressed(
        outpath,
        seq=np.array(seqs, dtype=np.uint16),
        tx_ts=np.array(tx_tss, dtype=np.uint64),
        timestamp=ts_arr,
        timestamp_wall_start=wall_start,
    )
    elapsed = time.monotonic() - stats["t0"]
    print(f"[{tag}] Saved {n} frames to {outpath}  ({elapsed:.1f}s, {n/elapsed:.1f} fps)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Capture UWB CIR + timestamps (1 TX + up to 3 RX)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--name", help="Run name (default: timestamp)")
    args = parser.parse_args()

    # Scanning always happens — no separate --scan flag needed.
    role_to_port = discover_roles(args.baud)
    if not role_to_port:
        return

    missing = {"tx", "rx1", "rx2", "rx3"} - set(role_to_port)
    if missing:
        print(f"\n[uwb] Roles not found: {sorted(missing)} — continuing with {sorted(role_to_port)}")

    # ── Run directory ─────────────────────────────────────────────────────
    default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.name:
        run_name = args.name
    else:
        try:
            run_name = input(f"\n[uwb] Run name [{default_name}]: ").strip() or default_name
        except EOFError:
            run_name = default_name

    run_dir = LOGDIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[uwb] Output: {run_dir}\n")

    # ── Launch capture threads ────────────────────────────────────────────
    print(f"[uwb] === Arming {len(role_to_port)} board(s) ===")
    stop    = threading.Event()
    threads = []

    for role, port in sorted(role_to_port.items()):
        outpath = run_dir / f"{role}.npz"
        fn      = capture_initiator if role == "tx" else capture_responder
        t = threading.Thread(
            target=fn,
            args=(role, port, args.baud, outpath, stop),
            daemon=True,
        )
        threads.append(t)

    for t in threads:
        t.start()

    print(f"\n[uwb] === Capturing — Press Ctrl+C to stop ===\n")

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
