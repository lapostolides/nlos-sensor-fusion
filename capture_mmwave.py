#!/usr/bin/env python3
"""
capture_mmwave.py - Capture data from TI IWR6843ISK mmWave radar over UART.

The IWR6843ISK exposes two UART ports via its XDS110 debug probe:
  - CLI port  (115200 baud): accepts chirp config commands (lower COM number)
  - Data port (921600 baud): streams TLV-framed detection output (higher COM)

Ports are auto-detected by USB VID/PID (TI XDS110). Override with --cli/--data.

Usage:
  python capture_mmwave.py
  python capture_mmwave.py --name my-experiment
  python capture_mmwave.py --cfg radar_profile.cfg
  python capture_mmwave.py --cli COM3 --data COM4   # manual override

Output (.npz in data/radar/logs/<run-name>/):
  radar.npz  — detected_points, range_profile, point_snr, point_noise,
               frame_number, timestamp
"""

import argparse
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports

# ── TLV frame constants ──────────────────────────────────────────────────────

MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"

# Frame header: 8 fields, each uint32 (after magic word)
FRAME_HDR_FMT = "<8I"
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)  # 32

# TLV header
TLV_HDR_FMT = "<2I"
TLV_HDR_SIZE = struct.calcsize(TLV_HDR_FMT)  # 8

# TLV types (OOB demo)
TLV_DETECTED_POINTS = 1       # x, y, z, velocity (float32 each per object)
TLV_RANGE_PROFILE = 2         # uint16 array (range bins)
TLV_NOISE_PROFILE = 3         # uint16 array
TLV_AZIMUTH_HEATMAP = 4       # complex float per antenna x range bin
TLV_RANGE_DOPPLER = 5         # uint16 heatmap
TLV_STATS = 6                 # timing stats
TLV_SIDE_INFO = 7             # SNR + noise per detected point (uint16 each)
TLV_AZIMUTH_ELEVATION = 8     # spherical coordinates

POINT_STRUCT_SIZE = 16  # 4 x float32 (x, y, z, velocity)
SIDE_INFO_SIZE = 4      # 2 x uint16 (snr, noise)

LOGDIR = Path("data/radar/logs")

# Default chirp config for IWR6843ISK short-range people detection.
# Override with --cfg <file> to load a custom profile.
DEFAULT_CFG = """\
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 60.75 5 7 57.14 0 0 70 1 256 5209 0 0 158
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 2
chirpCfg 2 2 0 0 0 0 0 4
frameCfg 0 2 16 0 100 1 0
guiMonitor -1 1 1 0 0 0 1
cfarCfg -1 0 2 8 4 3 0 15 1
cfarCfg -1 1 0 4 2 3 1 15 1
multiObjBeamForming -1 1 0.5
clutterRemoval -1 0
calibDcRangeSig -1 0 -5 8 256
extendedMaxVelocity -1 0
lvdsStreamCfg -1 0 0 0
compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
measureRangeBiasAndRxChanPhase 0 1.5 0.2
CQRxSatMonitor 0 3 5 121 0
CQSigImgMonitor 0 127 4
analogMonitor 0 0
aoaFovCfg -1 -90 90 -90 90
cfarFovCfg -1 0 0 9.0
cfarFovCfg -1 1 -1 1.0
calibData 0 0 0
sensorStart
"""


# ── Port auto-detection ───────────────────────────────────────────────────────

def find_iwr_ports() -> list:
    """Find IWR6843ISK serial ports (FTDI dual-UART bridge).

    The IWR6843ISK uses an FTDI FT2232 chip exposing two COM ports. We detect by:
      - USB VID 0x0451 (Texas Instruments), or
      - FTDI VID 0x0403, or
      - Description containing 'XDS' or 'FTDI'

    Returns ports sorted by COM number (lower = CLI, higher = data).
    """
    result = []
    for p in serial.tools.list_ports.comports():
        hwid = (p.hwid or "").upper()
        desc = (p.description or "").upper()
        if "0451" in hwid:                     # TI VID
            result.append(p)
        elif "0403" in hwid:                   # FTDI VID
            result.append(p)
        elif any(k in desc for k in ("XDS", "FTDI")):
            result.append(p)
    result.sort(key=lambda p: p.device)
    return result


def detect_ports() -> tuple[str, str] | None:
    """Auto-detect CLI and data ports. Returns (cli_port, data_port) or None."""
    ports = find_iwr_ports()
    if len(ports) < 2:
        if ports:
            print(f"[mmWave] Found only 1 port ({ports[0].device}) — need 2.")
            print(f"        Description: {ports[0].description}")
        else:
            print("[mmWave] No IWR6843ISK ports found.")
            print("        Install FTDI drivers from:")
            print("          C:\\ti\\mmwave_sdk_03_06_02_00-LTS\\tools\\ftdi\\")
            print("        Then unplug and re-plug the board.")
            print("        Available COM ports:")
            for p in serial.tools.list_ports.comports():
                print(f"          {p.device}: {p.description}  [{p.hwid}]")
        return None

    cli_port = ports[0].device
    data_port = ports[1].device

    print(f"[mmWave] Auto-detected ports:")
    for p in ports[:2]:
        print(f"  {p.device}: {p.description}")
    print(f"[mmWave] CLI  port: {cli_port}")
    print(f"[mmWave] Data port: {data_port}")

    return cli_port, data_port


# ── Frame parsing ─────────────────────────────────────────────────────────────

def parse_frame(data: bytes):
    """Parse one TLV frame. Returns dict of parsed TLVs or None on error."""
    if len(data) < len(MAGIC_WORD) + FRAME_HDR_SIZE:
        return None

    offset = len(MAGIC_WORD)
    (version, total_len, platform, frame_num, time_cpu,
     num_obj, num_tlvs, sub_frame) = struct.unpack_from(FRAME_HDR_FMT, data, offset)
    offset += FRAME_HDR_SIZE

    result = {
        "frame_number": frame_num,
        "num_detected_obj": num_obj,
        "num_tlvs": num_tlvs,
        "time_cpu_cycles": time_cpu,
        "platform": platform,
        "version": version,
    }

    for _ in range(num_tlvs):
        if offset + TLV_HDR_SIZE > len(data):
            break
        tlv_type, tlv_len = struct.unpack_from(TLV_HDR_FMT, data, offset)
        offset += TLV_HDR_SIZE
        payload = data[offset:offset + tlv_len]
        offset += tlv_len

        if tlv_type == TLV_DETECTED_POINTS and len(payload) >= num_obj * POINT_STRUCT_SIZE:
            pts = np.frombuffer(payload[:num_obj * POINT_STRUCT_SIZE],
                                dtype=np.float32).reshape(num_obj, 4)
            result["detected_points"] = pts  # columns: x, y, z, velocity

        elif tlv_type == TLV_RANGE_PROFILE:
            result["range_profile"] = np.frombuffer(payload, dtype=np.uint16).copy()

        elif tlv_type == TLV_NOISE_PROFILE:
            result["noise_profile"] = np.frombuffer(payload, dtype=np.uint16).copy()

        elif tlv_type == TLV_SIDE_INFO and len(payload) >= num_obj * SIDE_INFO_SIZE:
            si = np.frombuffer(payload[:num_obj * SIDE_INFO_SIZE],
                               dtype=np.uint16).reshape(num_obj, 2)
            result["point_snr"] = si[:, 0]
            result["point_noise"] = si[:, 1]

        elif tlv_type == TLV_STATS:
            result["stats_raw"] = payload

    return result


# ── UART helpers ──────────────────────────────────────────────────────────────

def send_config(cli_port: serial.Serial, cfg_text: str):
    """Send chirp config line-by-line to the CLI port and print responses."""
    print("[mmWave] Sending configuration...")
    for line in cfg_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        cli_port.write((line + "\n").encode())
        time.sleep(0.03)
        resp = cli_port.read(cli_port.in_waiting or 0).decode(errors="replace").strip()
        status = f" -> {resp}" if resp else ""
        print(f"  {line}{status}")
    time.sleep(0.1)
    remaining = cli_port.read(cli_port.in_waiting or 0).decode(errors="replace").strip()
    if remaining:
        for r in remaining.splitlines():
            print(f"  {r}")
    print("[mmWave] Configuration sent.")


def read_frame(data_port: serial.Serial, timeout: float = 2.0) -> bytes | None:
    """Block until a full TLV frame is received or timeout expires."""
    buf = bytearray()
    deadline = time.monotonic() + timeout

    # Sync to magic word
    while time.monotonic() < deadline:
        chunk = data_port.read(max(data_port.in_waiting, 1))
        if not chunk:
            continue
        buf.extend(chunk)
        idx = buf.find(MAGIC_WORD)
        if idx >= 0:
            buf = buf[idx:]
            break
    else:
        return None

    # Read enough for frame header
    while len(buf) < len(MAGIC_WORD) + FRAME_HDR_SIZE:
        if time.monotonic() > deadline:
            return None
        chunk = data_port.read(max(data_port.in_waiting, 1))
        if chunk:
            buf.extend(chunk)

    # Extract total packet length from header
    total_len = struct.unpack_from("<I", buf, len(MAGIC_WORD) + 4)[0]
    if total_len > 65536:
        return None  # sanity check

    # Read remaining bytes
    while len(buf) < total_len:
        if time.monotonic() > deadline:
            return None
        remaining = total_len - len(buf)
        chunk = data_port.read(min(remaining, 4096))
        if chunk:
            buf.extend(chunk)

    return bytes(buf[:total_len])


# ── Main capture loop ────────────────────────────────────────────────────────

def capture(cli_port_name: str, data_port_name: str, cfg_text: str,
            run_dir: Path, cli_baud: int = 115200, data_baud: int = 921600):
    cli_ser = serial.Serial(cli_port_name, cli_baud, timeout=0.5)
    print(f"[mmWave] CLI port opened: {cli_port_name} @ {cli_baud}")

    data_ser = serial.Serial(data_port_name, data_baud, timeout=0.5)
    print(f"[mmWave] Data port opened: {data_port_name} @ {data_baud}")

    send_config(cli_ser, cfg_text)

    run_dir.mkdir(parents=True, exist_ok=True)
    outpath = run_dir / "radar.npz"
    print(f"[mmWave] Output: {outpath}")
    print("[mmWave] Capturing -- press Ctrl+C to stop\n")

    all_points = []
    all_snr = []
    all_noise = []
    all_range_prof = []
    frame_numbers = []
    timestamps = []
    point_counts = []

    t0 = time.monotonic()
    n_frames = 0

    try:
        while True:
            raw = read_frame(data_ser)
            if raw is None:
                continue

            parsed = parse_frame(raw)
            if parsed is None:
                continue

            t_now = time.monotonic()
            n_frames += 1
            frame_numbers.append(parsed["frame_number"])
            timestamps.append(t_now)

            pts = parsed.get("detected_points", np.zeros((0, 4), dtype=np.float32))
            all_points.append(pts)
            point_counts.append(len(pts))

            all_snr.append(parsed.get("point_snr", np.zeros(len(pts), dtype=np.uint16)))
            all_noise.append(parsed.get("point_noise", np.zeros(len(pts), dtype=np.uint16)))

            rp = parsed.get("range_profile")
            if rp is not None:
                all_range_prof.append(rp)

            if n_frames == 1:
                print(f"[mmWave] First frame after {t_now - t0:.1f}s  "
                      f"({parsed['num_detected_obj']} objects)")

            if n_frames % 100 == 0:
                elapsed = t_now - t0
                fps = n_frames / elapsed
                total_pts = sum(point_counts)
                print(f"[mmWave] {n_frames} frames | {fps:.1f} fps | "
                      f"{total_pts} total detections")

    except KeyboardInterrupt:
        pass

    print("\n[mmWave] Stopping sensor...")
    try:
        cli_ser.write(b"sensorStop\n")
        time.sleep(0.1)
    except Exception:
        pass
    cli_ser.close()
    data_ser.close()

    if n_frames == 0:
        print("[mmWave] No frames captured.")
        return

    ts_arr = np.array(timestamps, dtype=np.float64)
    wall_start = np.float64(time.time() - (time.monotonic() - timestamps[0]))
    ts_arr -= ts_arr[0]

    if sum(point_counts) > 0:
        points_cat = np.concatenate(all_points, axis=0)
        snr_cat = np.concatenate(all_snr)
        noise_cat = np.concatenate(all_noise)
    else:
        points_cat = np.zeros((0, 4), dtype=np.float32)
        snr_cat = np.zeros(0, dtype=np.uint16)
        noise_cat = np.zeros(0, dtype=np.uint16)

    save_dict = dict(
        detected_points=points_cat,
        point_snr=snr_cat,
        point_noise=noise_cat,
        point_counts=np.array(point_counts, dtype=np.uint32),
        frame_number=np.array(frame_numbers, dtype=np.uint32),
        timestamp=ts_arr,
        timestamp_wall_start=wall_start,
    )

    if len(all_range_prof) == n_frames and len(set(len(r) for r in all_range_prof)) == 1:
        save_dict["range_profile"] = np.array(all_range_prof, dtype=np.uint16)

    np.savez_compressed(outpath, **save_dict)
    elapsed = time.monotonic() - t0
    print(f"[mmWave] Saved {n_frames} frames to {outpath}  "
          f"({elapsed:.1f}s, {n_frames / elapsed:.1f} fps, "
          f"{sum(point_counts)} total detections)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Capture IWR6843ISK mmWave radar data over UART")
    parser.add_argument("--cli", help="CLI port override (e.g. COM3)")
    parser.add_argument("--data", help="Data port override (e.g. COM4)")
    parser.add_argument("--name", help="Run name (default: timestamp)")
    parser.add_argument("--cfg", help="Path to chirp config .cfg file "
                        "(default: built-in short-range profile)")
    parser.add_argument("--cli-baud", type=int, default=115200,
                        help="CLI port baud rate (default: 115200)")
    parser.add_argument("--data-baud", type=int, default=921600,
                        help="Data port baud rate (default: 921600)")
    args = parser.parse_args()

    # Resolve ports
    if args.cli and args.data:
        cli_port, data_port = args.cli, args.data
    elif args.cli or args.data:
        parser.error("Provide both --cli and --data, or neither for auto-detect.")
        return
    else:
        result = detect_ports()
        if result is None:
            return
        cli_port, data_port = result

    # Load config
    if args.cfg:
        cfg_path = Path(args.cfg)
        if not cfg_path.exists():
            print(f"[mmWave] Config file not found: {cfg_path}")
            return
        cfg_text = cfg_path.read_text()
        print(f"[mmWave] Using config: {cfg_path}")
    else:
        cfg_text = DEFAULT_CFG
        print("[mmWave] Using default short-range config")

    # Run directory
    default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.name:
        run_name = args.name
    else:
        try:
            run_name = input(f"[mmWave] Run name [{default_name}]: ").strip() or default_name
        except EOFError:
            run_name = default_name

    run_dir = LOGDIR / run_name

    capture(cli_port, data_port, cfg_text, run_dir,
            cli_baud=args.cli_baud, data_baud=args.data_baud)


if __name__ == "__main__":
    main()
