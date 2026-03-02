"""
PANS firmware UART shell reader for DWM1001-Dev.

The default PANS firmware exposes a text shell over USB-UART at 115200 baud.
Wake the shell with two newlines, then send commands and read responses.

Useful commands:
  si   - system info (node role, firmware version, etc.)
  lec  - continuous location engine output (tag mode only)
  les  - location engine statistics
  help - list all commands

Usage:
  python uwb_pans_reader.py              # auto-detect port, print si
  python uwb_pans_reader.py --port COM4  # specify port
  python uwb_pans_reader.py --stream     # stream continuous shell output
"""

import argparse
import glob
import sys
import time

import serial
import serial.tools.list_ports


BAUD = 115200
PROMPT = b"dwm> "


def find_dwm_port() -> str:
    """
    Return the first serial port that looks like a DWM1001-Dev UART shell.
    The board exposes two COM ports via J-Link OB; the shell is on the one
    whose description contains 'JLink' or 'USB Serial'.
    Falls back to prompting the user if ambiguous.
    """
    ports = list(serial.tools.list_ports.comports())
    candidates = [
        p for p in ports
        if any(kw in (p.description or "").upper()
               for kw in ("JLINK", "J-LINK", "USB SERIAL", "DWM"))
    ]

    if len(candidates) == 1:
        print(f"[uwb] Auto-detected port: {candidates[0].device} ({candidates[0].description})")
        return candidates[0].device

    if len(candidates) > 1:
        print("[uwb] Multiple candidate ports found:")
        for i, p in enumerate(candidates):
            print(f"  [{i}] {p.device} — {p.description}")
        idx = int(input("Select index: "))
        return candidates[idx].device

    # Fallback: list all ports
    if ports:
        print("[uwb] No DWM port auto-detected. Available ports:")
        for i, p in enumerate(ports):
            print(f"  [{i}] {p.device} — {p.description}")
        idx = int(input("Select index: "))
        return ports[idx].device

    raise RuntimeError("No serial ports found. Is the board plugged in?")


def wake_shell(ser: serial.Serial) -> bool:
    """
    Send newlines to activate the PANS shell.
    Returns True if the dwm> prompt is received.
    """
    ser.reset_input_buffer()
    for _ in range(3):
        ser.write(b"\r\n")
        time.sleep(0.3)
    return _wait_for_prompt(ser, timeout=3.0)


def _wait_for_prompt(ser: serial.Serial, timeout: float = 3.0) -> bool:
    buf = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            if PROMPT in buf:
                return True
    return False


def send_command(ser: serial.Serial, cmd: str, timeout: float = 3.0) -> str:
    """
    Send a shell command and return the response text (up to the next prompt).
    """
    ser.reset_input_buffer()
    ser.write((cmd + "\r\n").encode())

    buf = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            if PROMPT in buf:
                break

    # Strip the echoed command and trailing prompt
    text = buf.decode(errors="replace")
    text = text.replace(cmd, "", 1).replace("dwm> ", "").strip()
    return text


def stream(ser: serial.Serial):
    """Print raw shell output continuously (Ctrl-C to stop)."""
    print("[uwb] Streaming shell output (Ctrl-C to stop)...")
    try:
        while True:
            data = ser.read(ser.in_waiting or 1)
            if data:
                sys.stdout.write(data.decode(errors="replace"))
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[uwb] Stream stopped.")


def open_port(port: str) -> serial.Serial:
    return serial.Serial(
        port=port,
        baudrate=BAUD,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.1,
    )


def read_system_info(port: str = None) -> dict:
    """
    Connect to PANS shell, send 'si', parse key=value lines, return as dict.
    """
    port = port or find_dwm_port()
    with open_port(port) as ser:
        if not wake_shell(ser):
            raise RuntimeError("No dwm> prompt received. Wrong port or board not running PANS?")
        raw = send_command(ser, "si")

    info = {}
    for line in raw.splitlines():
        line = line.strip()
        if "=" in line:
            k, _, v = line.partition("=")
            info[k.strip()] = v.strip()
        elif line:
            info.setdefault("_lines", []).append(line)
    return info


def scan_ports():
    """
    List all serial ports and print any bytes they emit in 1 second.
    Useful for finding which port is the PANS shell.
    """
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    print(f"Found {len(ports)} port(s):\n")
    for p in ports:
        print(f"  {p.device} — {p.description} (hwid: {p.hwid})")
        try:
            with open_port(p.device) as ser:
                # Send wake and listen for 1.5s
                ser.write(b"\r\n\r\n")
                time.sleep(1.5)
                data = ser.read(ser.in_waiting)
                if data:
                    preview = data[:120].decode(errors="replace").replace("\r", "").replace("\n", " ")
                    print(f"    -> received {len(data)} bytes: {preview!r}")
                else:
                    print(f"    -> no data received")
        except Exception as e:
            print(f"    -> could not open: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="DWM1001 PANS shell reader")
    parser.add_argument("--port", help="Serial port (e.g. COM4). Auto-detected if omitted.")
    parser.add_argument("--stream", action="store_true", help="Stream raw shell output")
    parser.add_argument("--cmd", default="si", help="Shell command to run (default: si)")
    parser.add_argument("--scan", action="store_true", help="Scan all ports and show what they emit")
    args = parser.parse_args()

    if args.scan:
        scan_ports()
        return

    port = args.port or find_dwm_port()

    with open_port(port) as ser:
        print(f"[uwb] Opened {port} at {BAUD} baud")
        if not wake_shell(ser):
            print("[uwb] WARNING: No dwm> prompt. Wrong port or board not in PANS shell mode.")
            print("[uwb] Falling through to stream anyway...")

        if args.stream:
            stream(ser)
        else:
            response = send_command(ser, args.cmd)
            print(f"\n--- {args.cmd} ---")
            print(response)


if __name__ == "__main__":
    main()
