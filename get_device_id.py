#!/usr/bin/env python3
"""
get_device_id.py - Register a UWB board's hardware ID to a role.

Run once per board to build device_roles.json.  Plug in the board you want to
label, then run:

    python get_device_id.py tx
    python get_device_id.py rx1
    python get_device_id.py rx2
    python get_device_id.py rx3

The script scans all connected JLink ports, finds boards not yet in
device_roles.json, and saves the mapping.  If only one new board is found it
is assigned automatically.  If multiple new boards are found you are prompted
to pick one by port.
"""

import argparse
import json
import sys
from pathlib import Path

import serial
import serial.tools.list_ports
import time

DEVICE_ROLES_FILE = Path("device_roles.json")
VALID_ROLES = ("tx", "rx1", "rx2", "rx3")


def find_jlink_ports():
    result = []
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").upper()
        if any(k in desc for k in ("JLINK", "J-LINK", "DWM")):
            result.append(p)
    result.sort(key=lambda p: p.device)
    return result


def probe_board_id(port: str, baud: int = 115200, timeout: float = 2.0) -> "str | None":
    print(f"  {port}: requesting hardware ID...", end="", flush=True)
    try:
        ser = serial.Serial(port, baud, timeout=0.3)
    except Exception as e:
        print(f" failed to open ({e})")
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
        print(" no response (wrong firmware or board not powered)")
        return None
    finally:
        ser.close()


def load_roles() -> dict:
    if DEVICE_ROLES_FILE.exists():
        with open(DEVICE_ROLES_FILE) as f:
            return json.load(f)
    return {}


def save_roles(roles: dict):
    with open(DEVICE_ROLES_FILE, "w") as f:
        json.dump(roles, f, indent=2)
    print(f"\nSaved to {DEVICE_ROLES_FILE}:")
    for hwid, role in roles.items():
        print(f"  {hwid}: {role}")


def main():
    parser = argparse.ArgumentParser(
        description="Register a UWB board's hardware ID to a role in device_roles.json."
    )
    parser.add_argument(
        "role",
        choices=VALID_ROLES,
        help="Role to assign to the discovered board.",
    )
    args = parser.parse_args()
    role = args.role

    existing = load_roles()
    already_assigned = {r: hwid for hwid, r in existing.items()}

    if role in already_assigned:
        print(f"[get_device_id] '{role}' is already mapped to {already_assigned[role]}.")
        try:
            ans = input("Overwrite? [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    # Scan all JLink ports
    jlinks = find_jlink_ports()
    if not jlinks:
        print("[get_device_id] No JLink ports found. Is the board plugged in?")
        sys.exit(1)

    print(f"\n[get_device_id] Scanning {len(jlinks)} JLink port(s)...")
    found: dict[str, str] = {}  # port -> hwid
    for p in jlinks:
        hwid = probe_board_id(p.device)
        if hwid:
            found[p.device] = hwid

    if not found:
        print("\n[get_device_id] No boards responded. Check firmware and power.")
        sys.exit(1)

    # Filter to boards not yet registered (or the one being overwritten)
    registered_ids = set(existing.keys()) - {already_assigned.get(role, "")}
    new_boards = {port: hwid for port, hwid in found.items()
                  if hwid not in registered_ids}

    if not new_boards:
        print("\n[get_device_id] All found boards are already registered:")
        for port, hwid in found.items():
            print(f"  {port}: {hwid} → {existing.get(hwid, '(unknown)')}")
        print("Plug in the new board and try again.")
        sys.exit(1)

    # Pick the board to assign
    if len(new_boards) == 1:
        port, hwid = next(iter(new_boards.items()))
        print(f"\n[get_device_id] Found 1 new board: {port} → {hwid}")
    else:
        print(f"\n[get_device_id] Found {len(new_boards)} unregistered boards:")
        options = list(new_boards.items())
        for i, (port, hwid) in enumerate(options):
            print(f"  [{i}] {port}: {hwid}")
        try:
            idx = int(input("Which board to assign as '{}': ".format(role)).strip())
        except (ValueError, EOFError):
            print("Invalid selection. Aborted.")
            sys.exit(1)
        if not (0 <= idx < len(options)):
            print("Out of range. Aborted.")
            sys.exit(1)
        port, hwid = options[idx]

    # Remove old entry for this role if overwriting
    updated = {k: v for k, v in existing.items() if v != role}
    updated[hwid] = role
    save_roles(updated)

    print(f"\n[get_device_id] Done. '{role}' → {hwid} ({port})")

    # Show remaining unregistered roles
    assigned_roles = set(updated.values())
    remaining = [r for r in VALID_ROLES if r not in assigned_roles]
    if remaining:
        print(f"\nStill unregistered: {remaining}")
        print(f"Run: python get_device_id.py {remaining[0]}")
    else:
        print("\nAll 4 roles registered. Ready to capture:")
        print("  python capture_uwb.py")


if __name__ == "__main__":
    main()
