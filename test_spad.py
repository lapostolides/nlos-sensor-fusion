#!/usr/bin/env python3
"""Minimal SPAD diagnostic — isolates serial communication."""

import struct
import time
import serial
import serial.tools.list_ports


def find_spad_port():
    for p in serial.tools.list_ports.comports():
        tag = ""
        if p.vid == 0x0483:
            tag = "  ← STMicro (SPAD)"
        elif "JLINK" in (p.description or "").upper():
            tag = "  ← JLink (UWB)"
        print(f"  {p.device}: vid=0x{p.vid or 0:04X} pid=0x{p.pid or 0:04X} "
              f"desc={p.description!r}{tag}")

    stm = [p for p in serial.tools.list_ports.comports() if p.vid == 0x0483]
    if not stm:
        print("ERROR: No STMicro port found.")
        return None
    return stm[0].device


def try_baud(port, baud, send_config=False):
    """Open port at given baud, optionally send config, read for 3s."""
    try:
        ser = serial.Serial(port, baud, timeout=0.5)
    except Exception as e:
        print(f"    {baud:>10}: FAILED to open ({e})")
        return False
    time.sleep(0.5)

    # Drain any startup message
    raw = ser.read(ser.in_waiting or 0)

    if send_config:
        config = struct.pack(
            "<13H",
            16, 1, 30, 10, 0, 8, 16, 0, 0, 1, 1, 4, 4,
        )
        ser.write(config)
        time.sleep(1.5)

    # Read for 2 seconds
    t0 = time.time()
    total_bytes = 0
    first_chunk = b""
    while time.time() - t0 < 2:
        n = ser.in_waiting
        if n:
            chunk = ser.read(n)
            total_bytes += len(chunk)
            if not first_chunk:
                first_chunk = chunk
        else:
            time.sleep(0.01)

    ser.close()

    if total_bytes > 0:
        try:
            preview = first_chunk[:120].decode(errors="replace")
        except Exception:
            preview = repr(first_chunk[:120])
        print(f"    {baud:>10}: {total_bytes} bytes  preview: {preview!r}")
        return True
    else:
        print(f"    {baud:>10}: 0 bytes")
        return False


def main():
    print("=== Port detection ===")
    port = find_spad_port()
    if not port:
        return

    print(f"\nUsing: {port}")

    # Step 1: Try common baud rates WITHOUT config
    print("\n=== Baud rate scan (no config sent) ===")
    bauds = [2_250_000, 921_600, 460_800, 230_400, 115_200, 9600]
    found_baud = None
    for b in bauds:
        if try_baud(port, b, send_config=False):
            found_baud = b
            break

    # Step 2: Try with config sent
    print("\n=== Baud rate scan (config sent) ===")
    for b in bauds:
        if try_baud(port, b, send_config=True):
            found_baud = b
            break

    if found_baud:
        print(f"\n>>> Data received at {found_baud} baud!")
    else:
        print("\n>>> No data at any baud rate.")
        print("    Try: unplug USB, wait 5s, replug, then run again.")

    # Step 3: cc_hardware driver (only if data was seen)
    if found_baud == 2_250_000:
        print("\n=== cc_hardware driver test ===")
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        from cc_hardware.drivers.spads import SPADSensor
        from cc_hardware.drivers.spads.spad_wrappers import SPADMergeWrapperConfig
        from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHConfig4x4

        wrapped = VL53L8CHConfig4x4.create(port=port)
        config = SPADMergeWrapperConfig.create(wrapped=wrapped, data_type="HISTOGRAM")
        sensor = SPADSensor.create_from_config(config)
        print(f"  sensor.is_okay = {sensor.is_okay}")

        # Force-send config
        inner = sensor.unwrapped if hasattr(sensor, "unwrapped") else sensor
        if hasattr(inner, "_write_queue"):
            print("  Force-sending config...")
            inner._write_queue.put(inner.config.pack())
            time.sleep(2)

        print("  Calling accumulate() with 10s timeout...")
        import threading
        result = [None]
        def _acc():
            result[0] = sensor.accumulate()
        t = threading.Thread(target=_acc, daemon=True)
        t.start()
        t.join(timeout=10)
        if t.is_alive():
            print("  TIMEOUT: accumulate() blocked for 10s.")
        else:
            data = result[0]
            print(f"  SUCCESS: got {type(data).__name__}")
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"    {k}: shape={v.shape if hasattr(v, 'shape') else '?'}")
        sensor.close()


if __name__ == "__main__":
    main()
