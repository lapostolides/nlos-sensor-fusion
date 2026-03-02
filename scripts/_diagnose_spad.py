"""
_diagnose_spad.py - Read raw bytes from COM4 at 2.25 MB baud for 5 seconds
and print what the VL53L8CH firmware is actually sending.

Run this WITHOUT full_capture.py open (only one process can hold the port).
"""
import time
import serial

PORT    = "COM4"
BAUD    = 2_250_000
TIMEOUT = 5.0   # seconds to sniff

print(f"Opening {PORT} at {BAUD} baud ...")
try:
    s = serial.Serial(PORT, BAUD, timeout=0.5)
except Exception as e:
    print(f"ERROR: Could not open port: {e}")
    raise SystemExit(1)

print(f"Listening for {TIMEOUT}s — press Ctrl+C to stop early\n")
t0 = time.perf_counter()
lines_seen = 0
raw_bytes  = 0

try:
    while time.perf_counter() - t0 < TIMEOUT:
        line = s.readline()
        if line:
            raw_bytes += len(line)
            lines_seen += 1
            try:
                decoded = line.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                decoded = repr(line)
            print(f"[{lines_seen:4d}] {decoded}")
except KeyboardInterrupt:
    pass
finally:
    s.close()

elapsed = time.perf_counter() - t0
print(f"\n--- {lines_seen} lines / {raw_bytes} bytes in {elapsed:.1f}s ---")
if lines_seen == 0:
    print("No data received. Possible causes:")
    print("  1. Arduino firmware not running / needs re-flash")
    print("  2. Wrong baud rate on firmware side")
    print("  3. Wrong COM port")
    print("  4. Arduino needs a hardware reset (unplug/replug)")
