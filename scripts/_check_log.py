"""
_check_log.py  — inspect the most recent PKL log.
Confirms overhead_cam and sensor_cam frames are both present and distinct.
"""
import pickle, sys
import numpy as np
from pathlib import Path

LOG = Path("data/logs/2026-03-01/18-42-12/test_data.pkl")

records = []
with open(LOG, "rb") as f:
    while True:
        try:
            records.append(pickle.load(f))
        except EOFError:
            break

print(f"File : {LOG}")
print(f"Total records : {len(records)}\n")

if not records:
    print("Empty file.")
    sys.exit(1)

# Header record
header = records[0]
print("=== Header keys ===")
for k, v in header.items():
    print(f"  {k}: {v}")

# Find first data record that has both cameras
data_records = [r for r in records[1:] if isinstance(r, dict)]
print(f"\nData records  : {len(data_records)}")

# What keys do data records have?
all_keys = set()
for r in data_records:
    all_keys.update(r.keys())
print(f"Keys seen     : {sorted(all_keys)}\n")

# Count presence
has_overhead = sum(1 for r in data_records if "overhead_cam" in r)
has_sensor   = sum(1 for r in data_records if "sensor_cam"   in r)
has_spad     = sum(1 for r in data_records if "spad"         in r)
print(f"Records with overhead_cam : {has_overhead}/{len(data_records)}")
print(f"Records with sensor_cam   : {has_sensor}/{len(data_records)}")
print(f"Records with spad         : {has_spad}/{len(data_records)}")

# Find first record with both cameras
both = [r for r in data_records if "overhead_cam" in r and "sensor_cam" in r]
if not both:
    print("\nNo record has both cameras simultaneously.")
    sys.exit(1)

rec = both[0]
ov_rgb = rec["overhead_cam"]["raw_rgb"]    # HxWx3
sc_rgb = rec["sensor_cam"]["raw_rgb"]      # HxWx3

print(f"\n=== First record with both cameras (iter={rec.get('iter','?')}) ===")
print(f"  overhead_cam shape : {ov_rgb.shape}  dtype={ov_rgb.dtype}")
print(f"  sensor_cam shape   : {sc_rgb.shape}  dtype={sc_rgb.dtype}")

# Are they pixel-identical? (would mean same camera fed both slots)
if ov_rgb.shape == sc_rgb.shape:
    identical = np.array_equal(ov_rgb, sc_rgb)
    print(f"  Pixel-identical    : {identical}")
    if not identical:
        diff = np.abs(ov_rgb.astype(int) - sc_rgb.astype(int))
        print(f"  Mean abs diff      : {diff.mean():.1f}  max={diff.max()}")
else:
    print(f"  Different shapes → definitely different cameras")

# Mean brightness as a quick sanity check
print(f"  overhead_cam mean brightness : {ov_rgb.mean():.1f}")
print(f"  sensor_cam   mean brightness : {sc_rgb.mean():.1f}")

# Save a frame from each for visual inspection
import cv2
cv2.imwrite("_check_overhead.jpg", ov_rgb)
cv2.imwrite("_check_sensorcam.jpg", sc_rgb)
print("\nSaved _check_overhead.jpg and _check_sensorcam.jpg for visual inspection.")
