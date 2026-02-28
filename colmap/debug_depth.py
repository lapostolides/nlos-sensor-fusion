"""Decode depth map headers and check what stereo_fusion sees."""
import numpy as np, os, struct
from pathlib import Path

dm_dir = Path("data/testlaserroom/colmap_output/dense/stereo/depth_maps")
files = sorted(dm_dir.glob("*.photometric.bin"))
print(f"Found {len(files)} photometric maps")

for fn in files[:3]:
    data = fn.read_bytes()
    # Find all '&' (0x26 = 38) positions in the header
    amp = [i for i, b in enumerate(data[:100]) if b == 38]
    print(f"\n{fn.name}")
    print(f"  Header bytes: {data[:amp[2]+1]}")
    header_str = data[:amp[2]+1].decode('ascii', errors='replace')
    parts = header_str.split('&')
    W, H, C = int(parts[0]), int(parts[1]), int(parts[2])
    print(f"  W={W}  H={H}  C={C}")
    payload_start = amp[2]+1
    payload_bytes = len(data) - payload_start
    expected_bytes = W * H * C * 4
    print(f"  Payload: {payload_bytes} bytes (expected {expected_bytes} for {W}x{H}x{C} float32)")
    print(f"  File size: {fn.stat().st_size/1e6:.2f} MB")

    # Read just the depth channel (first W*H float32 values)
    d = np.frombuffer(data[payload_start:payload_start + W*H*4], dtype=np.float32)
    valid = d[d > 0]
    invalid = (d == 0).sum()
    print(f"  depth channel: valid={len(valid)}/{len(d)} ({100*len(valid)/len(d):.1f}%)")
    if len(valid):
        print(f"  depth range: [{valid.min():.3f}, {valid.max():.3f}]  median={np.median(valid):.3f}")
    print(f"  zero/invalid pixels: {invalid}")

    # If C > 1, show what the other channels look like
    if C > 1:
        for ch in range(1, C):
            start = payload_start + ch * W * H * 4
            ch_data = np.frombuffer(data[start:start + W*H*4], dtype=np.float32)
            print(f"  channel {ch}: min={ch_data.min():.3f}, max={ch_data.max():.3f}, mean={ch_data.mean():.3f}")

# Check workspace config file for filter flag
cfg_path = Path("data/testlaserroom/colmap_output/dense/stereo/patch-match.cfg")
print(f"\npatch-match.cfg first 10 lines:")
for line in cfg_path.read_text().splitlines()[:10]:
    print(f"  {line}")

# Check if there's a workspace options file
for p in Path("data/testlaserroom/colmap_output/dense").rglob("*.cfg"):
    print(f"\nConfig file: {p}")
    print(p.read_text()[:300])
