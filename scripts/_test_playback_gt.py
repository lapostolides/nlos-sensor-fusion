"""Smoke-test: pre-compute GT detections on first 50 records."""
import playback
from pathlib import Path

path = sorted(Path("data/logs").rglob("*.pkl"), key=lambda p: p.stat().st_mtime)[-1]
records = playback.load_records(path)
print(f"Loaded {len(records)} records from {path}")

dets = playback.precompute_detections(records[:50], "yolov8n.pt", 0.3)
with_person = [(i, d) for i, d in enumerate(dets) if d]
print(f"Frames with detections: {len(with_person)}/50")
for i, locs in with_person[:3]:
    for loc in locs:
        print(f"  frame {i}: ID={loc.track_id} conf={loc.confidence:.2f} "
              f"center=({loc.center[0]:.0f}, {loc.center[1]:.0f}) px")
