"""Quick smoke-test: run GroundTruthDetector over the first 10 frames."""
from pathlib import Path
from ground_truth import GroundTruthDetector, iter_log

LOG = sorted(Path("data/logs").rglob("*.pkl"), key=lambda p: p.stat().st_mtime)[-1]
print(f"Log: {LOG}\n")

detector = GroundTruthDetector(model="yolov8n.pt", confidence_threshold=0.3)

for i, det in enumerate(iter_log(LOG, detector, normalized=True)):
    print(f"iter {det.iter:4d} | frame {det.frame.shape} | "
          f"{len(det.locations)} person(s) detected")
    for loc in det.locations:
        cx, cy = loc.center
        print(f"          track_id={loc.track_id}  conf={loc.confidence:.2f}"
              f"  center=({cx:.3f}, {cy:.3f})")
    if i >= 9:
        print("... (stopping after 10 frames)")
        break
