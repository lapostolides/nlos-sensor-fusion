"""
Ground truth person detection from RGB camera frames using YOLO.

Run as a script to process a PKL log and save detections to a sidecar JSON:

    python ground_truth.py [path/to/capture.pkl] [--model yolov8n.pt] [--conf 0.3]

The sidecar file is written next to the PKL as <stem>_gt.json and loaded
automatically by playback.py when --gt is passed.
"""

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from ultralytics import YOLO

# COCO class ID for person
PERSON_CLASS_ID = 0

# Camera key used to extract ground truth frames from PKL log records.
# The overhead_cam (eMeet C960) provides a top-down view of the scene,
# making it the primary ground truth camera for person localisation.
GT_CAMERA_KEY = "overhead_cam"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class PersonLocation:
    """Location of a detected person in the frame."""

    bbox: tuple[float, float, float, float]  # (x, y, w, h) pixels
    center: tuple[float, float]              # (cx, cy) pixels
    confidence: float                        # detection confidence [0, 1]
    track_id: Optional[int]                  # persistent ID (None if untracked)

    def to_dict(self) -> dict:
        return {
            "bbox":       list(self.bbox),
            "center":     list(self.center),
            "confidence": round(self.confidence, 4),
            "track_id":   self.track_id,
        }

    @staticmethod
    def from_dict(d: dict) -> "PersonLocation":
        return PersonLocation(
            bbox=tuple(d["bbox"]),          # type: ignore[arg-type]
            center=tuple(d["center"]),      # type: ignore[arg-type]
            confidence=d["confidence"],
            track_id=d["track_id"],
        )


@dataclass
class LogDetection:
    """Detection results for one iteration in a PKL log file."""

    iter: int
    locations: List[PersonLocation]
    frame: np.ndarray  # the BGR frame used for detection


# ── Detector ──────────────────────────────────────────────────────────────────

class GroundTruthDetector:
    """Detects one or more people in RGB camera frames using YOLO."""

    def __init__(self, model: str = "yolov8n.pt", confidence_threshold: float = 0.3):
        self.model = YOLO(model)
        self.confidence_threshold = confidence_threshold

    def process(
        self,
        frame: np.ndarray,
        normalized: bool = False,
    ) -> List[PersonLocation]:
        """
        Detect people in *frame* and return their locations.

        Args:
            frame: BGR image (H, W, 3).
            normalized: If True coordinates are in [0, 1]; otherwise pixels.
        """
        results = self.model.track(frame, verbose=False, persist=True)[0]
        height, width = frame.shape[:2]
        locations: List[PersonLocation] = []

        for box in results.boxes:
            if int(box.cls) != PERSON_CLASS_ID:
                continue
            if float(box.conf) < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x, y = float(x1), float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)
            cx, cy = x + w / 2, y + h / 2

            if normalized:
                x, w, cx = x / width, w / width, cx / width
                y, h, cy = y / height, h / height, cy / height

            locations.append(PersonLocation(
                bbox=(x, y, w, h),
                center=(cx, cy),
                confidence=float(box.conf),
                track_id=int(box.id.cpu().item()) if box.id is not None else None,
            ))

        return locations


# ── PKL iteration ─────────────────────────────────────────────────────────────

def iter_log(
    pkl_path: Path | str,
    detector: GroundTruthDetector,
    camera_key: str = GT_CAMERA_KEY,
    normalized: bool = False,
) -> Iterator[LogDetection]:
    """
    Iterate over records in a PKL log file and yield person detections.

    Skips records without a camera frame for *camera_key*.
    """
    with open(Path(pkl_path), "rb") as f:
        while True:
            try:
                record = pickle.load(f)
            except EOFError:
                break
            if "metadata" in record or "iter" not in record:
                continue
            cam_data = record.get(camera_key)
            if cam_data is None:
                continue
            frame = cam_data["raw_rgb"]
            yield LogDetection(
                iter=record["iter"],
                locations=detector.process(frame, normalized=normalized),
                frame=frame,
            )


# ── Sidecar JSON I/O ──────────────────────────────────────────────────────────

def gt_path(pkl_path: Path) -> Path:
    """Return the sidecar GT JSON path for *pkl_path*."""
    return pkl_path.with_name(pkl_path.stem + "_gt.json")


def save_gt(
    pkl_path: Path,
    detections: dict[int, List[PersonLocation]],
    model: str,
    confidence: float,
) -> Path:
    """
    Write detections to a sidecar JSON next to *pkl_path*.

    *detections* maps iter → list[PersonLocation] (pixel coords).
    Returns the path written.
    """
    out = gt_path(pkl_path)
    payload = {
        "source_pkl":          str(pkl_path.resolve()),
        "model":               model,
        "confidence_threshold": confidence,
        "camera_key":          GT_CAMERA_KEY,
        "created":             datetime.now().isoformat(),
        "detections": {
            str(it): [loc.to_dict() for loc in locs]
            for it, locs in detections.items()
        },
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def load_gt(pkl_path: Path) -> dict[int, List[PersonLocation]] | None:
    """
    Load the sidecar GT JSON for *pkl_path*.

    Returns a dict mapping iter → list[PersonLocation], or None if the
    sidecar does not exist.
    """
    p = gt_path(pkl_path)
    if not p.exists():
        return None
    with open(p) as f:
        payload = json.load(f)
    return {
        int(it): [PersonLocation.from_dict(d) for d in locs]
        for it, locs in payload["detections"].items()
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _run(pkl_path: Path, model: str, confidence: float) -> None:
    print(f"File  : {pkl_path}")

    existing = gt_path(pkl_path)
    if existing.exists():
        print(f"Overwriting existing GT file: {existing}")

    detector = GroundTruthDetector(model=model, confidence_threshold=confidence)

    detections: dict[int, List[PersonLocation]] = {}
    n_frames = n_persons = 0

    for det in iter_log(pkl_path, detector, normalized=False):
        detections[det.iter] = det.locations
        n_frames += 1
        n_persons += len(det.locations)
        if n_frames % 50 == 0:
            print(f"  {n_frames} frames processed ...", end="\r")

    print(f"  {n_frames} frames processed.       ")
    out = save_gt(pkl_path, detections, model, confidence)
    print(f"Saved : {out}")
    print(f"Frames with overhead_cam : {n_frames}")
    print(f"Total detections         : {n_persons}")
    print(f"Frames with a person     : {sum(1 for v in detections.values() if v)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLO ground-truth detection on a capture PKL and save results."
    )
    parser.add_argument("pkl", nargs="?", help="Path to capture PKL file.")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO model (default: yolov8n.pt).")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3).")
    args = parser.parse_args()

    if args.pkl:
        path = Path(args.pkl)
        if not path.exists():
            print(f"Error: {path} does not exist.")
            sys.exit(1)
    else:
        logs = sorted(Path("data/logs").rglob("*.pkl"), key=lambda p: p.stat().st_mtime)
        if not logs:
            print("No PKL file found in data/logs/.")
            sys.exit(1)
        path = logs[-1]
        print(f"Auto-detected: {path}")

    _run(path, model=args.model, confidence=args.conf)


if __name__ == "__main__":
    main()
