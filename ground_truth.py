"""
Ground truth person detection from RGB camera frames using YOLO.

Run as a script to process a capture run and save detections:

    python ground_truth.py data/logs/my-run/               # new format
    python ground_truth.py data/logs/my-run/data.pkl        # legacy PKL
    python ground_truth.py --model yolov8n.pt --conf 0.3

The GT file is written as gt.json (run dir) or <stem>_gt.json (PKL) and
loaded automatically by playback.py when --gt is passed.
"""

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import torch
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
    pose: Optional[str] = field(default=None)       # Pose enum value string, or None
    pose_conf: float    = field(default=0.0)        # pose classifier confidence [0, 1]

    def to_dict(self) -> dict:
        d: dict = {
            "bbox":       list(self.bbox),
            "center":     list(self.center),
            "confidence": round(self.confidence, 4),
            "track_id":   self.track_id,
        }
        if self.pose is not None:
            d["pose"]      = self.pose
            d["pose_conf"] = round(self.pose_conf, 4)
        return d

    @staticmethod
    def from_dict(d: dict) -> "PersonLocation":
        return PersonLocation(
            bbox=tuple(d["bbox"]),          # type: ignore[arg-type]
            center=tuple(d["center"]),      # type: ignore[arg-type]
            confidence=d["confidence"],
            track_id=d["track_id"],
            pose=d.get("pose"),
            pose_conf=d.get("pose_conf", 0.0),
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

    def __init__(
        self,
        model: str = "models/yolov8n.pt",
        confidence_threshold: float = 0.3,
        device: Optional[str] = None,
    ):
        self.model = YOLO(model)
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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
        results = self.model.track(frame, verbose=False, persist=True, device=self.device)[0]
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


# ── Frame iteration ──────────────────────────────────────────────────────────

def _iter_pkl(pkl_path: Path, camera_key: str) -> Iterator[tuple[int, "np.ndarray"]]:
    """Yield (iter, frame) from a legacy PKL file."""
    with open(pkl_path, "rb") as f:
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
            yield record["iter"], cam_data["raw_rgb"]


def _iter_run_dir(run_dir: Path, camera_key: str) -> Iterator[tuple[int, "np.ndarray"]]:
    """Yield (iter, frame) from a new-format run directory."""
    import cv2
    if camera_key == "overhead_cam":
        img_dir = run_dir / "overhead_cam"
    else:
        img_dir = run_dir / camera_key / "rgb"
    if not img_dir.exists():
        return
    for i, img_path in enumerate(sorted(img_dir.glob("*.jpg"))):
        frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame is not None:
            yield i, frame


def iter_log(
    path: Path | str,
    detector: GroundTruthDetector,
    classifier=None,
    camera_key: str = GT_CAMERA_KEY,
    normalized: bool = False,
) -> Iterator[LogDetection]:
    """
    Iterate over frames from a run directory or legacy PKL file and yield
    person detections.
    """
    path = Path(path)
    if path.is_dir() and (path / "manifest.json").exists():
        frame_iter = _iter_run_dir(path, camera_key)
    else:
        frame_iter = _iter_pkl(path, camera_key)

    for iter_idx, frame in frame_iter:
        locs = detector.process(frame, normalized=False)

        if classifier is not None and locs:
            poses = classifier.classify_frame(frame, locs)
            for loc, p in zip(locs, poses):
                loc.pose      = p.pose.value
                loc.pose_conf = p.confidence

        if normalized:
            h, w = frame.shape[:2]
            for loc in locs:
                bx, by, bw, bh = loc.bbox
                cx, cy = loc.center
                loc.bbox   = (bx / w, by / h, bw / w, bh / h)
                loc.center = (cx / w, cy / h)

        yield LogDetection(iter=iter_idx, locations=locs, frame=frame)


# ── Sidecar JSON I/O ──────────────────────────────────────────────────────────

def gt_path(path: Path) -> Path:
    """Return the GT JSON path for a run dir or legacy PKL."""
    if path.is_dir():
        return path / "gt.json"
    return path.with_name(path.stem + "_gt.json")


def save_gt(
    path: Path,
    detections: dict[int, List[PersonLocation]],
    model: str,
    confidence: float,
) -> Path:
    """
    Write detections to a GT JSON alongside *path* (run dir or PKL).

    *detections* maps iter -> list[PersonLocation] (pixel coords).
    Returns the path written.
    """
    out = gt_path(path)
    payload = {
        "source":              str(path.resolve()),
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


def load_gt(path: Path) -> dict[int, List[PersonLocation]] | None:
    """
    Load the GT JSON for *path* (run dir or legacy PKL).

    Returns a dict mapping iter -> list[PersonLocation], or None if the
    file does not exist.
    """
    p = gt_path(path)
    if not p.exists():
        return None
    with open(p) as f:
        payload = json.load(f)
    return {
        int(it): [PersonLocation.from_dict(d) for d in locs]
        for it, locs in payload["detections"].items()
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _run(
    path:        Path,
    model:       str,
    confidence:  float,
    pose_model:  Optional[str] = None,
    device:      Optional[str] = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Input  : {path}")
    print(f"Device : {device}")

    existing = gt_path(path)
    if existing.exists():
        print(f"Overwriting existing GT file: {existing}")

    detector   = GroundTruthDetector(model=model, confidence_threshold=confidence,
                                     device=device)
    classifier = None
    if pose_model is not None:
        from pose_estimation import PoseClassifier  # type: ignore[import]
        classifier = PoseClassifier(model=pose_model, device=device)
        print(f"Pose   : {pose_model}")

    detections: dict[int, List[PersonLocation]] = {}
    n_frames = n_persons = 0

    for det in iter_log(path, detector, classifier=classifier, normalized=False):
        detections[det.iter] = det.locations
        n_frames += 1
        n_persons += len(det.locations)
        if n_frames % 50 == 0:
            print(f"  {n_frames} frames processed ...", end="\r")

    print(f"  {n_frames} frames processed.       ")
    out = save_gt(path, detections, model, confidence)
    print(f"Saved : {out}")
    print(f"Frames with overhead_cam : {n_frames}")
    print(f"Total detections         : {n_persons}")
    print(f"Frames with a person     : {sum(1 for v in detections.values() if v)}")
    if pose_model is not None:
        pose_counts: dict[str, int] = {}
        for locs in detections.values():
            for loc in locs:
                if loc.pose:
                    pose_counts[loc.pose] = pose_counts.get(loc.pose, 0) + 1
        print(f"Pose breakdown           : {pose_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLO ground-truth detection on a capture PKL and save results."
    )
    parser.add_argument(
        "path", nargs="?",
        help="Run directory (with manifest.json) or legacy .pkl file.",
    )
    parser.add_argument("--model", default="models/yolov8n.pt",
                        help="YOLO model (default: yolov8n.pt).")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3).")
    parser.add_argument("--pose", dest="pose_model", nargs="?",
                        const="models/yolo11n-pose.pt", default=None,
                        metavar="MODEL",
                        help=(
                            "Enable pose classification.  Optionally provide a "
                            "model name (default: yolo11n-pose.pt).  "
                            "Example: --pose  or  --pose yolov8n-pose.pt"
                        ))
    parser.add_argument("--device", default=None,
                        help=(
                            "Compute device for YOLO inference "
                            "(default: cuda if available, else cpu).  "
                            "Examples: cuda, cpu, cuda:0"
                        ))
    args = parser.parse_args()

    if args.path:
        path = Path(args.path)
        if not path.exists():
            print(f"Error: {path} does not exist.")
            sys.exit(1)
    else:
        # Auto-detect: most recent run across both formats
        candidates: list[Path] = []
        for mf in Path("data/logs").rglob("manifest.json"):
            candidates.append(mf.parent)
        for pk in Path("data/logs").rglob("*.pkl"):
            candidates.append(pk)
        if not candidates:
            print("No runs or PKL files found in data/logs/.")
            sys.exit(1)
        path = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"Auto-detected: {path}")

    _run(path, model=args.model, confidence=args.conf,
         pose_model=args.pose_model, device=args.device)


if __name__ == "__main__":
    main()
