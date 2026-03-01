"""
Ground truth person detection from RGB camera frames using YOLO.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from ultralytics import YOLO

# COCO class ID for person
PERSON_CLASS_ID = 0

# Camera key used to extract ground truth frames from PKL log records.
# Currently points to the tracking RealSense (the scene-facing camera).
# Once the dedicated ground truth camera is added to the capture pipeline,
# change this to "ground_truth_realsense_data".
GT_CAMERA_KEY = "tracking_realsense_data"


@dataclass
class PersonLocation:
    """Location of a detected person in the frame."""

    bbox: tuple[float, float, float, float]  # (x, y, w, h)
    center: tuple[float, float]  # (cx, cy)
    confidence: float  # detection confidence [0, 1]
    track_id: Optional[int]  # persistent ID across frames (None if untracked)


class GroundTruthDetector:
    """
    Detects one or more people in RGB camera frames using YOLO.
    """

    def __init__(self, model: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Args:
            model: YOLO model path or name (e.g. yolov8n.pt, yolov8s.pt).
            confidence_threshold: Minimum confidence for person detections (0–1).
        """
        self.model = YOLO(model)
        self.confidence_threshold = confidence_threshold

    def process(
        self,
        frame: np.ndarray,
        normalized: bool = False,
    ) -> List[PersonLocation]:
        """
        Detect people in a frame and return their locations.

        Args:
            frame: RGB or BGR image as numpy array (H, W, C).
            normalized: If True, return coordinates in [0, 1] range.
                       If False, return pixel coordinates.

        Returns:
            List of PersonLocation, each with bbox, center, confidence, and track_id.
        """
        results = self.model.track(frame, verbose=False, persist=True)[0]

        height, width = frame.shape[:2]

        # locations of detected people in the frame
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
            cx = x + w / 2
            cy = y + h / 2

            if normalized:
                x, w, cx = x / width, w / width, cx / width
                y, h, cy = y / height, h / height, cy / height

            confidence = float(box.conf)
            track_id = (
                int(box.id.cpu().item()) if box.id is not None else None
            )

            locations.append(
                PersonLocation(
                    bbox=(x, y, w, h),
                    center=(cx, cy),
                    confidence=confidence,
                    track_id=track_id,
                )
            )

        return locations


@dataclass
class LogDetection:
    """Detection results for one iteration in a PKL log file."""

    iter: int
    locations: List[PersonLocation]
    frame: np.ndarray  # the BGR frame used for detection


def iter_log(
    pkl_path: Path | str,
    detector: GroundTruthDetector,
    camera_key: str = GT_CAMERA_KEY,
    normalized: bool = False,
) -> Iterator[LogDetection]:
    """
    Iterate over records in a PKL log file and yield person detections.

    Skips the metadata record (first entry). Each subsequent record contains
    SPAD data plus one or more camera frames; detection runs on the frame
    identified by ``camera_key``.

    Args:
        pkl_path: Path to the .pkl log file produced by a capture script.
        detector: GroundTruthDetector instance to use for detection.
        camera_key: Key in each record for the camera data to detect from.
                    Defaults to GT_CAMERA_KEY. Update to
                    "ground_truth_realsense_data" once the new camera is
                    integrated into the capture pipeline.
        normalized: If True, return coordinates in [0, 1] range.

    Yields:
        LogDetection with the iteration index, detected locations, and frame.
    """
    with open(Path(pkl_path), "rb") as f:
        while True:
            try:
                record = pickle.load(f)
            except EOFError:
                break
            if "metadata" in record:
                continue
            frame = record[camera_key]["aligned_rgb_image"]
            yield LogDetection(
                iter=record["iter"],
                locations=detector.process(frame, normalized=normalized),
                frame=frame,
            )
