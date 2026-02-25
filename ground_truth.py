"""
Ground truth person detection from RGB camera frames using YOLO.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

# COCO class ID for person
PERSON_CLASS_ID = 0


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
