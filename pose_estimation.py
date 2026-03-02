"""
Coarse human pose classification from an elevated wall-mounted camera.

Uses YOLO11-pose (Ultralytics) for 17-point COCO keypoint detection and applies
geometric classifiers to produce one of four coarse pose labels per tracked person:

    STILL       — person is stationary
    LOCOMOTION  — walking or running (merged; not separable in a single frame)
    CROUCHING   — crouching / squatting
    ARMS_RAISED — at least one arm raised clearly above shoulder level
    UNKNOWN     — insufficient keypoint confidence to classify

Used by ground_truth.py to label pose alongside YOLO person detections.
These labels serve as ground-truth annotations for training the NLOS
localization model.

Camera assumption
-----------------
The camera is wall-mounted and elevated (not nadir).  It provides a natural
perspective view of the scene — standard COCO-trained YOLO-pose models work
out-of-the-box without NToP fine-tuning.

Thresholds
----------
All classification thresholds are exposed as constructor arguments and can
be tuned for a specific camera height / angle / focal length.  The defaults
target a camera at roughly 1.5–2.5 m height, 30–60° depression angle.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]

if TYPE_CHECKING:
    from ground_truth import PersonLocation  # type: ignore[import]


# ── COCO 17-keypoint index map ─────────────────────────────────────────────────

_KP = {
    "nose":            0,
    "left_eye":        1,  "right_eye":       2,
    "left_ear":        3,  "right_ear":       4,
    "left_shoulder":   5,  "right_shoulder":  6,
    "left_elbow":      7,  "right_elbow":     8,
    "left_wrist":      9,  "right_wrist":     10,
    "left_hip":        11, "right_hip":       12,
    "left_knee":       13, "right_knee":      14,
    "left_ankle":      15, "right_ankle":     16,
}


# ── Data types ────────────────────────────────────────────────────────────────

class Pose(str, Enum):
    STILL       = "still"
    LOCOMOTION  = "locomotion"  # walking / running merged
    CROUCHING   = "crouching"
    ARMS_RAISED = "arms_raised"
    UNKNOWN     = "unknown"


@dataclass
class PersonPose:
    """Coarse pose classification result for one tracked person."""

    track_id:   Optional[int]
    pose:       Pose
    confidence: float   # classifier confidence in [0, 1]

    def to_dict(self) -> dict:
        return {
            "track_id":   self.track_id,
            "pose":       self.pose.value,
            "confidence": round(self.confidence, 4),
        }

    @staticmethod
    def from_dict(d: dict) -> "PersonPose":
        return PersonPose(
            track_id=d.get("track_id"),
            pose=Pose(d.get("pose", "unknown")),
            confidence=d.get("confidence", 0.0),
        )


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _get_kp(
    kps: np.ndarray,
    idx: int,
    min_conf: float = 0.3,
) -> Optional[tuple[float, float]]:
    """Return (x, y) for keypoint *idx*, or None if confidence is too low."""
    x, y, c = float(kps[idx, 0]), float(kps[idx, 1]), float(kps[idx, 2])
    return (x, y) if c >= min_conf else None


def _iou(b1: tuple, b2: tuple) -> float:
    """Intersection-over-union for two (x, y, w, h) bounding boxes."""
    ax1, ay1 = b1[0],          b1[1]
    ax2, ay2 = b1[0] + b1[2],  b1[1] + b1[3]
    bx1, by1 = b2[0],          b2[1]
    bx2, by2 = b2[0] + b2[2],  b2[1] + b2[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def _crouch_score(kps: np.ndarray, min_conf: float = 0.3) -> float:
    """
    Returns a score in [0, 1] where 1 = clearly crouching.

    Primary signal: torso compactness (torso_height / shoulder_width).
    - Standing: ratio ≈ 0.9–1.8 → score near 0
    - Half-crouch: ratio ≈ 0.5–0.8 → score near 0.5–1

    Booster: hip-to-knee vertical distance, when both knee keypoints are
    visible.  A crouching person has bent knees close to their hips.

    Both metrics are normalised by shoulder width to be camera-scale
    invariant.
    """
    ls = _get_kp(kps, _KP["left_shoulder"],  min_conf)
    rs = _get_kp(kps, _KP["right_shoulder"], min_conf)
    lh = _get_kp(kps, _KP["left_hip"],       min_conf)
    rh = _get_kp(kps, _KP["right_hip"],      min_conf)

    if not (ls and rs and lh and rh):
        return 0.0

    sw = abs(rs[0] - ls[0])  # shoulder width in pixels
    if sw < 4.0:              # degenerate (person seen perfectly side-on)
        return 0.0

    shoulder_y = (ls[1] + rs[1]) / 2.0
    hip_y      = (lh[1] + rh[1]) / 2.0
    torso_h    = max(0.0, hip_y - shoulder_y)  # positive: hips below shoulders
    ratio      = torso_h / sw

    # Linear map: ratio in [0.35, 1.05] → score in [1.0, 0.0]
    score = 1.0 - float(np.clip((ratio - 0.35) / 0.70, 0.0, 1.0))

    # Boost: knee height relative to hip.
    # Standing: (knee_y - hip_y)/sw ≈ 0.9–1.3
    # Crouching: (knee_y - hip_y)/sw ≈ 0.1–0.5
    lk = _get_kp(kps, _KP["left_knee"],  min_conf)
    rk = _get_kp(kps, _KP["right_knee"], min_conf)
    if lk and rk:
        knee_y    = (lk[1] + rk[1]) / 2.0
        hk_ratio  = max(0.0, knee_y - hip_y) / sw
        knee_score = 1.0 - float(np.clip((hk_ratio - 0.25) / 0.70, 0.0, 1.0))
        score = max(score, knee_score)

    return float(np.clip(score, 0.0, 1.0))


def _arms_raised_score(kps: np.ndarray, min_conf: float = 0.3) -> float:
    """
    Returns a score in [0, 1] where 1 = arm(s) clearly raised above the head.

    From an elevated wall camera, arms raised toward the ceiling appear at
    *lower* image-Y values (closer to the top of the frame) than the shoulders.
    The raise amount is normalised by shoulder width.

    Score = max raise across both wrists, clipped at 1.0:
        raise_norm = (shoulder_y - wrist_y) / shoulder_width
    A raise_norm > 0  means the wrist is above the shoulder line.
    A raise_norm ≈ 1  means the wrist is one shoulder-width above shoulders
    (clearly overhead).
    """
    ls = _get_kp(kps, _KP["left_shoulder"],  min_conf)
    rs = _get_kp(kps, _KP["right_shoulder"], min_conf)
    lw = _get_kp(kps, _KP["left_wrist"],     min_conf)
    rw = _get_kp(kps, _KP["right_wrist"],    min_conf)

    if not (ls or rs):
        return 0.0

    # Shoulder reference
    valid_s = [p for p in [ls, rs] if p is not None]
    shoulder_y = sum(p[1] for p in valid_s) / len(valid_s)
    sw = abs(rs[0] - ls[0]) if (ls and rs) else 1.0

    max_raise = 0.0
    for wrist in [lw, rw]:
        if wrist is None:
            continue
        raise_norm = (shoulder_y - wrist[1]) / max(sw, 1.0)
        max_raise = max(max_raise, raise_norm)

    return float(np.clip(max_raise, 0.0, 1.0))


# ── Classifier ────────────────────────────────────────────────────────────────

class PoseClassifier:
    """
    Classifies each tracked person's coarse pose from an elevated wall camera.

    Workflow per frame
    ------------------
    1. Run YOLO11-pose on the frame to obtain (bbox, 17-keypoint) detections.
    2. Match each PersonLocation (from GroundTruthDetector) to the nearest
       pose detection by bounding-box IoU.
    3. Apply geometric classifiers (priority order):
           CROUCHING   — torso compactness / knee-hip distance
           ARMS_RAISED — wrist above shoulder line
           STILL       — mean centre displacement < threshold  (temporal)
           LOCOMOTION  — mean centre displacement ≥ threshold  (temporal)
           UNKNOWN     — fallback when keypoints are insufficient

    The temporal STILL/LOCOMOTION decision uses a per-track ring-buffer of
    recent frame centres.  This works correctly for the post-hoc PKL labelling
    workflow where frames are processed in capture order.

    Parameters
    ----------
    model : str
        Ultralytics model identifier.  ``yolo11n-pose.pt`` (default) auto-
        downloads on first use.  ``yolov8n-pose.pt`` is also supported.
    crouch_threshold : float
        Crouch score above which a person is labelled CROUCHING.  Default 0.55.
    arms_threshold : float
        Arms-raised score above which a person is labelled ARMS_RAISED.
        Default 0.45.
    still_disp_px : float
        Mean per-frame centre displacement (pixels) below which a person is
        labelled STILL.  Tune upward for distant cameras, downward for close
        cameras.  Default 8.0.
    history_len : int
        Number of recent frames used for the STILL/LOCOMOTION decision.
        Default 10.
    kp_conf : float
        Minimum YOLO keypoint confidence accepted.  Default 0.3.
    iou_thresh : float
        Minimum IoU required to match a pose detection to a PersonLocation.
        Default 0.25.
    device : str or None
        Compute device for YOLO inference (e.g. ``"cuda"``, ``"cpu"``).
        Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        model:            str            = "models/yolo11n-pose.pt",
        crouch_threshold: float          = 0.55,
        arms_threshold:   float          = 0.45,
        still_disp_px:    float          = 8.0,
        history_len:      int            = 10,
        kp_conf:          float          = 0.3,
        iou_thresh:       float          = 0.25,
        device:           Optional[str]  = None,
    ) -> None:
        self._model             = YOLO(model)
        self.crouch_threshold   = crouch_threshold
        self.arms_threshold     = arms_threshold
        self.still_disp_px      = still_disp_px
        self._history_len       = history_len
        self._kp_conf           = kp_conf
        self._iou_thresh        = iou_thresh
        self._device            = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._centers: dict[int, deque] = {}  # track_id → deque[(cx, cy)]

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear per-track centre history.  Call between separate recordings."""
        self._centers.clear()

    def classify_frame(
        self,
        frame:     np.ndarray,
        locations: list["PersonLocation"],
    ) -> list[PersonPose]:
        """
        Classify pose for every person in *locations*.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3) — the same frame passed to GroundTruthDetector.
        locations : list[PersonLocation]
            Detections from GroundTruthDetector (pixel coordinates).

        Returns
        -------
        list[PersonPose]
            One PersonPose per entry in *locations*, in the same order.
        """
        # Run YOLO-Pose on the full frame
        result = self._model(frame, verbose=False, device=self._device)[0]
        detections = self._extract_detections(result)

        return [self._classify_one(loc, detections) for loc in locations]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_detections(self, result) -> list[dict]:
        """Parse YOLO result into a list of {bbox, kps} dicts."""
        dets = []
        if result.keypoints is None or len(result.boxes) == 0:
            return dets
        for box, kps in zip(result.boxes, result.keypoints.data):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            dets.append({
                "bbox": (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                "kps":  kps.cpu().numpy(),  # (17, 3)  x, y, conf
            })
        return dets

    def _classify_one(
        self,
        loc:        "PersonLocation",
        detections: list[dict],
    ) -> PersonPose:
        track_id = loc.track_id

        # Update centre history
        if track_id is not None:
            if track_id not in self._centers:
                self._centers[track_id] = deque(maxlen=self._history_len)
            self._centers[track_id].append(loc.center)

        # Match to best YOLO-Pose detection by IoU
        best_kps, best_iou = None, self._iou_thresh
        for det in detections:
            iou = _iou(loc.bbox, det["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_kps = det["kps"]

        if best_kps is None:
            # No keypoint match — fall back to motion only
            return self._motion_class(track_id, confidence_scale=0.5)

        # ── Geometry-based priority ────────────────────────────────────────

        c_score = _crouch_score(best_kps, self._kp_conf)
        if c_score >= self.crouch_threshold:
            return PersonPose(track_id=track_id, pose=Pose.CROUCHING,
                              confidence=c_score)

        a_score = _arms_raised_score(best_kps, self._kp_conf)
        if a_score >= self.arms_threshold:
            return PersonPose(track_id=track_id, pose=Pose.ARMS_RAISED,
                              confidence=a_score)

        return self._motion_class(track_id, confidence_scale=1.0)

    def _motion_class(
        self,
        track_id:         Optional[int],
        confidence_scale: float = 1.0,
    ) -> PersonPose:
        """Classify STILL vs LOCOMOTION from centre displacement history."""
        disp = self._mean_displacement(track_id)
        if disp is None:
            return PersonPose(track_id=track_id, pose=Pose.UNKNOWN,
                              confidence=0.0)

        if disp < self.still_disp_px:
            conf = (1.0 - disp / self.still_disp_px) * confidence_scale
            return PersonPose(track_id=track_id, pose=Pose.STILL,
                              confidence=float(np.clip(conf, 0.0, 1.0)))
        else:
            conf = min(disp / (self.still_disp_px * 3.0), 1.0) * confidence_scale
            return PersonPose(track_id=track_id, pose=Pose.LOCOMOTION,
                              confidence=float(np.clip(conf, 0.0, 1.0)))

    def _mean_displacement(self, track_id: Optional[int]) -> Optional[float]:
        """Mean per-frame centre displacement (px) over the history window."""
        if track_id is None:
            return None
        hist = self._centers.get(track_id)
        if hist is None or len(hist) < 2:
            return None
        pts = list(hist)
        disps = [
            ((pts[i + 1][0] - pts[i][0]) ** 2 +
             (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5
            for i in range(len(pts) - 1)
        ]
        return sum(disps) / len(disps)
