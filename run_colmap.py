#!/usr/bin/env python3
"""COLMAP pipeline with optional ArUco-based scale estimation.

Reads images from data/test, runs sparse reconstruction, and estimates
real-world scale if ArUco markers are detected.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "data/test"
WORKSPACE = ROOT / "data/colmap_output"
DB = WORKSPACE / "database.db"
SPARSE_DIR = WORKSPACE / "sparse/0"

# ArUco settings — adjust to your marker
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE_M = 0.10  # physical side length in meters


# =============================================================================
# COLMAP
# =============================================================================


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_colmap() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)

    run(["colmap", "feature_extractor", "--database_path", str(DB), "--image_path", str(IMAGE_DIR)])
    run(["colmap", "exhaustive_matcher", "--database_path", str(DB)])
    run(["colmap", "mapper", "--database_path", str(DB), "--image_path", str(IMAGE_DIR), "--output_path", str(SPARSE_DIR.parent)])
    run(["colmap", "model_converter", "--input_path", str(SPARSE_DIR), "--output_path", str(SPARSE_DIR), "--output_type", "TXT"])


# =============================================================================
# ArUco scale estimation
# =============================================================================


def detect_aruco_cc_hardware(image_paths: list[Path]) -> dict[str, np.ndarray]:
    """Alternative to detect_aruco() using ArucoLocalizationAlgorithm from spad_center.

    Requires cc_hardware to be installed (pip install -e spad_center).
    Returns the same {image_name: corners (N, 4, 2)} format as detect_aruco().
    """
    # Ensure cc_hardware is importable when running from the repo root
    sys.path.insert(0, str(ROOT / "spad_center/pkgs/algos"))
    sys.path.insert(0, str(ROOT / "spad_center/pkgs/utils"))
    from cc_hardware.algos.aruco import ArucoLocalizationAlgorithm

    class _StaticCamera:
        """Minimal Camera stub that serves a single static image."""
        is_okay = True

        def __init__(self):
            self._img = None

        def set(self, img: np.ndarray) -> None:
            self._img = img

        def accumulate(self, _n: int) -> np.ndarray:
            return self._img[np.newaxis]

        def close(self) -> None:
            pass

    cam = _StaticCamera()
    # ArucoLocalizationAlgorithm sets up the cv2.aruco.ArucoDetector; we reuse it
    algo = ArucoLocalizationAlgorithm(cam, aruco_dict=ARUCO_DICT, marker_size=MARKER_SIZE_M)

    detections = {}
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cam.set(img)
        corners, ids, _ = algo._detector.detectMarkers(img)
        if ids is not None:
            detections[path.name] = np.array([c[0] for c in corners])  # (N, 4, 2)

    return detections


def read_db_keypoints(image_name: str) -> tuple[int | None, np.ndarray | None]:
    """Returns (image_id, keypoints (K, 2)) from the COLMAP database."""
    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()
    cur.execute("SELECT image_id FROM images WHERE name = ?", (image_name,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return None, None
    image_id = row[0]
    cur.execute("SELECT data FROM keypoints WHERE image_id = ?", (image_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return image_id, None
    kps = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 6)
    return image_id, kps[:, :2]


def read_sparse_model() -> tuple[dict, dict]:
    """Returns (points3d {id: xyz}, img_kp_map {name: {kp_idx: point3d_id}})."""
    points3d = {}
    with open(SPARSE_DIR / "points3D.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            points3d[int(parts[0])] = np.array(parts[1:4], dtype=float)

    img_kp_map = {}
    with open(SPARSE_DIR / "images.txt") as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):
        name = lines[i].split()[9]
        pts = lines[i + 1].split()
        img_kp_map[name] = {
            j // 3: int(pts[j + 2])
            for j in range(0, len(pts), 3)
            if int(pts[j + 2]) != -1
        }
    return points3d, img_kp_map


def estimate_scale(detections: dict[str, np.ndarray]) -> float | None:
    """Compute median scale (m/unit) from ArUco corner 3D points."""
    points3d, img_kp_map = read_sparse_model()
    scales = []

    for img_name, markers in detections.items():
        _, keypoints = read_db_keypoints(img_name)
        if keypoints is None or img_name not in img_kp_map:
            continue
        kp_to_3d = img_kp_map[img_name]

        for corners in markers:  # corners: (4, 2), adjacent pairs = MARKER_SIZE_M apart
            pts3d = []
            for corner in corners:
                dists = np.linalg.norm(keypoints - corner, axis=1)
                idx = int(np.argmin(dists))
                if dists[idx] < 5.0 and idx in kp_to_3d:
                    pts3d.append(points3d[kp_to_3d[idx]])

            for i in range(len(pts3d)):
                for j in range(i + 1, len(pts3d)):
                    colmap_dist = np.linalg.norm(pts3d[i] - pts3d[j])
                    if colmap_dist > 0:
                        # adjacent corners → MARKER_SIZE_M; diagonal → MARKER_SIZE_M * √2
                        real_dist = MARKER_SIZE_M if abs(i - j) in (1, 3) else MARKER_SIZE_M * np.sqrt(2)
                        scales.append(real_dist / colmap_dist)

    if not scales:
        print("No scale estimated — ArUco corners not matched to 3D points.")
        return None

    scale = float(np.median(scales))
    print(f"Scale: {scale:.6f} m/unit  (n={len(scales)})")
    return scale


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    run_colmap()

    image_paths = sorted(IMAGE_DIR.iterdir())
    detections = detect_aruco(image_paths)

    if detections:
        print(f"ArUco markers detected in {len(detections)} image(s).")
        estimate_scale(detections)
    else:
        print("No ArUco markers detected; skipping scale estimation.")


if __name__ == "__main__":
    main()
