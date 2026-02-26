#!/usr/bin/env python3
"""COLMAP pipeline with optional ArUco-based scale estimation.

Reads images from data/test, runs sparse reconstruction, and estimates
real-world scale if ArUco markers are detected.
"""

import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PilImage

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "data/test"
WORKSPACE = ROOT / "data/colmap_output"
COLMAP_BIN = "C:/dev/bin/colmap.exe"  # override if not on PATH
DB = WORKSPACE / "database.db"
SPARSE_DIR = WORKSPACE / "sparse/0"

# Feature extraction / matching settings
# COLMAP's recommended working resolution is 1600–3000px on the long edge.
# Images above this are downscaled internally during extraction (originals untouched).
COLMAP_MAX_IMAGE_SIZE = 2000
# Maximum matches the GPU can hold without OOM on an 8 GB card.
# The brute-force buffer is max_num_matches² × 4 bytes; 32768² ≈ 4.3 GB fits, 49152² ≈ 9.7 GB does not.
GPU_SAFE_MAX_MATCHES = 32768

# ArUco settings — adjust to your marker
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE_M = 0.10  # physical side length in meters


# =============================================================================
# COLMAP
# =============================================================================


def print_gpu_info() -> None:
    """Print GPU name and memory at startup."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True
        ).strip()
        print(f"  GPU: {out}")
    except Exception:
        print("  GPU: nvidia-smi not found — GPU status unavailable")


class _GpuMonitor:
    """Polls nvidia-smi in a background thread and prints GPU stats periodically."""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL
                ).strip()
                gpu_util, mem_util, mem_used, mem_total, temp = out.split(", ")
                print(f"  [GPU] util={gpu_util}%  mem={mem_used}/{mem_total}MiB ({mem_util}%)  temp={temp}°C",
                      flush=True)
            except Exception:
                pass


def run(cmd: list[str], label: str = "") -> None:
    tag = f"[{label}] " if label else ""
    print(f"\n{'='*60}")
    print(f"{tag}$ {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    cmd[0] = COLMAP_BIN
    monitor = _GpuMonitor(interval=5.0)
    monitor.start()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
    finally:
        monitor.stop()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    print(f"\n{tag}Done in {elapsed:.1f}s")


def audit_images(image_dir: Path) -> None:
    """Print a pre-run summary of image resolutions, focal lengths, and lens models.
    Warns if multiple lenses/resolutions are detected (common iPhone issue).
    """
    EXIF_FOCAL_TAG = 37386       # FocalLength
    EXIF_LENS_TAG  = 42036       # LensModel
    EXIF_MODEL_TAG = 272         # Model (device)

    resolutions: dict[tuple, list[str]] = {}
    focal_lengths: dict[str, list[str]] = {}
    lens_models: dict[str, list[str]] = {}

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    for p in images:
        try:
            with PilImage.open(p) as img:
                res = img.size  # (w, h)
                exif = img._getexif() or {}
        except Exception:
            continue
        resolutions.setdefault(res, []).append(p.name)
        fl = exif.get(EXIF_FOCAL_TAG)
        fl_str = f"{float(fl):.1f}mm" if fl else "unknown"
        focal_lengths.setdefault(fl_str, []).append(p.name)
        lens = exif.get(EXIF_LENS_TAG, "unknown")
        lens_models.setdefault(str(lens), []).append(p.name)

    print(f"\n--- Image audit ({len(images)} images) ---")
    print(f"  Resolutions:")
    for res, names in sorted(resolutions.items()):
        print(f"    {res[0]}x{res[1]}: {len(names)} images")
    print(f"  Focal lengths:")
    for fl, names in sorted(focal_lengths.items()):
        print(f"    {fl}: {len(names)} images")
    print(f"  Lens models:")
    for lens, names in sorted(lens_models.items()):
        print(f"    {lens}: {len(names)} images")

    if len(resolutions) > 1:
        print("  WARNING: multiple resolutions detected — iPhone may have switched lenses!")
    if len(focal_lengths) > 1:
        print("  WARNING: multiple focal lengths detected — mixed lenses will hurt reconstruction!")
    print("---")


def summarize_reconstruction(sparse_dir: Path) -> None:
    """Parse the TXT sparse model and print a concise reconstruction summary."""
    cameras_txt = sparse_dir / "cameras.txt"
    images_txt  = sparse_dir / "images.txt"
    points_txt  = sparse_dir / "points3D.txt"

    if not all(p.exists() for p in [cameras_txt, images_txt, points_txt]):
        print("  (reconstruction TXT files not found, skipping summary)")
        return

    # Count registered images
    reg_images = sum(1 for l in images_txt.read_text().splitlines()
                     if l and not l.startswith("#") and len(l.split()) > 8)
    reg_images //= 2  # images.txt has 2 lines per image

    # Count 3D points and collect reprojection errors
    errors = []
    num_points = 0
    for line in points_txt.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        num_points += 1
        parts = line.split()
        if len(parts) >= 7:
            errors.append(float(parts[7]))  # mean reprojection error per point

    print(f"\n--- Reconstruction summary ---")
    print(f"  Registered images : {reg_images}")
    print(f"  3D points         : {num_points}")
    if errors:
        print(f"  Reprojection error: mean={np.mean(errors):.3f}px  median={np.median(errors):.3f}px  max={np.max(errors):.3f}px")
    if reg_images == 0:
        print("  WARNING: no images registered — reconstruction failed entirely")
    if errors and np.median(errors) > 2.0:
        print("  WARNING: high reprojection error — reconstruction may be unreliable")
    print("---\n")


def sorted_by_capture_time(image_dir: Path) -> list[str]:
    """Return image filenames sorted by EXIF capture time, falling back to filename order."""
    EXIF_DATETIME_TAG = 36867  # DateTimeOriginal

    def capture_time(p: Path):
        try:
            with PilImage.open(p) as img:
                exif = img._getexif()
                if exif and EXIF_DATETIME_TAG in exif:
                    return datetime.strptime(exif[EXIF_DATETIME_TAG], "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass
        return datetime.min  # fallback: sorts to front, then filename sort handles ties

    images = [p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    images.sort(key=lambda p: (capture_time(p), p.name))
    print(f"  Image order (first 5): {[p.name for p in images[:5]]}")
    return [p.name for p in images]


def write_image_list(image_dir: Path, workspace: Path) -> Path:
    """Write a chronologically sorted image list file for COLMAP and return its path."""
    names = sorted_by_capture_time(image_dir)
    list_path = workspace / "image_list.txt"
    list_path.write_text("\n".join(names))
    print(f"  Wrote image list ({len(names)} images) -> {list_path}")
    return list_path


def get_max_features_from_db(db_path: Path) -> int:
    """Return the largest feature count across all images in the COLMAP database.

    Used to set max_num_matches dynamically: no need to over-allocate GPU memory
    when the actual feature counts are well below the GPU-safe ceiling.
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT MAX(rows) FROM keypoints")
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row and row[0] else 8192


def run_colmap() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print(f"\n{'#'*60}")
    print(f"  COLMAP pipeline starting")
    print(f"  Images : {IMAGE_DIR}")
    print(f"  Output : {WORKSPACE}")
    print_gpu_info()
    print(f"{'#'*60}")

    audit_images(IMAGE_DIR)
    image_list = write_image_list(IMAGE_DIR, WORKSPACE)

    run(["colmap", "feature_extractor",
         "--database_path", str(DB),
         "--image_path", str(IMAGE_DIR),
         "--ImageReader.single_camera", "1",
         "--ImageReader.max_image_size", str(COLMAP_MAX_IMAGE_SIZE),
         "--image_list_path", str(image_list),
         "--SiftExtraction.use_gpu", "0",            # CPU enforces max_num_features; GPU ignores it
         "--SiftExtraction.max_num_features", "16384",
         "--SiftExtraction.domain_size_pooling", "1",  # better distinctiveness on repetitive textures
         ],
        label="1/4 feature_extractor")

    # Use the actual max feature count from the DB to set the match buffer,
    # capped at what the GPU can hold without OOM.
    actual_max = get_max_features_from_db(DB)
    max_matches = min(actual_max, GPU_SAFE_MAX_MATCHES)
    print(f"  Max features in DB: {actual_max} → max_num_matches={max_matches}")

    run(["colmap", "exhaustive_matcher",
         "--database_path", str(DB),
         "--FeatureMatching.max_num_matches", str(max_matches),
         "--SiftMatching.guided_matching", "1",  # epipolar filtering reduces false matches
         ],
        label="2/4 exhaustive_matcher")
    run(["colmap", "mapper", "--database_path", str(DB), "--image_path", str(IMAGE_DIR), "--output_path", str(SPARSE_DIR.parent)],
        label="3/4 mapper")
    run(["colmap", "model_converter", "--input_path", str(SPARSE_DIR), "--output_path", str(SPARSE_DIR), "--output_type", "TXT"],
        label="4/4 model_converter")

    summarize_reconstruction(SPARSE_DIR)
    print(f"\n{'#'*60}")
    print(f"  COLMAP pipeline complete in {time.time() - t_total:.1f}s")
    print(f"{'#'*60}\n")


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
