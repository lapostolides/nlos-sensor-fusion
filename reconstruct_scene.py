#!/usr/bin/env python3
"""
reconstruct_scene.py — 3D scene reconstruction from RealSense RGB-D captures.

Takes a capture directory (from capture_realsense.py or full_capture.py)
and reconstructs a metric 3D mesh using Open3D TSDF volume integration.

Two backends:
  GPU (CUDA) — tensor SLAM pipeline: single-pass track→integrate→raycast,
               tracks against model raycast (less drift), ~10x faster.
  CPU        — legacy pipeline: frame-to-frame odometry then TSDF integration.

The GPU backend requires Open3D built with CUDA (not the default pip wheel on
Windows). When CUDA is unavailable the script automatically falls back to CPU.

Usage:
    python reconstruct_scene.py data/logs/kitchen-scan/
    python reconstruct_scene.py data/logs/kitchen-scan/ --every 5 --voxel-size 0.005
    python reconstruct_scene.py data/logs/kitchen-scan/ --device CPU:0
    python reconstruct_scene.py   # auto-detect latest run with sensor_cam data

Output:
    <run_dir>/reconstruction/mesh.ply
    <run_dir>/reconstruction/pointcloud.ply
    <run_dir>/reconstruction/trajectory.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


# ── Helpers ──────────────────────────────────────────────────────────────


def resolve_run_dir(path_arg: str | None) -> Path:
    """Resolve run directory: explicit path or auto-detect latest."""
    if path_arg:
        run_dir = Path(path_arg)
        if not run_dir.exists():
            print(f"Error: {run_dir} does not exist.")
            sys.exit(1)
        return run_dir

    log_root = Path("data/logs")
    if not log_root.exists():
        print("No data/logs/ directory found.")
        sys.exit(1)

    candidates = [
        mf.parent
        for mf in log_root.rglob("manifest.json")
        if (mf.parent / "sensor_cam" / "depth").exists()
    ]
    if not candidates:
        print("No runs with sensor_cam data found in data/logs/.")
        sys.exit(1)

    run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[reconstruct] Auto-detected: {run_dir}")
    return run_dir


def resolve_device(device_arg: str) -> str:
    """Return 'cuda' or 'cpu' based on CLI arg and hardware availability."""
    cuda_ok = o3d.core.cuda.is_available()
    if device_arg == "auto":
        if cuda_ok:
            print("[reconstruct] Device: CUDA:0 (auto-detected)")
            return "cuda"
        else:
            print("[reconstruct] Device: CPU (CUDA not available)")
            return "cpu"
    if device_arg.upper().startswith("CUDA"):
        if cuda_ok:
            print(f"[reconstruct] Device: {device_arg}")
            return "cuda"
        else:
            print(f"[reconstruct] WARNING: {device_arg} requested but CUDA "
                  f"unavailable — falling back to CPU")
            return "cpu"
    print("[reconstruct] Device: CPU")
    return "cpu"


def load_manifest(run_dir: Path) -> dict:
    """Load and validate manifest.json."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found.")
        sys.exit(1)
    with open(manifest_path) as f:
        return json.load(f)


def get_intrinsics(manifest: dict) -> tuple[dict, int, int]:
    """Extract raw intrinsics dict and resolution from manifest.

    Returns (color_intrinsics_dict, width, height).
    """
    sensor = manifest.get("sensors", {}).get("sensor_cam")
    if sensor is None:
        print("Error: manifest.json has no sensor_cam metadata.")
        sys.exit(1)
    ci = sensor["intrinsics"]["color"]
    w, h = sensor["resolution"]
    return ci, w, h


def load_depth_filtered(
    depth_dir: Path,
    frame_idx: int,
    n_total: int,
    window: int,
    cache: dict[int, np.ndarray | None],
) -> np.ndarray | None:
    """Load a temporally-filtered depth frame.

    For the given *frame_idx*, loads raw depth frames in
    ``[frame_idx - window, frame_idx + window]`` (clamped to ``[0, n_total)``),
    computes the per-pixel median over the window (excluding zero / invalid
    pixels), and returns the result as a uint16 ndarray in millimetres.

    Parameters
    ----------
    depth_dir:
        Directory containing depth PNGs named ``{idx:06d}.png``.
    frame_idx:
        The centre frame index.
    n_total:
        Total number of frames in the dataset (for boundary clamping).
    window:
        Half-width of the temporal window.  0 = no filtering (returns the
        single frame as-is).
    cache:
        Mutable cache mapping frame index -> uint16 depth array (or None for
        missing frames).  Populated on demand; pruned by the caller.

    Returns
    -------
    np.ndarray or None
        Filtered depth image (H, W) uint16, or None if the centre frame is
        missing.
    """
    if window == 0:
        # Fast path: no filtering — load the single frame
        if frame_idx not in cache:
            p = depth_dir / f"{frame_idx:06d}.png"
            cache[frame_idx] = (
                cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.exists() else None
            )
        return cache[frame_idx]

    lo = max(0, frame_idx - window)
    hi = min(n_total - 1, frame_idx + window)
    indices = list(range(lo, hi + 1))

    # Populate cache for every frame in the window
    for idx in indices:
        if idx not in cache:
            p = depth_dir / f"{idx:06d}.png"
            cache[idx] = (
                cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.exists() else None
            )

    # Centre frame must exist
    if cache.get(frame_idx) is None:
        return None

    # Collect valid (non-None) depth arrays
    valid_frames = [cache[idx] for idx in indices if cache[idx] is not None]
    if len(valid_frames) == 1:
        return valid_frames[0]

    # Stack and compute per-pixel median, treating zero as invalid
    stack = np.stack(valid_frames, axis=0).astype(np.float32)  # (N, H, W)
    stack[stack == 0] = np.nan

    with np.errstate(invalid="ignore"):
        median = np.nanmedian(stack, axis=0)

    # All-NaN pixels (all zeros) → keep as 0
    return np.where(np.isnan(median), 0, median).astype(np.uint16)


# ═══════════════════════════════════════════════════════════════════════════
# GPU backend — tensor SLAM (requires Open3D with CUDA)
# ═══════════════════════════════════════════════════════════════════════════


def run_slam_gpu(
    frame_indices: list[int],
    rgb_dir: Path,
    depth_dir: Path,
    ci: dict,
    height: int,
    width: int,
    *,
    voxel_size: float,
    block_count: int,
    trunc_multiplier: float,
    depth_scale: float,
    depth_max: float,
    depth_min: float = 0.1,
    depth_window: int = 0,
    n_total: int = 0,
) -> tuple[o3d.t.pipelines.slam.Model, list[np.ndarray]]:
    """Single-pass GPU SLAM: track → integrate → raycast per frame."""
    device = o3d.core.Device("CUDA:0")
    K = np.array([
        [ci["fx"], 0.0,      ci["ppx"]],
        [0.0,      ci["fy"], ci["ppy"]],
        [0.0,      0.0,      1.0      ],
    ])
    intrinsic = o3d.core.Tensor(K, dtype=o3d.core.Dtype.Float64, device=device)

    model = o3d.t.pipelines.slam.Model(
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=block_count,
        transformation=o3d.core.Tensor(np.eye(4)),
        device=device,
    )
    input_frame = o3d.t.pipelines.slam.Frame(height, width, intrinsic, device)
    raycast_frame = o3d.t.pipelines.slam.Frame(height, width, intrinsic, device)

    T = o3d.core.Tensor(np.eye(4))
    poses: list[np.ndarray] = []
    failures = 0
    n = len(frame_indices)
    depth_cache: dict[int, np.ndarray | None] = {}
    t_start = time.perf_counter()

    for i, idx in enumerate(frame_indices):
        fname = f"{idx:06d}"
        rgb_path = rgb_dir / f"{fname}.jpg"

        if not rgb_path.exists():
            print(f"  [{i+1}/{n}] WARNING: frame {idx} missing, skipping")
            poses.append(poses[-1].copy() if poses else np.eye(4))
            failures += 1
            continue

        # ── Load depth (optionally filtered) ─────────────────────────
        if depth_window > 0:
            depth_np = load_depth_filtered(
                depth_dir, idx, n_total, depth_window, depth_cache,
            )
            if depth_np is None:
                print(f"  [{i+1}/{n}] WARNING: frame {idx} depth missing, skipping")
                poses.append(poses[-1].copy() if poses else np.eye(4))
                failures += 1
                continue
            depth = o3d.t.geometry.Image(
                o3d.core.Tensor(np.ascontiguousarray(depth_np))
            ).to(device)
        else:
            depth_path = depth_dir / f"{fname}.png"
            if not depth_path.exists():
                print(f"  [{i+1}/{n}] WARNING: frame {idx} missing, skipping")
                poses.append(poses[-1].copy() if poses else np.eye(4))
                failures += 1
                continue
            depth = o3d.t.io.read_image(str(depth_path)).to(device)

        color = o3d.t.io.read_image(str(rgb_path)).to(device)
        input_frame.set_data_from_image("depth", depth)
        input_frame.set_data_from_image("color", color)

        if i > 0:
            try:
                result = model.track_frame_to_model(
                    input_frame, raycast_frame,
                    depth_scale, depth_max,
                )
                T = T @ result.transformation
            except RuntimeError:
                failures += 1

        poses.append(T.cpu().numpy().copy())
        model.update_frame_pose(i, T)
        model.integrate(
            input_frame, depth_scale, depth_max,
            trunc_voxel_multiplier=trunc_multiplier,
        )
        model.synthesize_model_frame(
            raycast_frame, depth_scale, depth_min, depth_max,
            trunc_voxel_multiplier=trunc_multiplier,
        )

        # Prune stale cache entries
        if depth_window > 0:
            cutoff = idx - depth_window
            for k in [k for k in depth_cache if k < cutoff]:
                del depth_cache[k]

        # Progress every 20 frames, at first frame, and at last frame
        if i == 0 or (i + 1) % 20 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t_start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / fps if fps > 0 else 0
            print(
                f"  [{i+1}/{n}] frame {idx} | "
                f"{fps:.1f} fps | {elapsed:.1f}s elapsed | ~{eta:.0f}s remaining"
            )

    if failures:
        print(f"  WARNING: tracking issues on {failures}/{n} frames")
    return model, poses


# ═══════════════════════════════════════════════════════════════════════════
# CPU backend — legacy odometry + ScalableTSDFVolume
# ═══════════════════════════════════════════════════════════════════════════


def _load_rgbd_legacy(
    rgb_dir: Path,
    depth_dir: Path,
    idx: int,
    depth_scale: float,
    depth_max: float,
    depth_np: np.ndarray | None = None,
) -> o3d.geometry.RGBDImage | None:
    """Load one RGB-D frame as a legacy Open3D RGBDImage.

    If *depth_np* is provided (e.g. from ``load_depth_filtered``), it is used
    directly instead of reading the depth PNG from disk.
    """
    fname = f"{idx:06d}"
    rgb_path = rgb_dir / f"{fname}.jpg"
    if not rgb_path.exists():
        return None

    color = o3d.io.read_image(str(rgb_path))

    if depth_np is not None:
        depth = o3d.geometry.Image(np.ascontiguousarray(depth_np))
    else:
        depth_path = depth_dir / f"{fname}.png"
        if not depth_path.exists():
            return None
        depth = o3d.io.read_image(str(depth_path))

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=depth_scale,
        depth_trunc=depth_max,
        convert_rgb_to_intensity=False,
    )


def run_slam_cpu(
    frame_indices: list[int],
    rgb_dir: Path,
    depth_dir: Path,
    ci: dict,
    height: int,
    width: int,
    *,
    voxel_size: float,
    sdf_trunc: float,
    depth_scale: float,
    depth_max: float,
    depth_window: int = 0,
    n_total: int = 0,
) -> tuple[o3d.pipelines.integration.ScalableTSDFVolume, list[np.ndarray]]:
    """Two-pass CPU pipeline: odometry then TSDF integration."""
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height,
        fx=ci["fx"], fy=ci["fy"],
        cx=ci["ppx"], cy=ci["ppy"],
    )

    n = len(frame_indices)
    depth_cache: dict[int, np.ndarray | None] = {}

    # Helper: load an RGB-D pair, with optional depth filtering
    def _load(idx: int) -> o3d.geometry.RGBDImage | None:
        if depth_window > 0:
            d = load_depth_filtered(depth_dir, idx, n_total, depth_window, depth_cache)
            return _load_rgbd_legacy(
                rgb_dir, depth_dir, idx, depth_scale, depth_max, depth_np=d,
            )
        return _load_rgbd_legacy(
            rgb_dir, depth_dir, idx, depth_scale, depth_max,
        )

    # ── Pass 1: frame-to-frame odometry ──────────────────────────────────
    print(f"  Pass 1/2: RGB-D odometry ({n} frames)")
    option = o3d.pipelines.odometry.OdometryOption(
        depth_diff_max=0.07,
        depth_min=0.1,
        depth_max=depth_max,
    )
    odo_method = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    poses: list[np.ndarray] = [np.eye(4)]
    prev_rgbd = _load(frame_indices[0])
    if prev_rgbd is None:
        print(f"Error: cannot load first frame {frame_indices[0]}.")
        sys.exit(1)

    failures = 0
    t_start = time.perf_counter()

    for i in range(1, n):
        curr_rgbd = _load(frame_indices[i])
        if curr_rgbd is None:
            poses.append(poses[-1].copy())
            failures += 1
            continue

        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            curr_rgbd, prev_rgbd,
            intrinsic, np.eye(4),
            odo_method, option,
        )

        if success:
            poses.append(poses[-1] @ trans)
        else:
            poses.append(poses[-1].copy())
            failures += 1

        prev_rgbd = curr_rgbd

        # Prune stale cache entries
        if depth_window > 0:
            cutoff = frame_indices[i] - depth_window
            for k in [k for k in depth_cache if k < cutoff]:
                del depth_cache[k]

        if (i + 1) % 20 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t_start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / fps if fps > 0 else 0
            print(
                f"    [{i+1}/{n}] frame {frame_indices[i]} | "
                f"{fps:.1f} fps | {elapsed:.1f}s elapsed | ~{eta:.0f}s remaining"
            )

    if failures:
        print(f"    WARNING: odometry failed on {failures}/{n - 1} pairs")
    t_odo = time.perf_counter() - t_start
    print(f"  Pass 1 done in {t_odo:.1f}s")

    # ── Pass 2: TSDF integration ─────────────────────────────────────────
    print(f"  Pass 2/2: TSDF integration ({n} frames)")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # Reset cache for pass 2 (different access pattern)
    depth_cache.clear()
    t_start = time.perf_counter()
    for i, (idx, pose) in enumerate(zip(frame_indices, poses)):
        rgbd = _load(idx)
        if rgbd is None:
            continue
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)

        # Prune stale cache entries
        if depth_window > 0:
            cutoff = idx - depth_window
            for k in [k for k in depth_cache if k < cutoff]:
                del depth_cache[k]

        if (i + 1) % 20 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t_start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / fps if fps > 0 else 0
            print(
                f"    [{i+1}/{n}] frame {idx} | "
                f"{fps:.1f} fps | {elapsed:.1f}s elapsed | ~{eta:.0f}s remaining"
            )

    t_int = time.perf_counter() - t_start
    print(f"  Pass 2 done in {t_int:.1f}s")

    return volume, poses


# ═══════════════════════════════════════════════════════════════════════════
# Extraction & saving
# ═══════════════════════════════════════════════════════════════════════════


def extract_and_save_gpu(model, output_dir: Path):
    """Extract mesh + point cloud from tensor SLAM model."""
    print("  Extracting triangle mesh...")
    mesh_t = model.extract_triangle_mesh()
    mesh = mesh_t.to_legacy()
    mesh.compute_vertex_normals()
    mesh_path = output_dir / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(
        f"  Mesh: {mesh_path} "
        f"({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)"
    )

    print("  Extracting point cloud...")
    pcd_t = model.extract_point_cloud()
    pcd = pcd_t.to_legacy()
    pcd_path = output_dir / "pointcloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    print(f"  Point cloud: {pcd_path} ({len(pcd.points)} points)")


def extract_and_save_cpu(volume, output_dir: Path):
    """Extract mesh + point cloud from legacy ScalableTSDFVolume."""
    print("  Extracting triangle mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh_path = output_dir / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(
        f"  Mesh: {mesh_path} "
        f"({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)"
    )

    print("  Extracting point cloud...")
    pcd = volume.extract_point_cloud()
    pcd_path = output_dir / "pointcloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    print(f"  Point cloud: {pcd_path} ({len(pcd.points)} points)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct a 3D scene from RealSense RGB-D capture.",
    )
    parser.add_argument(
        "path", nargs="?",
        help="Run directory with manifest.json (auto-detects latest if omitted)",
    )
    parser.add_argument(
        "--every", type=int, default=5,
        help="Use every Nth frame (default: 5, i.e. ~6fps from 30fps)",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.005,
        help="TSDF voxel size in metres (default: 0.005 = 5mm)",
    )
    parser.add_argument(
        "--trunc-multiplier", type=float, default=8.0,
        help="GPU: SDF trunc = voxel_size × this (default: 8.0)",
    )
    parser.add_argument(
        "--depth-scale", type=float, default=1000.0,
        help="Depth scale factor, mm to metres (default: 1000.0)",
    )
    parser.add_argument(
        "--depth-max", type=float, default=3.0,
        help="Max depth in metres to integrate (default: 3.0)",
    )
    parser.add_argument(
        "--depth-window", type=int, default=0,
        help="Temporal depth filter half-width in raw frames. "
             "0 = disabled. 2 = median of 5 frames (default: 0)",
    )
    parser.add_argument(
        "--block-count", type=int, default=100_000,
        help="GPU: max TSDF hash blocks (default: 100000)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="CUDA:0, CPU:0, or auto (default: auto)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: <run_dir>/reconstruction/)",
    )
    args = parser.parse_args()

    # ── Resolve paths & device ───────────────────────────────────────────
    run_dir = resolve_run_dir(args.path)
    output_dir = Path(args.output) if args.output else run_dir / "reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = resolve_device(args.device)

    manifest = load_manifest(run_dir)
    ci, width, height = get_intrinsics(manifest)
    sensor = manifest["sensors"]["sensor_cam"]
    print(
        f"[reconstruct] Camera: {sensor.get('device_name', 'RealSense')} "
        f"({width}x{height})"
    )
    print(
        f"[reconstruct] Intrinsics: fx={ci['fx']:.1f} fy={ci['fy']:.1f} "
        f"cx={ci['ppx']:.1f} cy={ci['ppy']:.1f}"
    )

    # ── Enumerate frames ─────────────────────────────────────────────────
    rgb_dir = run_dir / "sensor_cam" / "rgb"
    depth_dir = run_dir / "sensor_cam" / "depth"
    n_total = manifest.get("frames", {}).get("sensor_cam", {}).get("count", 0)
    if n_total == 0:
        print("Error: No sensor_cam frames in manifest.")
        sys.exit(1)

    frame_indices = list(range(0, n_total, args.every))
    n_frames = len(frame_indices)
    print(
        f"[reconstruct] Frames: {n_total} total, using every {args.every}th "
        f"= {n_frames} frames"
    )

    sdf_trunc = args.voxel_size * args.trunc_multiplier

    # ── Run reconstruction ───────────────────────────────────────────────
    t0 = time.perf_counter()

    depth_filter_msg = ""
    if args.depth_window > 0:
        depth_filter_msg = (
            f"\n  Depth filter: temporal median, "
            f"window={args.depth_window} ({2 * args.depth_window + 1} frames)"
        )

    if backend == "cuda":
        print(
            f"\n[reconstruct] === GPU SLAM (single-pass track+integrate+raycast) ===\n"
            f"  Voxel: {args.voxel_size}m | "
            f"Trunc: {sdf_trunc:.3f}m ({args.trunc_multiplier}× voxel) | "
            f"Depth max: {args.depth_max}m | Blocks: {args.block_count}"
            f"{depth_filter_msg}"
        )
        model_or_vol, poses = run_slam_gpu(
            frame_indices, rgb_dir, depth_dir, ci, height, width,
            voxel_size=args.voxel_size,
            block_count=args.block_count,
            trunc_multiplier=args.trunc_multiplier,
            depth_scale=args.depth_scale,
            depth_max=args.depth_max,
            depth_window=args.depth_window,
            n_total=n_total,
        )
    else:
        print(
            f"\n[reconstruct] === CPU pipeline (odometry → TSDF integration) ===\n"
            f"  Voxel: {args.voxel_size}m | "
            f"SDF trunc: {sdf_trunc:.3f}m | "
            f"Depth max: {args.depth_max}m"
            f"{depth_filter_msg}"
        )
        model_or_vol, poses = run_slam_cpu(
            frame_indices, rgb_dir, depth_dir, ci, height, width,
            voxel_size=args.voxel_size,
            sdf_trunc=sdf_trunc,
            depth_scale=args.depth_scale,
            depth_max=args.depth_max,
            depth_window=args.depth_window,
            n_total=n_total,
        )

    t_slam = time.perf_counter() - t0
    print(f"\n  SLAM total: {t_slam:.1f}s ({n_frames / t_slam:.1f} frames/s)")

    # ── Extract geometry ─────────────────────────────────────────────────
    print(f"\n[reconstruct] === Extracting geometry ===")
    t1 = time.perf_counter()

    if backend == "cuda":
        extract_and_save_gpu(model_or_vol, output_dir)
    else:
        extract_and_save_cpu(model_or_vol, output_dir)

    t_extract = time.perf_counter() - t1
    print(f"  Done in {t_extract:.1f}s")

    # ── Save trajectory ──────────────────────────────────────────────────
    trajectory = {
        "frame_indices": frame_indices,
        "poses": [pose.tolist() for pose in poses],
        "intrinsics": {
            "width": width, "height": height,
            "fx": ci["fx"], "fy": ci["fy"],
            "cx": ci["ppx"], "cy": ci["ppy"],
        },
        "params": {
            "backend": backend,
            "every": args.every,
            "voxel_size": args.voxel_size,
            "sdf_trunc": sdf_trunc,
            "trunc_multiplier": args.trunc_multiplier,
            "depth_scale": args.depth_scale,
            "depth_max": args.depth_max,
            "depth_window": args.depth_window,
            "block_count": args.block_count,
        },
    }
    traj_path = output_dir / "trajectory.json"
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"  Trajectory: {traj_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    total = time.perf_counter() - t0
    print(f"\n[reconstruct] Done in {total:.1f}s. Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
