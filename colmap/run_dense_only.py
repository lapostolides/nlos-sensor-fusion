"""Dense reconstruction with early-exit monitoring.

Strategy:
  - filter=0 so PatchMatch writes raw depth estimates (NCC check causes NaN on
    textureless walls → all zeros when filter=1 even with ncc_threshold=-1.0)
  - min_num_pixels=1 so stereo_fusion accepts single-view points without
    requiring cross-view depth agreement (which fails on textureless walls)
"""
import subprocess, sys, time, numpy as np
from pathlib import Path

# Use the conda-installed colmap (mpir.dll now resolved from pkgs/main)
COLMAP_EXE = "colmap"
DENSE_DIR  = Path("data/testlaserroom/colmap_output/dense")
DEPTH_DIR  = DENSE_DIR / "stereo" / "depth_maps"
FUSED_PLY  = DENSE_DIR / "fused_photo.ply"

EARLY_CHECK_AFTER = 5   # check after this many maps are written
POLL_INTERVAL     = 3   # seconds between polls


def read_depth_map(path):
    data = path.read_bytes()
    amp = [i for i, b in enumerate(data[:50]) if b == 38]
    return np.frombuffer(data[amp[2]+1:], dtype=np.float32)


def sample_maps(kind, n=5):
    files = sorted(DEPTH_DIR.glob(f"*.{kind}.bin"))
    print(f"\n--- {kind}: {len(files)} maps, sampling {min(n, len(files))} ---", flush=True)
    for f in files[:n]:
        d = read_depth_map(f)
        valid = d[d > 0]
        pct = 100 * len(valid) / len(d) if len(d) else 0
        med = float(np.median(valid)) if len(valid) > 0 else 0
        rng = f"[{valid.min():.2f}, {valid.max():.2f}]" if len(valid) > 0 else "n/a"
        print(f"  {f.name[:40]}: valid={pct:.1f}%  median={med:.2f}m  range={rng}", flush=True)
    return files


def run_patchmatch_monitored(cmd, label, kind):
    """Launch PatchMatch and kill early if first N depth maps are all zeros."""
    cmd = [COLMAP_EXE if str(c) == "colmap" else str(c) for c in cmd]
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.Popen(cmd)
    checked = False

    while proc.poll() is None:
        time.sleep(POLL_INTERVAL)
        files = sorted(DEPTH_DIR.glob(f"*.{kind}.bin"))

        if not checked and len(files) >= EARLY_CHECK_AFTER:
            checked = True
            print(f"\n[early check] {len(files)} maps written — sampling...", flush=True)
            sample_maps(kind, n=EARLY_CHECK_AFTER)

            all_zero = all((read_depth_map(f) > 0).sum() == 0 for f in files[:EARLY_CHECK_AFTER])
            if all_zero:
                proc.kill()
                proc.wait()
                print("\n[FATAL] All sampled maps are zero — killed COLMAP.", flush=True)
                print("Debug info:", flush=True)
                print(f"  Command: {' '.join(cmd)}", flush=True)
                print(f"  depth_min/depth_max may be outside scene range", flush=True)
                print(f"  With filter=0 this should not happen — check COLMAP version", flush=True)
                sys.exit(1)
            else:
                print(f"[early check] Non-zero depths confirmed — continuing.", flush=True)

    if proc.returncode not in (0, None):
        print(f"ERROR: COLMAP exited with code {proc.returncode}", flush=True)
        sys.exit(proc.returncode)
    print(f"Done: {label}", flush=True)


def run(cmd, label):
    cmd = [COLMAP_EXE if str(c) == "colmap" else str(c) for c in cmd]
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)
    print(f"Done: {label}", flush=True)


# ── Step 1: Clean up stale maps ──────────────────────────────────────────────
photo_files = list(DEPTH_DIR.glob("*.photometric.bin"))
geo_files   = list(DEPTH_DIR.glob("*.geometric.bin"))
if photo_files or geo_files:
    print(f"Deleting {len(photo_files)} photometric + {len(geo_files)} geometric maps...", flush=True)
    for f in photo_files + geo_files:
        f.unlink()

# ── Step 2: Photometric PatchMatch with filter=0 ─────────────────────────────
# filter=0: write raw PatchMatch depth estimates — no NCC filter, so textureless
# walls get their (noisy) depth hypotheses preserved rather than set to 0.
# Tight depth range reduces random depth variance on textureless surfaces.
run_patchmatch_monitored(
    ["colmap", "patch_match_stereo",
     "--workspace_path", str(DENSE_DIR),
     "--PatchMatchStereo.geom_consistency", "0",
     "--PatchMatchStereo.filter", "0",
     "--PatchMatchStereo.depth_min", "0.1",
     "--PatchMatchStereo.depth_max", "6.0",
     ],
    label="1/2 patch_match_stereo (photometric, filter=0, depth=[0.1,6])",
    kind="photometric",
)

sample_maps("photometric")

# ── Step 3: Stereo fusion ────────────────────────────────────────────────────
# min_num_pixels=1: accept any pixel visible in at least 1 view, no cross-view
# depth agreement required (which always fails on textureless walls).
# max_depth_error / max_reproj_error are still applied when merging multi-view
# observations, but with min_num_pixels=1 a point is kept even if no source
# agrees.
run(["colmap", "stereo_fusion",
     "--workspace_path", str(DENSE_DIR),
     "--input_type", "photometric",
     "--output_path", str(FUSED_PLY),
     "--StereoFusion.min_num_pixels", "1",
     "--StereoFusion.max_reproj_error", "5.0",
     "--StereoFusion.max_depth_error", "0.5",
     ], label="2/2 stereo_fusion (min_num_pixels=1)")

if FUSED_PLY.exists():
    size_mb = FUSED_PLY.stat().st_size / 1e6
    with open(FUSED_PLY, 'rb') as f:
        header = f.read(500).decode('ascii', errors='ignore')
    n_pts = next((int(l.split()[-1]) for l in header.splitlines()
                  if l.startswith("element vertex")), 0)
    print(f"\nResult: {n_pts:,} points, {size_mb:.1f} MB", flush=True)
else:
    print("ERROR: fused_photo.ply not created!", flush=True)
