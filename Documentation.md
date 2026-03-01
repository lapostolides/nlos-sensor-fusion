# NLOS Sensor Fusion — Documentation

## Overview

This project fuses data from two complementary sensors to locate objects hidden from direct view (Non-Line-of-Sight, NLOS):

- **SPAD** (VL53L8CH) — captures time-resolved photon histograms, sensitive to indirect light paths that bounce off walls and around occlusions.
- **RealSense RGB-D** — captures aligned color and depth frames for scene geometry and ground-truth person detection.

The top-level pipeline consists of three stages:

1. **Capture** — `full_capture.py` streams both sensors concurrently and writes synchronized records to a `.pkl` file.
2. **Post-hoc Synchronization** — `sync_data.py` aligns SPAD and camera frames by timestamp and produces a lightweight JSON index.
3. **Ground Truth Labeling** — `ground_truth.py` provides a YOLO-based person detector that can be applied to captured RGB frames.

---

## File Reference

| File | Role |
|---|---|
| `capture_config.py` | User-editable configuration for `full_capture.py` |
| `full_capture.py` | Multi-sensor capture script (SPAD + RealSense) |
| `sync_data.py` | Post-hoc temporal sync of SPAD and camera frames |
| `ground_truth.py` | YOLO person detector for ground-truth labeling |
| `requirements.txt` | Root-level Python dependencies |
| `spad_center/` | `cc_hardware` driver package (SPAD hardware, cameras, utilities) |

---

## Configuration — `capture_config.py`

Edit this file before running `full_capture.py`. It is a plain Python module; no CLI flags are needed.

```python
# Label attached to the output file name
OBJECT = "test"
```

### Sensor Toggles

| Variable | Type | Description |
|---|---|---|
| `USE_SPAD` | `bool` | Enable/disable VL53L8CH SPAD capture |
| `USE_REALSENSE` | `bool` | Enable/disable RealSense RGB-D capture |

### Capture Mode

| Variable | Values | Description |
|---|---|---|
| `CAPTURE_MODE` | `"loop"` | Continuous capture at sensor frame rate. Stop with **Ctrl+C**. |
| | `"manual"` | Press **Enter** to grab a frame, **q** to quit. |

### Live Preview

| Variable | Type | Description |
|---|---|---|
| `SHOW_SPAD_DASHBOARD` | `bool` | Show PyQtGraph histogram dashboard |
| `SHOW_REALSENSE_PREVIEW` | `bool` | Show OpenCV RGB + depth preview window |

### SPAD Settings

| Variable | Values | Description |
|---|---|---|
| `SPAD_RESOLUTION` | `"4x4"`, `"8x8"` | Sensor zone grid |
| `SPAD_PORT` | e.g. `"COM4"` | Serial port for the VL53L8CH (Windows). Use `/dev/ttyACM*` on Linux. |

### RealSense Settings

| Variable | Default | Description |
|---|---|---|
| `RS_WIDTH` | `848` | Frame width in pixels |
| `RS_HEIGHT` | `480` | Frame height in pixels |
| `RS_FPS` | `30` | Target frame rate |

---

## Capture Script — `full_capture.py`

### Usage

```bash
python full_capture.py
```

Configuration is read from `capture_config.py`. No arguments are accepted.

### What It Does

`full_capture.py` initializes both sensors, spawns a background thread for each, and writes every frame to a single `.pkl` file via `PklHandler`.

#### Initialization

1. Creates a timestamped output directory: `data/logs/YYYY-MM-DD/HH-MM-SS/`.
2. Starts a `Manager` context that owns all sensor handles and ensures clean teardown.
3. **SPAD setup** (`setup_spad`): initializes `VL53L8CHConfig` (4×4 or 8×8) wrapped in `SPADMergeWrapperConfig` with histogram data type. Optionally opens a PyQtGraph dashboard.
4. **RealSense setup** (`setup_realsense`): starts an aligned depth+color pipeline at the configured resolution/FPS. Records device name, serial number, and camera intrinsics into the metadata record.
5. Writes a `{"metadata": {...}}` record as the first entry in the PKL.

#### Worker Threads

Each enabled sensor runs in a dedicated daemon thread:

| Thread | Function | Behaviour |
|---|---|---|
| `spad-worker` | `_spad_worker` | Calls `sensor.accumulate()` in a tight loop; stores the latest result in a `LatestFrame` buffer. |
| `rs-worker` | `_realsense_worker` | Calls `pipeline.wait_for_frames()`; aligns depth to color; stores `raw_depth`, `raw_rgb`, `aligned_depth`, `aligned_rgb` in a `LatestFrame` buffer. |

`LatestFrame` is a single-slot, thread-safe buffer. Workers call `put()`; the main thread calls `get()` (non-blocking) or `wait_for_new()` (blocking with timeout).

#### Capture Loop

**Continuous mode** (`"loop"`):
- The main thread blocks on `pace_buf.wait_for_new()` — paced by the SPAD if enabled, otherwise by RealSense.
- Each iteration assembles a record:
  ```python
  {
      "iter": int,
      "spad": np.ndarray,          # histogram data (if USE_SPAD)
      "spad_timestamp": str,        # ISO 8601
      "realsense": dict,            # raw/aligned frames (if USE_REALSENSE)
      "realsense_timestamp": str,   # ISO 8601
  }
  ```
- The record is appended to the PKL and displayed in the live preview.
- Frame rate is printed every 30 iterations.

**Manual mode** (`"manual"`):
- Threads run in the background; pressing **Enter** snapshots the latest frames from each buffer.

#### Teardown

When the loop exits (Ctrl+C or `q`), `_join_workers` sets the stop event and joins threads with a 3-second timeout. The `Manager` context closes all sensor handles. OpenCV windows are destroyed.

---

## Post-hoc Sync — `sync_data.py`

### Usage

```bash
# Analyze the most recently created PKL in data/logs/
python sync_data.py

# Analyze a specific file
python sync_data.py path/to/capture.pkl

# Set a custom time-difference warning threshold (default: 100 ms)
python sync_data.py path/to/capture.pkl --max-dt-ms 50

# Skip writing the JSON index
python sync_data.py path/to/capture.pkl --no-index
```

### What It Does

`sync_data.py` performs **nearest-neighbour temporal matching** between SPAD frames and camera frames using their recorded timestamps.

#### Timestamp Layout Support

The script handles two record formats produced by `full_capture.py`:

| Format | Keys |
|---|---|
| Shared timestamp | `{"iter", "timestamp", "spad", "realsense"}` |
| Per-sensor timestamp | `{"iter", "spad", "spad_timestamp", "realsense", "realsense_timestamp"}` |

Per-sensor keys take priority; shared `"timestamp"` is used as a fallback.

#### Matching Algorithm

1. All SPAD timestamps and camera timestamps are extracted into sorted lists.
2. For every SPAD frame, a binary search (`bisect`) finds the camera frame with the minimum absolute time difference.
3. The result is a list of `(spad_idx, cam_idx, dt_seconds)` tuples.

#### Report

A summary is printed to stdout:

```
----------------------------------------------------
  SPAD frames:               120
  Camera frames:             180
----------------------------------------------------
  Matched SPAD frames:       120  (each -> nearest cam)
  Unique camera matched:     105
  Unmatched camera:           75

  Per-pair time differences (ms):
    SPAD[   0] <-> cam[   0]  dt =     4.21
    ...
  Average |dt|: 6.32 ms  (min 0.11, max 38.70)
  Avg interval between matched frames: 33.33 ms  (30.0 fps)

  All pairs within 100 ms threshold.
----------------------------------------------------
```

Pairs exceeding `--max-dt-ms` are flagged with `***`.

#### JSON Sync Index

Unless `--no-index` is passed, a JSON file is written alongside the PKL:

```
data/logs/2025-01-01/12-00-00/object_data_sync.json
```

Structure:

```json
{
  "source_pkl": "/absolute/path/to/object_data.pkl",
  "created": "2025-01-01T12:05:00",
  "max_dt_ms": 100.0,
  "n_spad": 120,
  "n_cam": 180,
  "n_pairs": 120,
  "n_pairs_over_threshold": 0,
  "pairs": [
    {
      "spad_idx": 0,
      "cam_idx": 0,
      "dt_ms": 4.211,
      "spad_ts": "2025-01-01T12:00:00.123",
      "cam_ts": "2025-01-01T12:00:00.127"
    },
    ...
  ]
}
```

The index contains only indices and timing metadata, not sensor arrays. Downstream code can use it to load matched pairs on demand from the raw PKL.

### Public API

```python
from sync_data import sync
from pathlib import Path

matches = sync(Path("data/logs/.../object_data.pkl"), max_dt_ms=50.0, save=True)
# matches: list of (spad_idx, cam_idx, dt_seconds)
```

---

## Ground Truth — `ground_truth.py`

Provides YOLO-based person detection to label captured RGB frames.

### Classes

#### `PersonLocation`

A dataclass representing a single detected person:

| Field | Type | Description |
|---|---|---|
| `bbox` | `(x, y, w, h)` | Bounding box (top-left corner + dimensions) |
| `center` | `(cx, cy)` | Center of the bounding box |
| `confidence` | `float` | Detection confidence in `[0, 1]` |
| `track_id` | `int \| None` | Persistent ByteTrack ID across frames; `None` if tracking failed |

#### `GroundTruthDetector`

```python
detector = GroundTruthDetector(model="yolov8n.pt", confidence_threshold=0.5)
```

| Parameter | Default | Description |
|---|---|---|
| `model` | `"yolov8n.pt"` | YOLO model name or path (e.g. `yolov8s.pt` for higher accuracy) |
| `confidence_threshold` | `0.5` | Minimum confidence; detections below this are discarded |

**`detector.process(frame, normalized=False) -> List[PersonLocation]`**

| Parameter | Type | Description |
|---|---|---|
| `frame` | `np.ndarray (H, W, C)` | RGB or BGR image |
| `normalized` | `bool` | If `True`, coordinates are returned in `[0, 1]` range |

Returns a list of `PersonLocation` objects. ByteTrack persistence is handled internally via `model.track(..., persist=True)`.

### Example

```python
import cv2
from ground_truth import GroundTruthDetector

detector = GroundTruthDetector()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    locations = detector.process(frame, normalized=True)
    for loc in locations:
        print(f"Person at {loc.center} (track_id={loc.track_id}, conf={loc.confidence:.2f})")
```

---

## Data Storage

All capture output is written to:

```
data/logs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── <OBJECT>_data.pkl       # raw sensor records (streaming pickle)
        └── <OBJECT>_data_sync.json # temporal sync index (written by sync_data.py)
```

`OBJECT` is set by `capture_config.py`.

### PKL Record Format

Each `.pkl` file contains a sequence of Python dicts appended sequentially (not a list). The first record is always a metadata header:

```python
{
    "metadata": {
        "object": "test",
        "start_time": "2025-01-01T12:00:00",
        "sensors": {
            "spad": {
                "sensor": "VL53L8CH (SPADMergeWrapper)",
                "resolution": "4x4",
                "ranging_frequency_hz": ...,
                "integration_time_ms": ...,
                "num_bins": ...,
            },
            "realsense": {
                "device_name": "Intel RealSense D435",
                "serial_number": "...",
                "resolution": [848, 480],
                "fps": 30,
                "intrinsics": {
                    "depth": {"ppx", "ppy", "fx", "fy", "model", "coeffs"},
                    "color": {"ppx", "ppy", "fx", "fy", "model", "coeffs"},
                },
            },
        },
    }
}
```

Subsequent records are data frames:

```python
{
    "iter": 0,
    "spad": np.ndarray,             # histogram array from VL53L8CH
    "spad_timestamp": "2025-01-01T12:00:00.123456",
    "realsense": {
        "raw_depth":     np.ndarray,  # uint16, mm, original depth frame
        "raw_rgb":       np.ndarray,  # uint8, BGR, original color frame
        "aligned_depth": np.ndarray,  # uint16, mm, depth aligned to color
        "aligned_rgb":   np.ndarray,  # uint8, BGR, color aligned to depth
    },
    "realsense_timestamp": "2025-01-01T12:00:00.130000",
}
```

### Reading the PKL

```python
import pickle
from pathlib import Path

records = []
with open("data/logs/.../test_data.pkl", "rb") as f:
    try:
        while True:
            records.append(pickle.load(f))
    except EOFError:
        pass

metadata = records[0]["metadata"]
frames = [r for r in records if "iter" in r]
```

---

## General Pipeline

```
┌─────────────────────────────────────────┐
│           capture_config.py             │
│  (sensor toggles, mode, port, FPS, ...) │
└───────────────────┬─────────────────────┘
                    │ imported by
                    ▼
┌─────────────────────────────────────────┐
│            full_capture.py              │
│                                         │
│  ┌────────────┐    ┌──────────────────┐ │
│  │ spad-worker│    │  rs-worker       │ │
│  │ (thread)   │    │  (thread)        │ │
│  └─────┬──────┘    └────────┬─────────┘ │
│        │ LatestFrame        │ LatestFrame│
│        └──────────┬─────────┘           │
│                   │ main thread         │
│            assemble record              │
│                   │                     │
│            PklHandler.append()          │
│                   │                     │
│     data/logs/YYYY-MM-DD/HH-MM-SS/     │
│         <object>_data.pkl               │
└───────────────────┬─────────────────────┘
                    │ post-capture
                    ▼
┌─────────────────────────────────────────┐
│             sync_data.py                │
│                                         │
│  Load PKL → extract timestamps          │
│  → nearest-neighbour match              │
│  → report + write _sync.json            │
└───────────────────┬─────────────────────┘
                    │ downstream use
                    ▼
┌─────────────────────────────────────────┐
│           ground_truth.py               │
│                                         │
│  GroundTruthDetector.process(rgb_frame) │
│  → List[PersonLocation]                 │
│  (bbox, center, confidence, track_id)   │
└─────────────────────────────────────────┘
```

### End-to-End Workflow

1. **Configure** — edit `capture_config.py` (set `OBJECT`, enable sensors, choose capture mode).
2. **Capture** — `python full_capture.py`. Data is written to `data/logs/`.
3. **Synchronize** — `python sync_data.py` (no arguments auto-detects the latest PKL). Produces `*_sync.json`.
4. **Label** — use `GroundTruthDetector` on the `"aligned_rgb"` arrays from matched RealSense frames to obtain person locations as ground truth.

---

## Dependencies

Root-level (install via `pip install -r requirements.txt`):

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `opencv-python` | RealSense preview, image processing |
| `ultralytics` | YOLOv8 person detection |
| `pyrealsense2` | Intel RealSense SDK |

`spad_center/` is a separate package (`cc_hardware`). Install with:

```bash
cd spad_center
pip install -e .
```
