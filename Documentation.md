# NLOS Sensor Fusion — Documentation

## Overview

Fuses SPAD histogram data (VL53L8CH) and RGB-D frames (RealSense) to locate objects hidden from direct view.

**Pipeline:**
```
capture_config.py → full_capture.py → sync_data.py → training/
                                                    → ground_truth.py
```

---

## File Reference

| File/Dir | Role |
|---|---|
| `capture_config.py` | Capture settings (edit before running) |
| `full_capture.py` | Concurrent multi-sensor capture |
| `sync_data.py` | Post-hoc timestamp matching + JSON index |
| `ground_truth.py` | YOLO person detector |
| `training/` | PyTorch training pipeline |
| `train.py` | Training entry point |
| `spad_center/` | `cc_hardware` drivers and utilities |

---

## Configuration — `capture_config.py`

```python
OBJECT = "test"           # output file label

USE_SPAD = True
USE_REALSENSE = True

CAPTURE_MODE = "loop"     # "loop" (Ctrl+C to stop) | "manual" (Enter per frame)

SHOW_SPAD_DASHBOARD = True
SHOW_REALSENSE_PREVIEW = True

SPAD_RESOLUTION = "4x4"  # "4x4" | "8x8"
SPAD_PORT = "COM4"        # "/dev/ttyACM*" on Linux

RS_WIDTH = 848
RS_HEIGHT = 480
RS_FPS = 30
```

---

## Capture — `full_capture.py`

```bash
python full_capture.py
```

Spawns one background thread per sensor. Each thread writes to a `LatestFrame` buffer; the main thread paces on the SPAD and assembles records.

**Output:** `data/logs/YYYY-MM-DD/HH-MM-SS/<OBJECT>_data.pkl`

**PKL record format:**

```python
# Record 0 — metadata
{"metadata": {"object", "start_time", "sensors": {"spad": {...}, "realsense": {...}}}}

# Records 1..N — data frames
{
    "iter": int,
    "spad": {SPADDataType.HISTOGRAM: np.ndarray},  # (H, W, bins)
    "spad_timestamp": "ISO 8601",
    "realsense": {
        "raw_depth":     np.ndarray,   # uint16, mm
        "raw_rgb":       np.ndarray,   # uint8, BGR
        "aligned_depth": np.ndarray,   # uint16, mm
        "aligned_rgb":   np.ndarray,   # uint8, BGR
    },
    "realsense_timestamp": "ISO 8601",
}
```

**Reading the PKL:**

```python
import cloudpickle as pickle

records = []
with open("test_data.pkl", "rb") as f:
    try:
        while True:
            records.append(pickle.load(f))
    except EOFError:
        pass

metadata = records[0]["metadata"]
frames = [r for r in records if "iter" in r]
```

---

## Sync — `sync_data.py`

```bash
python sync_data.py                          # auto-detects latest PKL
python sync_data.py path/to/capture.pkl
python sync_data.py capture.pkl --max-dt-ms 50
python sync_data.py capture.pkl --no-index   # skip writing JSON
```

Nearest-neighbour matches every SPAD frame to the closest camera frame by timestamp. Prints per-pair `|dt|`, summary stats, and flags pairs exceeding `--max-dt-ms` (default 100 ms).

Writes a JSON index alongside the PKL (unless `--no-index`):

```
<OBJECT>_data_sync.json
```

```json
{
  "source_pkl": "/abs/path/to/data.pkl",
  "n_pairs": 249,
  "n_pairs_over_threshold": 2,
  "pairs": [
    {"spad_idx": 0, "cam_idx": 0, "dt_ms": 4.2, "spad_ts": "...", "cam_ts": "..."}
  ]
}
```

The index stores indices and timestamps only — no arrays. Downstream code uses it to load matched pairs on demand from the raw PKL.

**API:**

```python
from sync_data import sync
matches = sync(Path("capture.pkl"), max_dt_ms=50.0, save=True)
# returns list of (spad_idx, cam_idx, dt_seconds)
```

---

## Ground Truth — `ground_truth.py`

YOLOv8 + ByteTrack person detector.

```python
from ground_truth import GroundTruthDetector

detector = GroundTruthDetector(model="yolov8n.pt", confidence_threshold=0.5)
locations = detector.process(frame, normalized=True)
# returns List[PersonLocation(bbox, center, confidence, track_id)]
```

Apply to `record["realsense"]["aligned_rgb"]` from matched pairs to generate labels.

---

## Training Pipeline — `training/`

PyTorch pipeline using Weights & Biases. Reads directly from raw PKL files via byte-offset indexing — no duplication of sensor arrays.

### Package Structure

| Module | Role |
|---|---|
| `training/config.py` | `DataConfig`, `TrainConfig`, `WandbConfig`, `Config` dataclasses |
| `training/dataset.py` | `NLOSDataset` — random-access PKL reads, tensor extraction |
| `training/datamodule.py` | `NLOSDataModule` — multi-capture support, train/val/test splits |
| `training/model.py` | `NLOSModel` (abstract), `ModelOutput` dataclass |
| `training/trainer.py` | Train/val loop, W&B logging, checkpointing, resume |

### Usage

```bash
python train.py --config path/to/config.json
python train.py --config config.json --resume checkpoints/last.pt
```

### Config

```python
from training import Config, DataConfig, TrainConfig, WandbConfig

config = Config(
    data=DataConfig(
        index_files=["data/logs/.../test_data_sync.json"],
        max_dt_ms=100.0,
        train_frac=0.70,
        val_frac=0.15,
        batch_size=16,
    ),
    train=TrainConfig(
        max_epochs=100,
        lr=1e-3,
        optimizer="adamw",   # "adamw" | "adam" | "sgd"
        scheduler="cosine",  # "cosine" | "step" | "none"
        checkpoint_dir="checkpoints/",
    ),
    wandb=WandbConfig(project="nlos-fusion"),
)
config.to_json("config.json")
```

### Implementing a Model

Subclass `NLOSModel` and implement three methods:

```python
from training import NLOSModel, ModelOutput
import torch

class MyModel(NLOSModel):
    def forward(self, spad, rgb, depth) -> ModelOutput:
        # spad:  (B, H, W, bins) float32
        # rgb:   (B, 3, H, W)    float32  [0, 1]
        # depth: (B, H, W)       float32  metres
        pred = ...
        return ModelOutput(prediction=pred)

    def loss(self, output: ModelOutput, batch: dict) -> torch.Tensor:
        return torch.nn.functional.mse_loss(output.prediction, batch["label"])

    def metrics(self, output: ModelOutput, batch: dict) -> dict[str, float]:
        return {"mae": (output.prediction - batch["label"]).abs().mean().item()}
```

Then train:

```python
from training import train
trainer = train(config, MyModel())
```

### Dataset Tensors

| Key | Shape | dtype | Notes |
|---|---|---|---|
| `spad` | `(B, H, W, bins)` | float32 | raw histogram counts |
| `rgb` | `(B, 3, H, W)` | float32 | RGB, [0, 1] |
| `depth` | `(B, H, W)` | float32 | metres; 0 = invalid |

---

## Dependencies

```bash
pip install -r requirements.txt   # numpy, opencv-python, ultralytics, pyrealsense2, wandb
pip install -e spad_center/        # cc_hardware drivers
```
