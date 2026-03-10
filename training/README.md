# Training Pipeline

PyTorch training pipeline for NLOS (Non-Line-of-Sight) human localisation and pose classification using fused multi-sensor data (SPAD histograms, RGB-D, UWB CIR).

## Quick Start

```bash
# 1. Add your run directories to a config file
#    Each run dir must contain sync.json (from sync_data.py) and gt.json

# 2. Train
python train.py --config configs/default.json

# 3. Resume from checkpoint
python train.py --config configs/default.json --resume checkpoints/last.pt
```

## Architecture Overview

```
                ┌──────────────┐
  SPAD (H,W,bins) │ SPADEncoder  │──┐
                └──────────────┘  │
                ┌──────────────┐  │   ┌────────────┐   ┌──────────────────┐
  RGB-D (4,H,W)   │ RGBDEncoder  │──┼──│ FusionNeck │──│ LocalizationHead │→ (cx, cy)
                └──────────────┘  │   └────────────┘   └──────────────────┘
                ┌──────────────┐  │         │          ┌──────────────────┐
  UWB CIR (3,1016)│ UWBEncoder   │──┘         └─────────│    PoseHead     │→ class logits
                └──────────────┘                       └──────────────────┘
```

Each encoder produces a `(B, embed_dim)` vector. The fusion neck concatenates them and projects to `(B, fused_dim)`. Task heads predict from the fused representation.

## Module Reference

| File | Class / Purpose |
|------|----------------|
| `config.py` | `Config`, `DataConfig`, `ModelConfig`, `TrainConfig`, `WandbConfig` — all hyperparameters as dataclasses, serializable to/from JSON |
| `dataset.py` | `NLOSDataset` — PyTorch Dataset reading synced multi-sensor frames from run directories |
| `datamodule.py` | `NLOSDataModule` — loads sync indices, filters pairs, splits train/val/test, vends DataLoaders |
| `encoders.py` | `SPADEncoder`, `RGBDEncoder`, `UWBEncoder`, `MmWaveEncoder` (placeholder) — per-sensor feature extractors |
| `fusion.py` | `FusionNeck`, `NLOSFusionModel` — concatenation-based fusion and the concrete model that wires everything together |
| `heads.py` | `LocalizationHead`, `PoseHead` — MLP prediction heads |
| `model.py` | `NLOSModel` (abstract base), `ModelOutput` — interface contract for the Trainer |
| `trainer.py` | `Trainer` — training loop with W&B logging, checkpointing, optimizer/scheduler management |

## Configuration

Configs are plain JSON files stored in `configs/`. Any field not specified falls back to the dataclass default. Provided configs:

| Config | Description |
|--------|-------------|
| `configs/default.json` | All sensors enabled, full hyperparameter set |
| `configs/full_fusion.json` | Same as default (all sensors) |
| `configs/spad_only.json` | SPAD-only ablation, no pose head |
| `configs/rgbd_only.json` | RGB-D-only ablation |
| `configs/uwb_only.json` | UWB-only ablation |

### Config Sections

**`data`** — Dataset and loading

| Field | Default | Description |
|-------|---------|-------------|
| `run_dirs` | `[]` | List of paths to run directories (must contain `sync.json`) |
| `max_dt_ms` | `100.0` | Max time offset (ms) between synced sensors; pairs above this are excluded |
| `train_frac` | `0.70` | Fraction of pairs for training |
| `val_frac` | `0.15` | Fraction for validation (test = remainder) |
| `seed` | `42` | Random seed for shuffling and splitting |
| `batch_size` | `16` | DataLoader batch size |
| `num_workers` | `4` | DataLoader worker processes |
| `require_gt` | `true` | Only include pairs with ground-truth detections |
| `gt_camera` | `"overhead_cam"` | Camera used for ground-truth person detection |

**`model`** — Architecture and loss

| Field | Default | Description |
|-------|---------|-------------|
| `use_spad` | `true` | Enable SPAD encoder |
| `use_rgbd` | `true` | Enable RGB-D encoder |
| `use_uwb` | `true` | Enable UWB encoder |
| `use_mmwave` | `false` | Enable mmWave encoder (not yet implemented) |
| `embed_dim` | `128` | Per-sensor embedding dimension |
| `fused_dim` | `256` | Post-fusion dimension |
| `pretrained_backbone` | `true` | Use ImageNet-pretrained ResNet-18 for RGB-D |
| `spad_bins` | `8` | Number of SPAD histogram bins |
| `spad_spatial` | `4` | SPAD spatial resolution (4 or 8) |
| `uwb_n_receivers` | `3` | Number of UWB receivers |
| `uwb_n_samples` | `1016` | CIR samples per frame (fixed by DW1000 hardware) |
| `use_pose_head` | `true` | Enable pose classification head |
| `n_pose_classes` | `5` | Number of pose classes (STILL, LOCOMOTION, CROUCHING, ARMS_RAISED, UNKNOWN) |
| `loc_loss_weight` | `1.0` | Weight for localisation loss (Smooth L1) |
| `pose_loss_weight` | `0.1` | Weight for pose classification loss (cross-entropy) |

**`train`** — Training loop

| Field | Default | Description |
|-------|---------|-------------|
| `max_epochs` | `100` | Total training epochs |
| `lr` | `1e-3` | Learning rate |
| `weight_decay` | `1e-4` | AdamW weight decay |
| `optimizer` | `"adamw"` | Optimizer: `"adamw"`, `"adam"`, or `"sgd"` |
| `scheduler` | `"cosine"` | LR scheduler: `"cosine"`, `"step"`, or `"none"` |
| `grad_clip` | `1.0` | Max gradient norm (0 = disabled) |
| `log_every_n_steps` | `10` | W&B step-level logging interval |
| `val_every_n_epochs` | `1` | Validation frequency |
| `checkpoint_dir` | `"checkpoints/"` | Directory for saved checkpoints |
| `resume_from` | `null` | Path to `.pt` checkpoint to resume from |

**`wandb`** — Weights & Biases

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable W&B logging |
| `project` | `"nlos-fusion"` | W&B project name |
| `entity` | `""` | W&B entity (team/user), empty = default |
| `run_name` | `""` | Custom run name, empty = auto-generated |
| `tags` | `[]` | Tags for filtering runs |
| `log_media_every_n_epochs` | `10` | Log sample batch as images (0 = disabled) |

## Data Requirements

Each run directory must contain:

```
<run_dir>/
├── manifest.json           # capture metadata
├── sync.json               # temporal alignment (from sync_data.py)
├── gt.json                 # ground-truth detections (from ground_truth.py)
├── spad.npz                # SPAD histograms, key "histograms" → (N, H, W, bins)
├── sensor_cam/
│   ├── rgb/000000.jpg      # RGB frames (BGR uint8)
│   └── depth/000000.png    # depth maps (uint16, millimetres)
├── overhead_cam/
│   └── 000000.jpg          # overhead camera frames
└── rx1.npz, rx2.npz, rx3.npz  # UWB CIR per receiver
```

**Preprocessing steps** before training:
1. Run `sync_data.py <run_dir>` to produce `sync.json`
2. Run ground-truth detection to produce `gt.json`
3. Add the run directory path to your config's `data.run_dirs` list

## Dataset Output Format

Each sample from `NLOSDataset` is a dict with:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `spad` | `(H, W, bins)` | float32 | Raw SPAD histogram counts |
| `rgb` | `(3, H, W)` | float32 | RGB image in [0, 1] |
| `depth` | `(H, W)` | float32 | Depth in metres (0 = invalid) |
| `uwb_cir` | `(3, 1016)` | float32 | CIR magnitude per receiver (0 if missing) |
| `uwb_fp_index` | `(3,)` | float32 | First-path index normalised to [0, 1] |
| `uwb_mask` | `(3,)` | bool | Which receivers have valid data |
| `gt_location` | `(2,)` | float32 | Normalised (cx, cy) target in [0, 1] |
| `gt_pose` | scalar | long | Pose class index (-1 if missing) |
| `gt_valid` | scalar | bool | Whether ground-truth label exists |
| `meta` | dict | — | `dt_ms`, `spad_idx`, `cam_idx`, `run_dir` |

## Encoders

### SPADEncoder
2-D CNN treating histogram bins as channels over the spatial grid (4x4 or 8x8). Uses `AdaptiveAvgPool2d` for resolution-agnostic pooling.

### RGBDEncoder
ResNet-18 backbone with 4 input channels (RGB + depth). When `pretrained=True`, the depth channel weight is initialised from the mean of the RGB weights. Classification head replaced with a linear projection.

### UWBEncoder
1-D CNN over the CIR waveform (receivers as channels) with progressive stride reduction. First-path index is projected separately via a small MLP and concatenated before the final projection.

### MmWaveEncoder
Placeholder for future mmWave 4-D point cloud data. Not yet implemented.

## Fusion

`FusionNeck` concatenates all active sensor embeddings and projects through a two-layer MLP with dropout. When only one sensor is active (ablation mode), a simple linear projection is used instead.

## Losses and Metrics

**Losses:**
- Localisation: Smooth L1 loss between predicted and ground-truth normalised coordinates
- Pose: Cross-entropy loss with `ignore_index=-1` for missing labels

**Metrics (logged to W&B):**
- `loc_mae`: Mean absolute error of location predictions
- `loc_dist`: Euclidean distance error (in normalised coordinates)
- `pose_acc`: Pose classification accuracy (only computed on valid labels)

## Checkpoints

The trainer saves two checkpoints to `checkpoint_dir`:
- `last.pt` — saved every epoch
- `best.pt` — saved when validation loss improves

Checkpoint contents:
```python
{
    "epoch": int,
    "global_step": int,
    "best_val_loss": float,
    "model": state_dict,
    "optimizer": state_dict,
    "scheduler": state_dict | None,
    "config": dict,  # full config for reproducibility
}
```

## Ablation Studies

Use the provided single-sensor configs to run ablation experiments:

```bash
# SPAD only
python train.py --config configs/spad_only.json

# RGB-D only
python train.py --config configs/rgbd_only.json

# UWB only
python train.py --config configs/uwb_only.json

# Full fusion (all sensors)
python train.py --config configs/full_fusion.json
```

Each config disables the irrelevant sensor encoders and adjusts the fusion neck accordingly (single-sensor mode uses a simple linear projection instead of concatenation).
