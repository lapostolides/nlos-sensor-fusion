# nlos-sensor-fusion

NLOS (Non-Line-of-Sight) sensor fusion — combining SPAD histogram data with RGB-D camera data to locate objects hidden from direct view.

## Structure

| Path | Description |
|------|-------------|
| `full_capture.py` | Unified multi-sensor capture script (SPAD, RealSense RGB-D) |
| `capture_config.py` | User-editable config for `full_capture.py` (sensor toggles, resolution, capture mode) |
| `ground_truth.py` | YOLO-based person detection for ground truth labeling |
| `spad_center/` | Hardware driver package (`cc_hardware`) for SPAD sensors, cameras, and stepper motors |
| `colmap/` | COLMAP 3D reconstruction scripts |
| `data/logs/` | Capture output (`.pkl` files, timestamped per run) |

## Quick Start

```bash
pip install -r requirements.txt
cd spad_center && pip install -e .
```

Edit `capture_config.py`, then:

```bash
python full_capture.py
```
