# capture_config.py - Edit this file to configure full_capture.py
# All sensor toggles, hardware config, and preview settings live here.
# No CLI args needed — just edit this file and run: python full_capture.py

OBJECT = "test"

# ── Sensor Toggles ─────────────────────────────────────────────────────
USE_SPAD         = True
USE_SENSOR_CAM   = True   # Intel RealSense RGB-D
USE_OVERHEAD_CAM = True   # eMeet C960 USB webcam (RGB only, no depth)
USE_UWB          = True   # DWM1001-DEV boards (1 TX + up to 3 RX)

# ── Capture Mode ───────────────────────────────────────────────────────
# "loop"   : continuous capture (~sensor fps), Ctrl+C to stop
# "manual" : Enter per frame, q to quit
CAPTURE_MODE = "loop"

# ── Live Preview ───────────────────────────────────────────────────────
SHOW_SPAD_DASHBOARD       = True   # PyQtGraph histogram dashboard
SHOW_SENSOR_CAM_PREVIEW   = True   # OpenCV RGB + depth preview window
SHOW_OVERHEAD_CAM_PREVIEW = True   # OpenCV RGB preview window for eMeet C960
SHOW_UWB_CIR_PREVIEW      = True   # Live matplotlib CIR plot (3 RX subplots)

# ── SPAD (VL53L8CH) ───────────────────────────────────────────────────
SPAD_RESOLUTION = "4x4"   # "8x8" or "4x4"
SPAD_PORT       = None     # Auto-detect by USB VID (STMicro). Set e.g. "COM4" to override.

# ── sensor_cam (RealSense RGB-D) ───────────────────────────────────────
SENSOR_CAM_WIDTH  = 848
SENSOR_CAM_HEIGHT = 480
SENSOR_CAM_FPS    = 30

# ── overhead_cam (eMeet C960 USB webcam) ──────────────────────────────
# Identified by Windows device name — plug order does not matter.
# Change this string if a different USB camera is used.
OVERHEAD_CAM_NAME   = "eMeet C960"
OVERHEAD_CAM_WIDTH  = 1920   # 1920x1080 uses the full sensor; default was 640x480
OVERHEAD_CAM_HEIGHT = 1080

# ── UWB (DWM1001-DEV) ────────────────────────────────────────────────
# Board roles are mapped via device_roles.json (see get_device_id.py).
UWB_BAUD = 115200
