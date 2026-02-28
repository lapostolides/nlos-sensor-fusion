# capture_config.py - Edit this file to configure full_capture.py

OBJECT = "test"

# ── Sensor Toggles ─────────────────────────────────────────────────────
USE_SPAD = True
USE_REALSENSE = False
# USE_THERMAL = False  # future
# USE_UWB = False      # future

# ── Capture Mode ───────────────────────────────────────────────────────
# "loop"   : continuous capture (~sensor fps), Ctrl+C to stop
# "manual" : Enter per frame, q to quit
CAPTURE_MODE = "loop"

# ── Live Preview ───────────────────────────────────────────────────────
SHOW_SPAD_DASHBOARD = True    # PyQtGraph histogram dashboard
SHOW_REALSENSE_PREVIEW = False # OpenCV RGB + depth preview window

# ── SPAD (VL53L8CH) ───────────────────────────────────────────────────
SPAD_RESOLUTION = "4x4"  # "8x8" or "4x4"

# ── RealSense RGB-D ───────────────────────────────────────────────────
RS_WIDTH = 848
RS_HEIGHT = 480
RS_FPS = 30
