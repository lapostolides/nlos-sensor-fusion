"""
_diagnose_overhead_fov.py
Queries the eMeet C960's current and supported resolutions,
and checks UVC zoom/pan/tilt controls.
"""
import sys
sys.path.insert(0, ".")
from cameras import find_usb_camera_index
import cv2

CAM_NAME = "eMeet C960"

idx = find_usb_camera_index(CAM_NAME)
print(f"eMeet C960 -> OpenCV index {idx}\n")

cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
if not cap.isOpened():
    print("ERROR: Could not open camera")
    raise SystemExit(1)

# Current default resolution
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Default resolution : {w}x{h} @ {fps:.0f}fps")

# UVC controls
props = {
    "ZOOM"        : cv2.CAP_PROP_ZOOM,
    "FOCUS"       : cv2.CAP_PROP_FOCUS,
    "PAN"         : cv2.CAP_PROP_PAN,
    "TILT"        : cv2.CAP_PROP_TILT,
    "AUTOFOCUS"   : cv2.CAP_PROP_AUTOFOCUS,
}
print("\nUVC controls (−1 = not supported):")
for name, prop in props.items():
    val = cap.get(prop)
    print(f"  {name:<12}: {val}")

# Try forcing 1920x1080 and see if the camera accepts it
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
w2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps2 = cap.get(cv2.CAP_PROP_FPS)
print(f"\nAfter requesting 1920x1080 : {w2}x{h2} @ {fps2:.0f}fps")

# Try forcing zoom to minimum (0)
zoom_before = cap.get(cv2.CAP_PROP_ZOOM)
if zoom_before >= 0:
    cap.set(cv2.CAP_PROP_ZOOM, 0)
    zoom_after = cap.get(cv2.CAP_PROP_ZOOM)
    print(f"Zoom: {zoom_before} -> {zoom_after}")

cap.release()
print("\nDone.")
