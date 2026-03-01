"""cameras.py — RGB / RGB-D camera wrappers for full_capture.py.

Provides a common RGBDCamera interface so full_capture.py can treat any camera
uniformly. Add a new camera type by subclassing RGBDCamera.

Supported cameras
-----------------
RealsenseCameraWrapper   Intel RealSense (depth + colour, aligned)   → sensor_cam
USBCameraWrapper         Generic USB webcam via OpenCV, e.g. eMeet C960 (colour only) → overhead_cam

Device identification
---------------------
* RealSense: identified by pyrealsense2, which only sees RealSense hardware.
  No USB webcam can appear here regardless of plug order.
* USB webcam: identified by calling Windows MFEnumDeviceSources via ctypes,
  which is the exact same API call OpenCV makes internally when assigning
  CAP_MSMF indices — so the index is guaranteed to match.
"""

from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from ctypes import POINTER, byref, c_uint32, c_void_p, c_wchar_p
from typing import TypedDict

import numpy as np


# ---------------------------------------------------------------------------
# Frame type
# ---------------------------------------------------------------------------

class CameraFrame(TypedDict, total=False):
    """Dict returned by RGBDCamera.get_frame().

    'raw_rgb' is always present.  Depth keys are only included for cameras
    that have a depth sensor (e.g. RealSense).
    """
    raw_rgb: np.ndarray        # HxWx3 BGR
    raw_depth: np.ndarray      # HxW uint16, millimetres
    aligned_rgb: np.ndarray    # colour frame aligned to depth viewport
    aligned_depth: np.ndarray  # depth frame aligned to colour viewport


# ---------------------------------------------------------------------------
# Windows MF camera enumeration — same API OpenCV uses for CAP_MSMF indices
# ---------------------------------------------------------------------------

class _GUID(ctypes.Structure):
    _fields_ = [
        ("d1", ctypes.c_uint32),
        ("d2", ctypes.c_uint16),
        ("d3", ctypes.c_uint16),
        ("d4", ctypes.c_uint8 * 8),
    ]

def _guid(a: int, b: int, c: int, *d: int) -> _GUID:
    return _GUID(a, b, c, (_GUID._fields_[3][1])(*d))  # type: ignore[attr-defined]

# GUIDs from Windows SDK mfidl.h / mfapi.h
_MF_DEVSOURCE_SOURCE_TYPE     = _guid(0xC60AC5FE, 0x252A, 0x478F, 0xA0, 0xEF, 0xBC, 0x8F, 0xA5, 0xF7, 0xCA, 0xD3)
_MF_DEVSOURCE_SOURCE_VIDCAP   = _guid(0x8AC3587A, 0x4AE7, 0x42D8, 0x99, 0xE0, 0x0A, 0x60, 0x13, 0xEE, 0xF9, 0x0F)
_MF_DEVSOURCE_FRIENDLY_NAME   = _guid(0x60D0E559, 0x52F8, 0x4FA2, 0xBB, 0xCE, 0xAC, 0xDB, 0x34, 0xA8, 0xEC, 0x01)

# IMFAttributes vtable indices (IUnknown=0-2, IMFAttributes=3-32)
_VTBL_RELEASE           = 2   # IUnknown::Release
_VTBL_SET_GUID          = 24  # IMFAttributes::SetGUID
_VTBL_GET_ALLOC_STRING  = 13  # IMFAttributes::GetAllocatedString


def _enumerate_mf_video_cameras() -> list[str]:
    """Return friendly names of MSMF video-capture devices in the exact order
    that OpenCV CAP_MSMF assigns its integer indices.

    Uses Windows MFEnumDeviceSources via ctypes — the same API call that
    OpenCV makes internally, so index N here == cv2.VideoCapture(N, CAP_MSMF).

    Runs in an isolated worker thread so that CoInitializeEx / MFStartup do
    not alter the calling thread's COM state (OpenCV's MSMF backend also
    calls CoInitializeEx on the main thread and must find it uninitialised).

    Returns an empty list if MF is unavailable (non-Windows).
    """
    import threading as _threading

    names: list[str] = []

    def _worker() -> None:
        try:
            ole32  = ctypes.windll.ole32   # type: ignore[attr-defined]
            mfplat = ctypes.windll.mfplat  # type: ignore[attr-defined]
            mf     = ctypes.windll.mf      # type: ignore[attr-defined]
        except AttributeError:
            return  # not Windows

        ole32.CoInitializeEx(None, 0)
        mfplat.MFStartup(0x00020070, 0)  # MF_VERSION = 2.0

        p_attr = c_void_p()
        if mfplat.MFCreateAttributes(byref(p_attr), 1):
            return

        vtbl = ctypes.cast(ctypes.cast(p_attr, POINTER(c_void_p))[0], POINTER(c_void_p))
        SetGUID = ctypes.CFUNCTYPE(ctypes.c_long, c_void_p, POINTER(_GUID), POINTER(_GUID))(vtbl[_VTBL_SET_GUID])
        if SetGUID(p_attr, byref(_MF_DEVSOURCE_SOURCE_TYPE), byref(_MF_DEVSOURCE_SOURCE_VIDCAP)):
            return

        pp_devices = c_void_p()
        c_count = c_uint32()
        mf.MFEnumDeviceSources.restype = ctypes.c_long
        if mf.MFEnumDeviceSources(p_attr, byref(pp_devices), byref(c_count)):
            return

        count = c_count.value
        if count == 0 or not pp_devices:
            return

        arr = ctypes.cast(pp_devices, POINTER(c_void_p * count)).contents
        for i, ptr in enumerate(arr):
            if not ptr:
                names.append(f"<null:{i}>")
                continue
            vtbl_i = ctypes.cast(ctypes.cast(ptr, POINTER(c_void_p))[0], POINTER(c_void_p))
            GetStr = ctypes.CFUNCTYPE(
                ctypes.c_long, c_void_p, POINTER(_GUID), POINTER(c_wchar_p), POINTER(c_uint32)
            )(vtbl_i[_VTBL_GET_ALLOC_STRING])
            pwsz = c_wchar_p()
            cch  = c_uint32()
            hr   = GetStr(ptr, byref(_MF_DEVSOURCE_FRIENDLY_NAME), byref(pwsz), byref(cch))
            names.append(pwsz.value if hr == 0 and pwsz.value else f"<unknown:{i}>")
            if pwsz:
                ole32.CoTaskMemFree(pwsz)
            ctypes.CFUNCTYPE(ctypes.c_ulong, c_void_p)(vtbl_i[_VTBL_RELEASE])(ptr)

        ole32.CoTaskMemFree(pp_devices)

    t = _threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=10.0)
    return names


def find_usb_camera_index(name_pattern: str) -> int:
    """Return the OpenCV MSMF index of the first camera whose name contains *name_pattern*.

    Uses MFEnumDeviceSources — the same enumeration OpenCV uses — so the
    returned index is guaranteed to match cv2.VideoCapture(index, CAP_MSMF).
    Raises RuntimeError if no matching camera is found, listing what is available.
    """
    cameras = _enumerate_mf_video_cameras()
    for i, name in enumerate(cameras):
        if name_pattern.lower() in name.lower():
            return i
    raise RuntimeError(
        f"Camera matching '{name_pattern}' not found on this machine.\n"
        f"MF-enumerated cameras: {cameras}\n"
        "Check that the camera is plugged in and Windows recognises it."
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class RGBDCamera(ABC):
    """Minimal camera interface consumed by full_capture.py worker threads.

    Lifecycle::

        cam = MyCamera(...)
        metadata = cam.start()      # open hardware, returns PKL-header dict
        frame = cam.get_frame()     # call in a loop / worker thread
        cam.close()                 # release hardware

    Cameras should be added to the Manager dict for bookkeeping
    (``manager.add(cam_key=cam)``), but cleanup must be done explicitly via
    ``close()`` because Manager only auto-closes cc_hardware Component objects.
    """

    @abstractmethod
    def start(self) -> dict:
        """Open hardware. Returns a metadata dict stored in the PKL header."""
        ...

    @abstractmethod
    def get_frame(self, timeout_ms: float = 500) -> CameraFrame | None:
        """Return the latest frame, or None on timeout / failure.

        Must be safe to call from a background thread.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all hardware resources. Safe to call more than once."""
        ...

    @property
    @abstractmethod
    def is_okay(self) -> bool:
        """True if the camera is open and operational."""
        ...

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Intel RealSense RGB-D  (sensor_cam)
# ---------------------------------------------------------------------------

class RealsenseCameraWrapper(RGBDCamera):
    """Intel RealSense RGB-D camera.

    Identified exclusively by pyrealsense2, which only enumerates RealSense
    hardware over its own USB protocol.  A USB webcam can never appear here
    regardless of plug order — identification is guaranteed.

    Streams colour + depth and returns depth-to-colour aligned frames.

    Parameters
    ----------
    width, height, fps:
        Stream resolution and frame rate.
    serial:
        Device serial number string.  Pass None to use the first connected
        device (safe when only one RealSense is attached).
    """

    def __init__(
        self,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
        serial: str | None = None,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._serial = serial
        self._pipeline = None
        self._align = None
        self._ok = False

    def start(self) -> dict:
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        if self._serial:
            cfg.enable_device(self._serial)
        cfg.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)
        cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)

        profile = self._pipeline.start(cfg)
        self._align = rs.align(rs.stream.color)
        self._ok = True

        dev = profile.get_device()
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)

        def _intr(stream):
            i = profile.get_stream(stream).as_video_stream_profile().get_intrinsics()
            return {"ppx": i.ppx, "ppy": i.ppy, "fx": i.fx, "fy": i.fy,
                    "model": str(i.model), "coeffs": i.coeffs}

        print(f"\033[1;34m[sensor_cam] RealSense connected: {name} (S/N: {serial})\033[0m")
        return {
            "type": "realsense",
            "device_name": name,
            "serial_number": serial,
            "resolution": [self._width, self._height],
            "fps": self._fps,
            "intrinsics": {
                "depth": _intr(rs.stream.depth),
                "color": _intr(rs.stream.color),
            },
        }

    def get_frame(self, timeout_ms: float = 500) -> CameraFrame | None:
        if not self._ok:
            return None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=int(timeout_ms))
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                return None

            aligned = self._align.process(frames)
            ad = aligned.get_depth_frame()
            ac = aligned.get_color_frame()
            if not ad or not ac:
                return None

            # .copy() is critical: the SDK reuses frame memory on the next call
            return {
                "raw_depth":     np.asanyarray(depth_frame.get_data()).copy(),
                "raw_rgb":       np.asanyarray(color_frame.get_data()).copy(),
                "aligned_depth": np.asanyarray(ad.get_data()).copy(),
                "aligned_rgb":   np.asanyarray(ac.get_data()).copy(),
            }
        except Exception:
            return None

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._ok = False

    @property
    def is_okay(self) -> bool:
        return self._ok


# ---------------------------------------------------------------------------
# Generic USB webcam (colour only)  (overhead_cam — eMeet C960)
# ---------------------------------------------------------------------------

class USBCameraWrapper(RGBDCamera):
    """OpenCV MediaFoundation RGB-only wrapper for a USB webcam.

    Tested with the eMeet C960 (VID_0BDA:PID_5876).
    Returns CameraFrame with only 'raw_rgb' — no depth stream.

    Device identification
    ---------------------
    Pass ``name_pattern`` (recommended) to auto-detect the correct OpenCV index
    via Windows WMI enumeration, so plug order cannot cause mix-ups.  The
    detected index is printed at startup so it can be verified.

    Alternatively pass an explicit ``index`` as a fallback (e.g. on non-Windows
    systems where WMI is unavailable).

    Parameters
    ----------
    name_pattern:
        Substring to search for in the Windows device name (case-insensitive).
        Example: ``"eMeet C960"``.  Set to None to use ``index`` directly.
    index:
        Fallback OpenCV index used when ``name_pattern`` is None.
    width, height, fps:
        Requested resolution / frame rate. Pass None to keep camera defaults.
    """

    def __init__(
        self,
        name_pattern: str | None = "eMeet C960",
        index: int = 1,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> None:
        self._name_pattern = name_pattern
        self._fallback_index = index
        self._req_width = width
        self._req_height = height
        self._req_fps = fps
        self._cap = None
        self._ok = False

    def start(self) -> dict:
        import cv2

        # Resolve the correct OpenCV index by name (preferred) or fall back to
        # the configured integer index.
        if self._name_pattern is not None:
            resolved_index = find_usb_camera_index(self._name_pattern)
            print(
                f"\033[1;34m[overhead_cam] WMI matched '{self._name_pattern}' "
                f"→ OpenCV index {resolved_index}\033[0m"
            )
        else:
            resolved_index = self._fallback_index
            print(
                f"\033[1;33m[overhead_cam] No name_pattern set, using index "
                f"{resolved_index} directly (plug-order sensitive)\033[0m"
            )

        self._cap = cv2.VideoCapture(resolved_index, cv2.CAP_MSMF)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"[overhead_cam] Cannot open camera at index {resolved_index}. "
                f"name_pattern='{self._name_pattern}'. "
                "Check that the camera is connected."
            )

        if self._req_width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._req_width)
        if self._req_height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._req_height)
        if self._req_fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, self._req_fps)

        w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._ok = True

        print(
            f"\033[1;34m[overhead_cam] eMeet C960 opened: "
            f"{w}x{h} @ {fps:.0f}fps\033[0m"
        )
        return {
            "type": "usb",
            "model": "eMeet C960",
            "opencv_index": resolved_index,
            "name_pattern": self._name_pattern,
            "resolution": [w, h],
            "fps": fps,
        }

    def get_frame(self, timeout_ms: float = 500) -> CameraFrame | None:  # noqa: ARG002
        if not self._ok or self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return {"raw_rgb": frame.copy()}

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._ok = False

    @property
    def is_okay(self) -> bool:
        return self._ok and self._cap is not None and self._cap.isOpened()
