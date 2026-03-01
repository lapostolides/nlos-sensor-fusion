#!/usr/bin/env python3
"""
full_capture.py - Unified multi-sensor capture for NLOS fusion.
Sensors: SPAD (VL53L8CH), sensor_cam (RealSense RGB-D), overhead_cam (eMeet C960).
Future: Thermal, UWB.

Each sensor runs in a dedicated background thread to avoid frame-rate drops.
The main thread handles Qt/OpenCV display and file I/O.

Sensor identification guarantees
---------------------------------
SPAD           : COM port is explicit (cfg.SPAD_PORT) — no ambiguity.
sensor_cam     : pyrealsense2 only enumerates RealSense hardware; a USB webcam
                 can never appear here regardless of plug order.
overhead_cam   : USBCameraWrapper resolves the OpenCV index by Windows device
                 name via WMI before opening, so plug order does not matter.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HYDRA_HYDRA_LOGGING__FILE"] = "false"
os.environ["HYDRA_JOB_LOGGING__FILE"] = "false"

import time
import threading
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from cc_hardware.drivers.spads import SPADSensor
from cc_hardware.drivers.spads.spad_wrappers import SPADMergeWrapperConfig
from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHConfig8x8, VL53L8CHConfig4x4
from cc_hardware.tools.dashboard.spad_dashboard.pyqtgraph import PyQtGraphDashboardConfig
from cc_hardware.utils.file_handlers import PklHandler
from cc_hardware.utils.manager import Manager

from cameras import RGBDCamera, RealsenseCameraWrapper, USBCameraWrapper
import capture_config as cfg

NOW = datetime.now()
LOGDIR = Path("data/logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S")

SPAD_CONFIGS       = {"8x8": VL53L8CHConfig8x8, "4x4": VL53L8CHConfig4x4}
SENSOR_CAM_WINDOW  = "sensor_cam  RealSense RGB+Depth"
OVERHEAD_CAM_WINDOW = "overhead_cam  eMeet C960 RGB"


# ── Thread-safe frame buffer ──────────────────────────────────────────

class LatestFrame:
    """Single-slot buffer holding the most recent frame from a sensor.

    Workers call put() from their thread; the main thread calls get() or
    wait_for_new() to consume frames.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data = None
        self._timestamp: str | None = None
        self._has_new = threading.Event()

    def put(self, data):
        """Store a new frame (called from worker thread)."""
        with self._lock:
            self._data = data
            self._timestamp = datetime.now().isoformat()
        self._has_new.set()

    def get(self):
        """Non-blocking read of the latest frame. Returns (data, timestamp)."""
        with self._lock:
            return self._data, self._timestamp

    def wait_for_new(self, timeout: float = 2.0):
        """Block until a new frame arrives, then return (data, timestamp).
        Returns (None, None) on timeout."""
        if not self._has_new.wait(timeout=timeout):
            return None, None
        self._has_new.clear()
        with self._lock:
            return self._data, self._timestamp


# ── Sensor Setup ───────────────────────────────────────────────────────

def setup_spad(manager: Manager) -> dict:
    """Init VL53L8CH SPAD sensor + optional live dashboard. Returns metadata."""
    wrapped_config = SPAD_CONFIGS[cfg.SPAD_RESOLUTION].create(port=cfg.SPAD_PORT)
    config = SPADMergeWrapperConfig.create(
        wrapped=wrapped_config,
        data_type="HISTOGRAM",
    )
    sensor = SPADSensor.create_from_config(config)
    assert sensor.is_okay, "SPAD sensor failed to initialize"
    manager.add(spad=sensor)

    if cfg.SHOW_SPAD_DASHBOARD:
        dash_cfg = PyQtGraphDashboardConfig.create()
        dashboard = dash_cfg.create_from_registry(config=dash_cfg, sensor=sensor)
        dashboard.setup()
        manager.add(dashboard=dashboard)

        w = dashboard.win
        screen = w.screen().geometry()
        w.move(screen.width() - w.width() - 10, 10)
        w.show()

    return {
        "sensor": "VL53L8CH (SPADMergeWrapper)",
        "resolution": f"{wrapped_config.width}x{wrapped_config.height}",
        "ranging_frequency_hz": wrapped_config.ranging_frequency_hz,
        "integration_time_ms": wrapped_config.integration_time_ms,
        "num_bins": wrapped_config.num_bins,
    }


def _setup_camera(cam: RGBDCamera, manager: Manager, key: str) -> dict:
    """Open *cam*, register it in *manager* under *key*, return metadata."""
    metadata = cam.start()
    assert cam.is_okay, f"Camera '{key}' failed to initialize"
    manager.add(**{key: cam})
    return metadata


# ── Qt Helper ──────────────────────────────────────────────────────────

def pump_qt():
    """Process pending Qt events so the dashboard stays responsive."""
    try:
        from pyqtgraph.Qt import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app:
            app.processEvents()
    except Exception:
        pass


# ── Worker Threads ─────────────────────────────────────────────────────

def _spad_worker(sensor, buf: LatestFrame, stop: threading.Event):
    """Continuously accumulate SPAD frames into *buf* until *stop* is set."""
    while not stop.is_set():
        try:
            buf.put(sensor.accumulate())
        except Exception as e:
            if not stop.is_set():
                print(f"  \033[1;31mSPAD worker: {e}\033[0m")


def _camera_worker(camera: RGBDCamera, buf: LatestFrame, stop: threading.Event):
    """Continuously capture frames from *camera* into *buf* until *stop* is set."""
    while not stop.is_set():
        frame = camera.get_frame(timeout_ms=500)
        if frame is not None:
            buf.put(frame)


def _start_workers(
    manager: Manager,
    spad_buf: LatestFrame,
    sensor_cam_buf: LatestFrame,
    overhead_cam_buf: LatestFrame,
    stop: threading.Event,
) -> list[threading.Thread]:
    """Launch background capture threads. Returns the thread list."""
    threads = []
    if cfg.USE_SPAD:
        t = threading.Thread(
            target=_spad_worker,
            args=(manager.components["spad"], spad_buf, stop),
            daemon=True, name="spad-worker",
        )
        t.start()
        threads.append(t)
    if cfg.USE_SENSOR_CAM:
        t = threading.Thread(
            target=_camera_worker,
            args=(manager.components["sensor_cam"], sensor_cam_buf, stop),
            daemon=True, name="sensor-cam-worker",
        )
        t.start()
        threads.append(t)
    if cfg.USE_OVERHEAD_CAM:
        t = threading.Thread(
            target=_camera_worker,
            args=(manager.components["overhead_cam"], overhead_cam_buf, stop),
            daemon=True, name="overhead-cam-worker",
        )
        t.start()
        threads.append(t)
    return threads


def _join_workers(stop: threading.Event, threads: list[threading.Thread]):
    """Signal workers to stop and wait for them."""
    stop.set()
    for t in threads:
        t.join(timeout=3.0)


# ── Display (main-thread only) ────────────────────────────────────────

def _update_display(manager: Manager, idx: int,
                    spad_data, sensor_cam_data, overhead_cam_data):
    """Push latest sensor data to Qt dashboard and OpenCV windows."""
    if spad_data is not None and cfg.SHOW_SPAD_DASHBOARD:
        pump_qt()
        manager.components["dashboard"].update(idx, data=spad_data)

    if sensor_cam_data is not None and cfg.SHOW_SENSOR_CAM_PREVIEW:
        depth_cm = cv2.applyColorMap(
            cv2.convertScaleAbs(sensor_cam_data["aligned_depth"], alpha=0.03),
            cv2.COLORMAP_JET,
        )
        cv2.imshow(SENSOR_CAM_WINDOW, np.hstack([sensor_cam_data["raw_rgb"], depth_cm]))
        cv2.waitKey(1)

    if overhead_cam_data is not None and cfg.SHOW_OVERHEAD_CAM_PREVIEW:
        cv2.imshow(OVERHEAD_CAM_WINDOW, overhead_cam_data["raw_rgb"])
        cv2.waitKey(1)


# ── Loop Modes ─────────────────────────────────────────────────────────

def run_loop(
    manager: Manager, writer: PklHandler,
    spad_buf: LatestFrame, sensor_cam_buf: LatestFrame, overhead_cam_buf: LatestFrame,
):
    """Continuous threaded capture. Paces on the slowest enabled sensor."""
    stop = threading.Event()
    threads = _start_workers(manager, spad_buf, sensor_cam_buf, overhead_cam_buf, stop)

    if cfg.USE_SPAD:
        pace_buf, pace_name = spad_buf, "SPAD"
    elif cfg.USE_SENSOR_CAM:
        pace_buf, pace_name = sensor_cam_buf, "sensor_cam"
    else:
        pace_buf, pace_name = overhead_cam_buf, "overhead_cam"

    print("\033[1;36mContinuous capture started (threaded). Press Ctrl+C to stop.\033[0m\n")
    idx = 0
    t0 = time.perf_counter()
    logged_sensor_cam = False
    logged_overhead_cam = False
    pace_stall_warned = False

    try:
        while True:
            # Short timeout so Qt events are pumped on every iteration and
            # camera previews stay live even when the pacing sensor is slow.
            paced, paced_ts = pace_buf.wait_for_new(timeout=0.1)

            # Always pump Qt so the SPAD dashboard stays responsive.
            pump_qt()

            if paced is None:
                # Pacing sensor has no new data — still show cameras so the
                # user can verify they are working, and warn if it persists.
                sc, _ = sensor_cam_buf.get()
                ov, _ = overhead_cam_buf.get()
                sd, _ = spad_buf.get()
                if sc is not None or ov is not None or sd is not None:
                    _update_display(manager, idx, sd, sc, ov)

                elapsed = time.perf_counter() - t0
                if elapsed > 5.0 and not pace_stall_warned:
                    print(
                        f"\033[1;31mWarning: '{pace_name}' (pacing sensor) has not "
                        f"produced data in {elapsed:.0f}s.\033[0m"
                    )
                    for name, buf, enabled in [
                        ("SPAD",         spad_buf,         cfg.USE_SPAD),
                        ("sensor_cam",   sensor_cam_buf,   cfg.USE_SENSOR_CAM),
                        ("overhead_cam", overhead_cam_buf, cfg.USE_OVERHEAD_CAM),
                    ]:
                        if enabled:
                            d, _ = buf.get()
                            status = "\033[1;32mOK\033[0m" if d is not None else "\033[1;31mno data\033[0m"
                            print(f"    {name:<14} {status}")
                    pace_stall_warned = True
                continue

            pace_stall_warned = False  # reset once pacing sensor resumes

            record = {"iter": idx}
            spad_data = spad_ts = None
            sensor_cam_data = sensor_cam_ts = None
            overhead_cam_data = overhead_cam_ts = None

            if cfg.USE_SPAD:
                if pace_buf is spad_buf:
                    spad_data, spad_ts = paced, paced_ts
                else:
                    spad_data, spad_ts = spad_buf.get()
                if spad_data is not None:
                    record["spad"] = spad_data
                    record["spad_timestamp"] = spad_ts

            if cfg.USE_SENSOR_CAM:
                if pace_buf is sensor_cam_buf:
                    sensor_cam_data, sensor_cam_ts = paced, paced_ts
                else:
                    sensor_cam_data, sensor_cam_ts = sensor_cam_buf.get()
                if sensor_cam_data is not None:
                    record["sensor_cam"] = sensor_cam_data
                    record["sensor_cam_timestamp"] = sensor_cam_ts
                    if not logged_sensor_cam:
                        print(
                            f"\033[1;34m[sensor_cam] streaming OK: "
                            f"RGB {sensor_cam_data['raw_rgb'].shape}, "
                            f"depth {sensor_cam_data['aligned_depth'].shape}\033[0m"
                        )
                        logged_sensor_cam = True

            if cfg.USE_OVERHEAD_CAM:
                if pace_buf is overhead_cam_buf:
                    overhead_cam_data, overhead_cam_ts = paced, paced_ts
                else:
                    overhead_cam_data, overhead_cam_ts = overhead_cam_buf.get()
                if overhead_cam_data is not None:
                    record["overhead_cam"] = overhead_cam_data
                    record["overhead_cam_timestamp"] = overhead_cam_ts
                    if not logged_overhead_cam:
                        print(
                            f"\033[1;34m[overhead_cam] streaming OK: "
                            f"RGB {overhead_cam_data['raw_rgb'].shape}\033[0m"
                        )
                        logged_overhead_cam = True

            _update_display(manager, idx, spad_data, sensor_cam_data, overhead_cam_data)
            writer.append(record)
            idx += 1

            if idx % 30 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  \033[1;33m{idx} frames | {idx / elapsed:.1f} fps | "
                      f"{elapsed:.1f}s\033[0m")
    except KeyboardInterrupt:
        pass

    _join_workers(stop, threads)
    elapsed = time.perf_counter() - t0
    fps = idx / elapsed if elapsed > 0 else 0
    print(f"\n\033[1;36mStopped. {idx} frames in {elapsed:.1f}s ({fps:.1f} fps)\033[0m")
    return idx


def run_manual(
    manager: Manager, writer: PklHandler,
    spad_buf: LatestFrame, sensor_cam_buf: LatestFrame, overhead_cam_buf: LatestFrame,
):
    """Manual capture with background threads. Enter grabs latest frames."""
    stop = threading.Event()
    threads = _start_workers(manager, spad_buf, sensor_cam_buf, overhead_cam_buf, stop)
    time.sleep(0.5)

    idx = 0
    try:
        while True:
            pump_qt()
            cmd = input(f"\033[1;32m[iter {idx}] Enter=capture, q=quit:\033[0m ").strip().lower()
            if cmd == "q":
                break

            record = {"iter": idx}
            spad_data = sensor_cam_data = overhead_cam_data = None

            if cfg.USE_SPAD:
                spad_data, spad_ts = spad_buf.get()
                if spad_data is not None:
                    record["spad"] = spad_data
                    record["spad_timestamp"] = spad_ts
                else:
                    print("  \033[1;33mWarning: no SPAD data yet\033[0m")

            if cfg.USE_SENSOR_CAM:
                sensor_cam_data, sensor_cam_ts = sensor_cam_buf.get()
                if sensor_cam_data is not None:
                    record["sensor_cam"] = sensor_cam_data
                    record["sensor_cam_timestamp"] = sensor_cam_ts
                else:
                    print("  \033[1;33mWarning: no sensor_cam data yet\033[0m")

            if cfg.USE_OVERHEAD_CAM:
                overhead_cam_data, overhead_cam_ts = overhead_cam_buf.get()
                if overhead_cam_data is not None:
                    record["overhead_cam"] = overhead_cam_data
                    record["overhead_cam_timestamp"] = overhead_cam_ts
                else:
                    print("  \033[1;33mWarning: no overhead_cam data yet\033[0m")

            _update_display(manager, idx, spad_data, sensor_cam_data, overhead_cam_data)
            writer.append(record)
            print(f"  \033[1;33mCaptured iter {idx}\033[0m")
            idx += 1
    except KeyboardInterrupt:
        pass

    _join_workers(stop, threads)
    return idx


# ── Main ───────────────────────────────────────────────────────────────

def main():
    LOGDIR.mkdir(parents=True, exist_ok=True)
    output = LOGDIR / f"{cfg.OBJECT}_data.pkl"
    assert not output.exists(), f"{output} already exists"

    print(
        "\033[1;35mConfig:"
        f" USE_SPAD={cfg.USE_SPAD},"
        f" USE_SENSOR_CAM={cfg.USE_SENSOR_CAM},"
        f" USE_OVERHEAD_CAM={cfg.USE_OVERHEAD_CAM}\033[0m"
    )

    spad_buf         = LatestFrame()
    sensor_cam_buf   = LatestFrame()
    overhead_cam_buf = LatestFrame()

    # Cameras are instantiated before the Manager block so they can be closed
    # explicitly in `finally`.  Manager only auto-closes cc_hardware Component
    # objects; raw camera objects are not Components.
    cameras: dict[str, RGBDCamera] = {}
    if cfg.USE_SENSOR_CAM:
        cameras["sensor_cam"] = RealsenseCameraWrapper(
            width=cfg.SENSOR_CAM_WIDTH,
            height=cfg.SENSOR_CAM_HEIGHT,
            fps=cfg.SENSOR_CAM_FPS,
        )
    if cfg.USE_OVERHEAD_CAM:
        cameras["overhead_cam"] = USBCameraWrapper(
            name_pattern=cfg.OVERHEAD_CAM_NAME,
            width=cfg.OVERHEAD_CAM_WIDTH,
            height=cfg.OVERHEAD_CAM_HEIGHT,
        )

    try:
        with Manager() as manager:
            metadata = {"object": cfg.OBJECT, "start_time": NOW.isoformat(), "sensors": {}}

            if cfg.USE_SPAD:
                metadata["sensors"]["spad"] = setup_spad(manager)
            for key, cam in cameras.items():
                metadata["sensors"][key] = _setup_camera(cam, manager, key)

            if cfg.SHOW_SENSOR_CAM_PREVIEW:
                cv2.namedWindow(SENSOR_CAM_WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(SENSOR_CAM_WINDOW, cfg.SENSOR_CAM_WIDTH * 2, cfg.SENSOR_CAM_HEIGHT)
                cv2.moveWindow(SENSOR_CAM_WINDOW, 10, 10)
            if cfg.SHOW_OVERHEAD_CAM_PREVIEW:
                cv2.namedWindow(OVERHEAD_CAM_WINDOW, cv2.WINDOW_NORMAL)
                cv2.moveWindow(OVERHEAD_CAM_WINDOW, 10, 10 + cfg.SENSOR_CAM_HEIGHT + 40)

            writer = PklHandler(output)
            manager.add(writer=writer)
            writer.append({"metadata": metadata})

            active = [n for n, on in [
                ("SPAD", cfg.USE_SPAD),
                ("sensor_cam (RealSense)", cfg.USE_SENSOR_CAM),
                ("overhead_cam (eMeet C960)", cfg.USE_OVERHEAD_CAM),
            ] if on]
            print(
                f"\033[1;36mSensors: {', '.join(active)} | Mode: {cfg.CAPTURE_MODE} "
                f"(threaded) | Output: {output.resolve()}\033[0m\n"
            )

            if cfg.CAPTURE_MODE == "loop":
                count = run_loop(
                    manager, writer, spad_buf, sensor_cam_buf, overhead_cam_buf,
                )
            else:
                count = run_manual(
                    manager, writer, spad_buf, sensor_cam_buf, overhead_cam_buf,
                )

    finally:
        for cam in cameras.values():
            cam.close()
        cv2.destroyAllWindows()

    print(f"\n\033[1;32mDone. {count} frames -> {output.resolve()}\033[0m\n")


if __name__ == "__main__":
    main()
