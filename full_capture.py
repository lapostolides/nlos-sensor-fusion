#!/usr/bin/env python3
"""
full_capture.py - Unified multi-sensor capture for NLOS fusion.
Sensors: SPAD (VL53L8CH), RealSense RGB-D.
Future: Thermal, UWB.

Each sensor runs in a dedicated background thread to avoid frame-rate drops.
The main thread handles Qt/OpenCV display and file I/O.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HYDRA_HYDRA_LOGGING__FILE"] = "false"
os.environ["HYDRA_JOB_LOGGING__FILE"] = "false"

import time
import threading
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from datetime import datetime

from cc_hardware.drivers.spads import SPADSensor
from cc_hardware.drivers.spads.spad_wrappers import SPADMergeWrapperConfig
from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHConfig8x8, VL53L8CHConfig4x4
from cc_hardware.tools.dashboard.spad_dashboard.pyqtgraph import PyQtGraphDashboardConfig
from cc_hardware.utils.file_handlers import PklHandler
from cc_hardware.utils.manager import Manager

import capture_config as cfg

NOW = datetime.now()
LOGDIR = Path("data/logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S")

SPAD_CONFIGS = {"8x8": VL53L8CHConfig8x8, "4x4": VL53L8CHConfig4x4}
REALSENSE_WINDOW = "RealSense RGB+Depth"


# ── Thread-safe frame buffer ──────────────────────────────────────────

class LatestFrame:
    """Single-slot buffer holding the most recent frame from a sensor.

    Workers call put() from their thread; the main thread calls get() or
    wait_for_new() to consume frames.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data = None
        self._timestamp: float | None = None
        self._has_new = threading.Event()

    def put(self, data):
        """Store a new frame (called from worker thread)."""
        with self._lock:
            self._data = data
            self._timestamp = time.perf_counter()
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


def setup_realsense(manager: Manager) -> dict:
    """Init RealSense RGB-D pipeline. Returns metadata with intrinsics."""
    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.depth, cfg.RS_WIDTH, cfg.RS_HEIGHT, rs.format.z16, cfg.RS_FPS)
    rs_cfg.enable_stream(rs.stream.color, cfg.RS_WIDTH, cfg.RS_HEIGHT, rs.format.bgr8, cfg.RS_FPS)
    profile = pipeline.start(rs_cfg)
    device = profile.get_device()
    device_name = device.get_info(rs.camera_info.name)
    serial_number = device.get_info(rs.camera_info.serial_number)
    align = rs.align(rs.stream.color)
    manager.add(rs_pipeline=pipeline, rs_align=align)

    print(f"\033[1;34mRealSense connected: {device_name} (S/N: {serial_number})\033[0m")
    if cfg.SHOW_REALSENSE_PREVIEW:
        cv2.namedWindow(REALSENSE_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(REALSENSE_WINDOW, cfg.RS_WIDTH * 2, cfg.RS_HEIGHT)
        cv2.moveWindow(REALSENSE_WINDOW, 10, 10)
        print("\033[1;34mRealSense preview enabled (RGB + aligned depth).\033[0m")

    def _intrinsics(stream):
        i = profile.get_stream(stream).as_video_stream_profile().get_intrinsics()
        return {"ppx": i.ppx, "ppy": i.ppy, "fx": i.fx, "fy": i.fy,
                "model": str(i.model), "coeffs": i.coeffs}

    return {
        "device_name": device_name,
        "serial_number": serial_number,
        "resolution": [cfg.RS_WIDTH, cfg.RS_HEIGHT], "fps": cfg.RS_FPS,
        "intrinsics": {
            "depth": _intrinsics(rs.stream.depth),
            "color": _intrinsics(rs.stream.color),
        },
    }


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


def _realsense_worker(pipeline, align, buf: LatestFrame, stop: threading.Event):
    """Continuously capture & align RealSense frames into *buf*."""
    while not stop.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=500)
        except Exception:
            continue

        try:
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            aligned = align.process(frames)
            ad = aligned.get_depth_frame()
            ac = aligned.get_color_frame()
            if not ad or not ac:
                continue

            # .copy() is critical: the SDK reuses frame memory on the next call
            buf.put({
                "raw_depth": np.asanyarray(depth_frame.get_data()).copy(),
                "raw_rgb": np.asanyarray(color_frame.get_data()).copy(),
                "aligned_depth": np.asanyarray(ad.get_data()).copy(),
                "aligned_rgb": np.asanyarray(ac.get_data()).copy(),
            })
        except Exception as e:
            if not stop.is_set():
                print(f"  \033[1;31mRealSense worker: {e}\033[0m")


def _start_workers(manager: Manager, spad_buf: LatestFrame, rs_buf: LatestFrame,
                   stop: threading.Event) -> list[threading.Thread]:
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
    if cfg.USE_REALSENSE:
        t = threading.Thread(
            target=_realsense_worker,
            args=(manager.components["rs_pipeline"], manager.components["rs_align"],
                  rs_buf, stop),
            daemon=True, name="rs-worker",
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

def _update_display(manager: Manager, idx: int, spad_data, rs_data):
    """Push latest sensor data to Qt dashboard and OpenCV window."""
    if spad_data is not None and cfg.SHOW_SPAD_DASHBOARD:
        pump_qt()
        manager.components["dashboard"].update(idx, data=spad_data)

    if rs_data is not None and cfg.SHOW_REALSENSE_PREVIEW:
        depth_cm = cv2.applyColorMap(
            cv2.convertScaleAbs(rs_data["aligned_depth"], alpha=0.03),
            cv2.COLORMAP_JET,
        )
        cv2.imshow(REALSENSE_WINDOW, np.hstack([rs_data["raw_rgb"], depth_cm]))
        cv2.waitKey(1)


# ── Loop Modes ─────────────────────────────────────────────────────────

def run_loop(manager: Manager, writer: PklHandler,
             spad_buf: LatestFrame, rs_buf: LatestFrame):
    """Continuous threaded capture. Paces on the slowest enabled sensor."""
    stop = threading.Event()
    threads = _start_workers(manager, spad_buf, rs_buf, stop)

    pace_buf = spad_buf if cfg.USE_SPAD else rs_buf

    print("\033[1;36mContinuous capture started (threaded). Press Ctrl+C to stop.\033[0m\n")
    idx = 0
    t0 = time.perf_counter()
    logged_rs = False
    try:
        while True:
            paced, _ = pace_buf.wait_for_new(timeout=2.0)
            if paced is None:
                continue

            record = {"iter": idx, "timestamp": datetime.now().isoformat()}
            spad_data = rs_data = None

            if cfg.USE_SPAD:
                spad_data = paced if pace_buf is spad_buf else spad_buf.get()[0]
                if spad_data is not None:
                    record["spad"] = spad_data

            if cfg.USE_REALSENSE:
                rs_data = paced if pace_buf is rs_buf else rs_buf.get()[0]
                if rs_data is not None:
                    record["realsense"] = rs_data
                    if not logged_rs:
                        print(
                            f"\033[1;34mRealSense streaming OK: "
                            f"RGB {rs_data['raw_rgb'].shape}, "
                            f"depth {rs_data['aligned_depth'].shape}\033[0m"
                        )
                        logged_rs = True

            _update_display(manager, idx, spad_data, rs_data)
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


def run_manual(manager: Manager, writer: PklHandler,
               spad_buf: LatestFrame, rs_buf: LatestFrame):
    """Manual capture with background threads. Enter grabs latest frames."""
    stop = threading.Event()
    threads = _start_workers(manager, spad_buf, rs_buf, stop)
    time.sleep(0.5)

    idx = 0
    try:
        while True:
            pump_qt()
            cmd = input(f"\033[1;32m[iter {idx}] Enter=capture, q=quit:\033[0m ").strip().lower()
            if cmd == "q":
                break

            record = {"iter": idx, "timestamp": datetime.now().isoformat()}
            spad_data = rs_data = None

            if cfg.USE_SPAD:
                spad_data, _ = spad_buf.get()
                if spad_data is not None:
                    record["spad"] = spad_data
                else:
                    print("  \033[1;33mWarning: no SPAD data yet\033[0m")

            if cfg.USE_REALSENSE:
                rs_data, _ = rs_buf.get()
                if rs_data is not None:
                    record["realsense"] = rs_data
                else:
                    print("  \033[1;33mWarning: no RealSense data yet\033[0m")

            _update_display(manager, idx, spad_data, rs_data)
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
        f" USE_REALSENSE={cfg.USE_REALSENSE},"
        f" SHOW_REALSENSE_PREVIEW={cfg.SHOW_REALSENSE_PREVIEW}\033[0m"
    )
    if cfg.SHOW_REALSENSE_PREVIEW and not cfg.USE_REALSENSE:
        print(
            "\033[1;33mWarning: SHOW_REALSENSE_PREVIEW=True but USE_REALSENSE=False; "
            "no RealSense stream will be shown.\033[0m"
        )

    spad_buf = LatestFrame()
    rs_buf = LatestFrame()

    with Manager() as manager:
        metadata = {"object": cfg.OBJECT, "start_time": NOW.isoformat(), "sensors": {}}

        if cfg.USE_SPAD:
            metadata["sensors"]["spad"] = setup_spad(manager)
        if cfg.USE_REALSENSE:
            metadata["sensors"]["realsense"] = setup_realsense(manager)

        writer = PklHandler(output)
        manager.add(writer=writer)
        writer.append({"metadata": metadata})

        active = [n for n, on in [("SPAD", cfg.USE_SPAD), ("RealSense", cfg.USE_REALSENSE)] if on]
        print(
            f"\033[1;36mSensors: {', '.join(active)} | Mode: {cfg.CAPTURE_MODE} "
            f"(threaded) | Output: {output.resolve()}\033[0m\n"
        )

        if cfg.CAPTURE_MODE == "loop":
            count = run_loop(manager, writer, spad_buf, rs_buf)
        else:
            count = run_manual(manager, writer, spad_buf, rs_buf)

    if cfg.USE_REALSENSE and cfg.SHOW_REALSENSE_PREVIEW:
        cv2.destroyAllWindows()

    print(f"\n\033[1;32mDone. {count} frames -> {output.resolve()}\033[0m\n")


if __name__ == "__main__":
    main()
