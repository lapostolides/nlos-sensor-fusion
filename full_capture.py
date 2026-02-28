#!/usr/bin/env python3
"""
full_capture.py - Unified multi-sensor capture for NLOS fusion.
Sensors: SPAD (VL53L8CH), RealSense RGB-D.
Future: Thermal, UWB.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HYDRA_HYDRA_LOGGING__FILE"] = "false"
os.environ["HYDRA_JOB_LOGGING__FILE"] = "false"

import time
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from datetime import datetime

from cc_hardware.drivers.spads import SPADSensor
from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHConfig8x8, VL53L8CHConfig4x4
from cc_hardware.tools.dashboard.spad_dashboard.pyqtgraph import PyQtGraphDashboardConfig
from cc_hardware.utils.file_handlers import PklHandler
from cc_hardware.utils.manager import Manager

import capture_config as cfg

NOW = datetime.now()
LOGDIR = Path("data/logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S")

SPAD_CONFIGS = {"8x8": VL53L8CHConfig8x8, "4x4": VL53L8CHConfig4x4}


# ── Sensor Setup ───────────────────────────────────────────────────────

def setup_spad(manager: Manager) -> dict:
    """Init VL53L8CH SPAD sensor + optional live dashboard. Returns metadata."""
    config = SPAD_CONFIGS[cfg.SPAD_RESOLUTION].create()
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
        "sensor": "VL53L8CH",
        "resolution": f"{config.width}x{config.height}",
        "ranging_frequency_hz": config.ranging_frequency_hz,
        "integration_time_ms": config.integration_time_ms,
        "num_bins": config.num_bins,
    }


def setup_realsense(manager: Manager) -> dict:
    """Init RealSense RGB-D pipeline. Returns metadata with intrinsics."""
    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.depth, cfg.RS_WIDTH, cfg.RS_HEIGHT, rs.format.z16, cfg.RS_FPS)
    rs_cfg.enable_stream(rs.stream.color, cfg.RS_WIDTH, cfg.RS_HEIGHT, rs.format.bgr8, cfg.RS_FPS)
    profile = pipeline.start(rs_cfg)
    align = rs.align(rs.stream.color)
    manager.add(rs_pipeline=pipeline, rs_align=align)

    def _intrinsics(stream):
        i = profile.get_stream(stream).as_video_stream_profile().get_intrinsics()
        return {"ppx": i.ppx, "ppy": i.ppy, "fx": i.fx, "fy": i.fy,
                "model": str(i.model), "coeffs": i.coeffs}

    return {
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


# ── Capture ────────────────────────────────────────────────────────────

def capture_frame(manager: Manager, idx: int) -> dict:
    """Capture one frame from all active sensors."""
    record = {"iter": idx, "timestamp": datetime.now().isoformat()}

    if cfg.USE_SPAD:
        pump_qt()
        data = manager.components["spad"].accumulate()
        if cfg.SHOW_SPAD_DASHBOARD:
            manager.components["dashboard"].update(idx, data=data)
        record["spad"] = data

    if cfg.USE_REALSENSE:
        pipeline = manager.components["rs_pipeline"]
        align = manager.components["rs_align"]
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        raw_rgb = np.asanyarray(frames.get_color_frame().get_data())
        aligned_depth = np.asanyarray(aligned.get_depth_frame().get_data())

        record["realsense"] = {
            "raw_depth": np.asanyarray(frames.get_depth_frame().get_data()),
            "raw_rgb": raw_rgb,
            "aligned_depth": aligned_depth,
            "aligned_rgb": np.asanyarray(aligned.get_color_frame().get_data()),
        }

        if cfg.SHOW_REALSENSE_PREVIEW:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(aligned_depth, alpha=0.03), cv2.COLORMAP_JET
            )
            cv2.imshow("RealSense", np.hstack([raw_rgb, depth_colormap]))
            cv2.waitKey(1)

    return record


# ── Loop Modes ─────────────────────────────────────────────────────────

def run_loop(manager: Manager, writer: PklHandler):
    """Continuous capture at sensor rate. Ctrl+C to stop."""
    print("\033[1;36mContinuous capture started. Press Ctrl+C to stop.\033[0m\n")
    idx = 0
    t0 = time.perf_counter()
    try:
        while True:
            try:
                writer.append(capture_frame(manager, idx))
                idx += 1
                if idx % 30 == 0:
                    elapsed = time.perf_counter() - t0
                    print(f"  \033[1;33m{idx} frames | {idx / elapsed:.1f} fps | {elapsed:.1f}s\033[0m")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  \033[1;31mFrame {idx} failed: {e}\033[0m")
    except KeyboardInterrupt:
        pass
    elapsed = time.perf_counter() - t0
    fps = idx / elapsed if elapsed > 0 else 0
    print(f"\n\033[1;36mStopped. {idx} frames in {elapsed:.1f}s ({fps:.1f} fps)\033[0m")
    return idx


def run_manual(manager: Manager, writer: PklHandler):
    """One capture per Enter. 'q' to quit."""
    idx = 0
    while True:
        pump_qt()
        cmd = input(f"\033[1;32m[iter {idx}] Enter=capture, q=quit:\033[0m ").strip().lower()
        if cmd == "q":
            break
        try:
            writer.append(capture_frame(manager, idx))
            print(f"  \033[1;33mCaptured iter {idx}\033[0m")
            idx += 1
        except Exception as e:
            print(f"  \033[1;31mCapture failed: {e}\033[0m")
    return idx


# ── Main ───────────────────────────────────────────────────────────────

def main():
    LOGDIR.mkdir(parents=True, exist_ok=True)
    output = LOGDIR / f"{cfg.OBJECT}_data.pkl"
    assert not output.exists(), f"{output} already exists"

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
        print(f"\033[1;36mSensors: {', '.join(active)} | Mode: {cfg.CAPTURE_MODE} | Output: {output.resolve()}\033[0m\n")

        if cfg.CAPTURE_MODE == "loop":
            count = run_loop(manager, writer)
        else:
            count = run_manual(manager, writer)

    if cfg.USE_REALSENSE and cfg.SHOW_REALSENSE_PREVIEW:
        cv2.destroyAllWindows()

    print(f"\n\033[1;32mDone. {count} frames -> {output.resolve()}\033[0m\n")


if __name__ == "__main__":
    main()
