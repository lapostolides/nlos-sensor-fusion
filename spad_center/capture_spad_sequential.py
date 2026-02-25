#!/usr/bin/env python3
import os
os.environ["HYDRA_HYDRA_LOGGING__FILE"] = "false"
os.environ["HYDRA_JOB_LOGGING__FILE"] = "false"

import sys
import time
import serial.tools.list_ports
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from glob import glob
from datetime import datetime
from functools import partial
from typing import Tuple

from cc_hardware.drivers.spads import SPADSensor
from cc_hardware.drivers.spads.spad_wrappers import SPADMergeWrapperConfig
from cc_hardware.drivers.spads.tmf8828 import TMF8828Config, SPADID, RangeMode
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler
from cc_hardware.utils.manager import Manager

# === Configuration Defaults (can be overridden by CLI) ===
OBJECT_NAME = "crater_calib"
SPAD_POSITION = (0.1, 0.4, 0.5)
SPAD_ID = SPADID.ID6 #ID6 3x3 or ID15 8x8
RANGE_MODE = RangeMode.LONG #LONG or SHORT 
DASHBOARD_CONFIG = "PyQtGraphDashboardConfig"
DASHBOARD_FULLSCREEN = False
RGBD_CAPTURE = True
LOGDIR = Path("logs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
NOW = datetime.now()

def main():
    run_cli(spad_capture)
# =========================================================


def find_sensor_port():
    if sys.platform == "darwin":
        ports = glob("/dev/cu.*")
        matches = [p for p in ports if "usbmodem" in p]
        if not matches:
            raise RuntimeError("No serial port matching 'usbmodem' found")
        return sorted(matches)[0]
    elif sys.platform == "win32":
        valid_descriptions = ["USB Serial Device", "Arduino Uno"]
        for port in serial.tools.list_ports.comports():
            if any(desc in port.description for desc in valid_descriptions):
                return port.device
        raise RuntimeError(f"No serial port with description in {valid_descriptions} found")
    elif sys.platform in ["linux", "wsl"]:
        ports = glob("/dev/ttyACM*")
        if not ports:
            raise RuntimeError("No serial port matching '/dev/ttyACM*' found")
        return sorted(ports)[0]
    else:
        raise RuntimeError("Unsupported platform")


def setup(
    manager: Manager,
    *,
    dashboard: SPADDashboardConfig,
    logdir: Path,
    object: str,
    spad_position: Tuple[float, float, float],
    spad_id: SPADID = SPADID.ID6,
    range_mode: RangeMode = RangeMode.LONG,
    use_realsense: bool = False,
):
    logdir.mkdir(parents=True, exist_ok=True)

    wrapped_sensor = TMF8828Config.create(spad_id=spad_id, range_mode=range_mode)
    sensor_config = SPADMergeWrapperConfig.create(
        wrapped=wrapped_sensor,
        data_type="HISTOGRAM",
    )
    spad = SPADSensor.create_from_config(sensor_config)
    if not spad.is_okay:
        get_logger().fatal("Failed to initialize SPAD sensor")
        return
    manager.add(spad=spad)

    dashboard: SPADDashboard = dashboard.create_from_registry(
        config=dashboard, sensor=spad
    )
    dashboard.setup()
    manager.add(dashboard=dashboard)
    w = manager.components["dashboard"].win
    s = w.screen().geometry()
    w.move(s.width() - w.width() - 10, 10)
    w.show()


    output_pkl = logdir / f"{object}_data.pkl"
    assert not output_pkl.exists(), f"Output file {output_pkl} already exists"
    pkl_writer = PklHandler(output_pkl)
    manager.add(writer=pkl_writer)

    metadata = {
        "object": object,
        "spad_position": {"x": spad_position[0], "y": spad_position[1], "z": spad_position[2]},
        "start_time": NOW.isoformat(),
        "spad_id": str(spad_id.name),
        "range_mode": str(range_mode.name),
        "capture_mode": "sequential"
    }

    if use_realsense:
        realsense_pipeline = rs.pipeline()
        realsense_config = rs.config()
        realsense_config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        realsense_config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        profile = realsense_pipeline.start(realsense_config)
        align = rs.align(rs.stream.color)

        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_intrinsics = depth_profile.get_intrinsics()
        color_intrinsics = color_profile.get_intrinsics()

        manager.add(realsense_pipeline=realsense_pipeline)
        manager.add(realsense_align=align)

        metadata["realsense_config"] = {
            "color_width": 848,
            "color_height": 480,
            "depth_width": 848,
            "depth_height": 480,
            "fps": 30,
        }
        metadata["realsense_intrinsics"] = {
            "depth": {
                "ppx": depth_intrinsics.ppx,
                "ppy": depth_intrinsics.ppy,
                "fx": depth_intrinsics.fx,
                "fy": depth_intrinsics.fy,
                "model": str(depth_intrinsics.model),
                "coeffs": depth_intrinsics.coeffs,
            },
            "color": {
                "ppx": color_intrinsics.ppx,
                "ppy": color_intrinsics.ppy,
                "fx": color_intrinsics.fx,
                "fy": color_intrinsics.fy,
                "model": str(color_intrinsics.model),
                "coeffs": color_intrinsics.coeffs,
            },
        }

    pkl_writer.append({"metadata": metadata})


def loop(
    iter: int,
    manager: Manager,
    spad: SPADSensor,
    dashboard: SPADDashboard,
    writer: PklHandler,
    use_realsense: bool = False,
    **kwargs,
) -> bool:
    get_logger().info(f"Starting iter {iter}...")

    data = spad.accumulate()
    dashboard.update(iter, data=data)
    record = {"iter": iter, **data}

    if use_realsense:
        pipeline = manager.components["realsense_pipeline"]
        align = manager.components["realsense_align"]

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        record["realsense_data"] = {
            "raw_depth_image": np.asanyarray(depth_frame.get_data()),
            "raw_rgb_image": np.asanyarray(color_frame.get_data()),
            "aligned_depth_image": np.asanyarray(aligned_depth_frame.get_data()),
            "aligned_rgb_image": np.asanyarray(aligned_color_frame.get_data()),
        }

    writer.append(record)
    return True


@register_cli
def spad_capture(
    dashboard: SPADDashboardConfig,  # no default allowed
    object: str = OBJECT_NAME,
    spad_position: Tuple[float, float, float] = SPAD_POSITION,
    spad_id: SPADID = SPAD_ID,
    range_mode: RangeMode = RANGE_MODE,
    logdir: Path = LOGDIR,
    use_realsense: bool = RGBD_CAPTURE,
):
    # _setup = partial(
    #     setup,
    #     dashboard=dashboard,
    #     logdir=logdir,
    #     object=object,
    #     spad_position=spad_position,
    #     spad_id=spad_id,
    #     range_mode=range_mode,
    #     use_realsense=use_realsense,
    # )

    with Manager() as manager:
        setup(manager, dashboard=dashboard, logdir=logdir, object=object,
            spad_position=spad_position, spad_id=spad_id, range_mode=range_mode, use_realsense=use_realsense)

        iter_count = 0
        try:
            print("\033[1;33mRunning initial capture (iter 0)...\033[0m")
            loop(iter_count, manager, manager.components["spad"], manager.components["dashboard"],
                manager.components["writer"], use_realsense=use_realsense)
            iter_count += 1
        except Exception as e:
            print(f"\033[1;31mInitial capture failed: {e}\033[0m")

        while True:
            user_in = input(f"\033[1;32mReady to capture (current index = {iter_count}). "
                            f"Press [Enter] for 2 captures, 'q' to quit:\033[0m ").strip().lower()
            if user_in == "q":
                break
            print("\033[1;33mCapturing 2 frames...\033[0m")
            try:
                for _ in range(2):
                    loop(iter_count, manager, manager.components["spad"], manager.components["dashboard"],
                        manager.components["writer"], use_realsense=use_realsense)
                    iter_count += 1
            except Exception as e:
                print(f"\033[1;31mCapture failed: {e}\033[0m")

    print(f"\n\033[1;32mAll done. Data saved to {(logdir / f'{object}_data.pkl').resolve()}\033[0m\n")


if __name__ == "__main__":
    main()
