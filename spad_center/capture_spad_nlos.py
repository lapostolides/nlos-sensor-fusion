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

# Import gantry-related modules
from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.drivers.stepper_motors.stepper_controller import (
    SnakeStepperController,
    SnakeStepperControllerConfig,
    SnakeControllerAxisConfig
)
from cc_hardware.drivers.stepper_motors.telemetrix_stepper import (
    TelemetrixStepperMotorSystem,
    SingleDrive1AxisGantryXConfig,
    SingleDrive1AxisGantryYConfig,
    TelemetrixStepperMotorSystemConfig,
    StepperMotorSystemAxis,
)

# === Configuration Defaults (can be overridden by CLI) ===
OBJECT_NAME = "crater_calib"
SPAD_POSITION = (0.1, 0.4, 0.5)
SPAD_ID = SPADID.ID6 #ID6 3x3 or ID15 8x8
RANGE_MODE = RangeMode.LONG #LONG or SHORT 
DASHBOARD_CONFIG = "PyQtGraphDashboardConfig"
NOW = datetime.now()
LOGDIR = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S")
X_SAMPLES = 3
Y_SAMPLES = 3

# =========================================================

def find_ports():
    ports = {p.description: p.device for p in serial.tools.list_ports.comports()}
    spad_port = None
    gantry_port = None
    for desc, dev in ports.items():
        if "Arduino Uno" in desc:
            spad_port = dev
        elif "USB-SERIAL" in desc:
            gantry_port = dev
    if not spad_port or not gantry_port:
        raise RuntimeError(f"Could not find SPAD (Arduino Uno) or Gantry (USB-SERIAL) port")
    return spad_port, gantry_port
SPAD_PORT, GANTRY_PORT = find_ports()

def setup(
    manager: Manager,
    *,
    dashboard: SPADDashboardConfig,
    logdir: Path,
    object: str,
    spad_position: Tuple[float, float, float],
    spad_id: SPADID = SPADID.ID6,
    range_mode: RangeMode = RangeMode.LONG,
    gantry: StepperMotorSystem, # Add gantry to setup
):
    logdir.mkdir(parents=True, exist_ok=True)

    wrapped_sensor = TMF8828Config.create(spad_id=spad_id, range_mode=range_mode, port=SPAD_PORT)
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

    # Setup Gantry
    controller_config = SnakeStepperControllerConfig(
        axes={
            "x": SnakeControllerAxisConfig(range=(0, 32), samples=X_SAMPLES),
            "y": SnakeControllerAxisConfig(range=(0, 32), samples=Y_SAMPLES)
        }
    )
    controller = SnakeStepperController(controller_config)
    manager.add(gantry=gantry, gantry_controller=controller)

    output_pkl = logdir / f"{object}_data.pkl"
    assert not output_pkl.exists(), f"Output file {output_pkl} already exists"
    pkl_writer = PklHandler(output_pkl)
    manager.add(writer=pkl_writer)

    spad_realsense_pipeline = rs.pipeline()
    tracking_realsense_pipeline = rs.pipeline()

    spad_realsense_config = rs.config()
    spad_realsense_config.enable_device("243222070291")
    spad_realsense_config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    spad_realsense_config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    spad_profile = spad_realsense_pipeline.start(spad_realsense_config)
    spad_align = rs.align(rs.stream.color)

    tracking_realsense_config = rs.config()
    tracking_realsense_config.enable_device("912112073082")
    tracking_realsense_config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    tracking_realsense_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    tracking_profile = tracking_realsense_pipeline.start(tracking_realsense_config)
    tracking_align = rs.align(rs.stream.color)

    manager.add(spad_realsense_pipeline=spad_realsense_pipeline)
    manager.add(tracking_realsense_pipeline=tracking_realsense_pipeline)
    manager.add(spad_realsense_align=spad_align)
    manager.add(tracking_realsense_align=tracking_align)

    metadata = {
        "object": object,
        "spad_position": {"x": spad_position[0], "y": spad_position[1], "z": spad_position[2]},
        "start_time": NOW.isoformat(),
        "spad_id": str(spad_id.name),
        "range_mode": str(range_mode.name),
        "capture_mode": "sequential",
        "realsense_configs": {
            "spad_realsense": {
                "serial_number": "243222070291",
                "color_width": 848,
                "color_height": 480,
                "depth_width": 848,
                "depth_height": 480,
                "fps": 30,
            },
            "tracking_realsense": {
                "serial_number": "912112073082",
                "color_width": 1280,
                "color_height": 720,
                "depth_width": 1280,
                "depth_height": 720,
                "fps": 30,
            }
        },
    }

    pkl_writer.append({"metadata": metadata})


def loop(
    iter: int,
    manager: Manager,
    spad: SPADSensor,
    dashboard: SPADDashboard,
    writer: PklHandler,
    **kwargs,
) -> bool:
    
    # move gantry
    gantry_controller = manager.components["gantry_controller"]
    gantry = manager.components["gantry"]
    pos = gantry_controller.get_position(iter, verbose=False)
    if pos is None:
        print(f"\n=== exiting loop ===")
        return False
    
    print(f"\n\033[1;32m=== capturing iter {iter+1}/{X_SAMPLES*Y_SAMPLES} ===\033[0m")
    gantry.move_to(pos["x"], pos["y"])
    time.sleep(0.5) # wait for gantry to settle

    # dummy capture, spad accumulates once to reset 
    data = spad.accumulate()

    #actual capture
    data = spad.accumulate()
    dashboard.update(iter, data=data)
    record = {"iter": iter, **data}

    spad_pipeline = manager.components["spad_realsense_pipeline"]
    spad_align = manager.components["spad_realsense_align"]
    spad_frames = spad_pipeline.wait_for_frames()
    spad_aligned_frames = spad_align.process(spad_frames)
    spad_aligned_depth_frame = spad_aligned_frames.get_depth_frame()
    spad_aligned_color_frame = spad_aligned_frames.get_color_frame()

    tracking_pipeline = manager.components["tracking_realsense_pipeline"]
    tracking_align = manager.components["tracking_realsense_align"]
    tracking_frames = tracking_pipeline.wait_for_frames()
    tracking_aligned_frames = tracking_align.process(tracking_frames)
    tracking_aligned_depth_frame = tracking_aligned_frames.get_depth_frame()
    tracking_aligned_color_frame = tracking_aligned_frames.get_color_frame()

    record["spad_realsense_data"] = {
        "aligned_depth_image": np.asanyarray(spad_aligned_depth_frame.get_data()),
        "aligned_rgb_image": np.asanyarray(spad_aligned_color_frame.get_data()),
    }
    record["tracking_realsense_data"] = {
        "aligned_depth_image": np.asanyarray(tracking_aligned_depth_frame.get_data()),
        "aligned_rgb_image": np.asanyarray(tracking_aligned_color_frame.get_data()),
    }

    writer.append(record)
    get_logger().info(f"Captured and recorded iter {iter}")

    return True

def cleanup(gantry: StepperMotorSystem, manager: Manager, **kwargs):
    gantry.move_to(0, 0)
    gantry.close()
    manager.components["spad_realsense_pipeline"].stop()
    manager.components["tracking_realsense_pipeline"].stop()

@register_cli
def spad_capture(
    dashboard: SPADDashboardConfig,  # no default allowed
    object: str = OBJECT_NAME,
    spad_position: Tuple[float, float, float] = SPAD_POSITION,
    spad_id: SPADID = SPAD_ID,
    range_mode: RangeMode = RANGE_MODE,
    logdir: Path = LOGDIR,
):
    gantry_config = TelemetrixStepperMotorSystemConfig(
        axes={
            StepperMotorSystemAxis.X: [SingleDrive1AxisGantryXConfig()],
            StepperMotorSystemAxis.Y: [SingleDrive1AxisGantryYConfig()],
        },
        port=GANTRY_PORT,
    )
    gantry_MAIN = TelemetrixStepperMotorSystem(config=gantry_config)
    _setup = partial(
        setup,
        dashboard=dashboard,
        logdir=logdir,
        object=object,
        spad_position=spad_position,
        spad_id=spad_id,
        range_mode=range_mode,
        gantry=gantry_MAIN,
    )

    _loop = partial(
        loop,
    )

    with Manager() as manager:
        manager.run(setup=_setup, loop=_loop, cleanup=partial(cleanup, gantry=gantry_MAIN, manager=manager))
    
    print(f"\n\033[1;32mAll done. Data saved to {(logdir / f'{object}_data.pkl').resolve()}\033[0m\n")

if __name__ == "__main__":
    run_cli(spad_capture)