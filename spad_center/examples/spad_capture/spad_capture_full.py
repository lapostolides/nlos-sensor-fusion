#!/usr/bin/env python3

import os
os.environ["HYDRA_HYDRA_LOGGING__FILE"] = "false"
os.environ["HYDRA_JOB_LOGGING__FILE"] = "false"

import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.drivers.spads.tmf8828 import SPADID
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler
from cc_hardware.utils.manager import Manager

import sys
try:
    import pyrealsense2 as rs
    import numpy as np
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

NOW = datetime.now()


def setup(
    manager: Manager,
    *,
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    logdir: Path,
    object: str,
    spad_position: Tuple[float, float, float],
    use_realsense: bool = False,
):
    logdir.mkdir(parents=True, exist_ok=True)

    # sensor.spad_id = SPADID.ID15
    spad = SPADSensor.create_from_config(sensor)
    if not spad.is_okay:
        get_logger().fatal("Failed to initialize SPAD sensor")
        return
    manager.add(spad=spad)

    dashboard = SPADDashboard.create_from_config(dashboard, sensor=spad)
    dashboard.setup()
    manager.add(dashboard=dashboard)

    output_pkl = logdir / f"{object}_data.pkl"
    assert not output_pkl.exists(), f"Output file {output_pkl} already exists"
    pkl_writer = PklHandler(output_pkl)
    manager.add(writer=pkl_writer)

    metadata = {
        "object": object,
        "spad_position": {
            "x": spad_position[0],
            "y": spad_position[1],
            "z": spad_position[2],
        },
        "start_time": NOW.isoformat(),
    }

    if use_realsense and REALSENSE_AVAILABLE:
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

    histogram = spad.accumulate()
    dashboard.update(iter, histograms=histogram)    
    data = {
        "iter": iter,
        "histogram": histogram,
    }

    if use_realsense and REALSENSE_AVAILABLE:
        pipeline = manager.components["realsense_pipeline"]
        align = manager.components["realsense_align"]

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        raw_depth_image = np.asanyarray(depth_frame.get_data())
        raw_color_image = np.asanyarray(color_frame.get_data())
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        aligned_color_image = np.asanyarray(aligned_color_frame.get_data())

        data["realsense_data"] = {
            "raw_depth_image": raw_depth_image,
            "raw_rgb_image": raw_color_image,
            "aligned_depth_image": aligned_depth_image,
            "aligned_rgb_image": aligned_color_image,
        }

        time.sleep(0.5)
    else:
        time.sleep(0.5)

    writer.append(data)

    return True


@register_cli
def spad_capture(
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    object: str,
    spad_position: Tuple[float, float, float],
    logdir: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S"),
    use_realsense: bool = False,
):
    _setup = partial(
        setup,
        sensor=sensor,
        dashboard=dashboard,
        logdir=logdir,
        object=object,
        spad_position=spad_position,
        use_realsense=use_realsense,
    )

    with Manager() as manager:
        manager.run(setup=_setup, loop=partial(loop, use_realsense=use_realsense))

        print(
            f"\033[1;32mPKL file saved to "
            f"{(logdir / f'{object}_data.pkl').resolve()}\033[0m"
        )


if __name__ == "__main__":
    run_cli(spad_capture)




# #!/usr/bin/env python3
# import sys
# import serial.tools.list_ports
# import subprocess
# from pathlib import Path
# from glob import glob
# from datetime import datetime

# # === Configuration ===
# OBJECT_NAME = "moon_crater"
# SPAD_POSITION = [0.1, 0.4, 0.5]
# DASHBOARD_CONFIG = "PyQtGraphDashboardConfig"
# DASHBOARD_FULLSCREEN = False
# RGBD_CAPTURE = True

# # === Log directory ===
# LOGDIR = Path("logs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")

# # === Script path ===
# CAPTURE_SCRIPT = Path(__file__).parent / "examples" / "spad_capture" / "spad_capture_new.py"

# def find_sensor_port():
#     if sys.platform == "darwin":
#         ports = glob("/dev/cu.*")
#         matches = [p for p in ports if "usbmodem" in p]
#         if not matches:
#             raise RuntimeError("No serial port matching 'usbmodem' found")
#         return sorted(matches)[0]
#     elif sys.platform == "win32":
#         valid_descriptions = ["USB Serial Device", "Arduino Uno"]
#         for port in serial.tools.list_ports.comports():
#             if any(desc in port.description for desc in valid_descriptions):
#                 return port.device
#         raise RuntimeError(f"No serial port with description in {valid_descriptions} found")
#     elif sys.platform in ["linux", "wsl"]:
#         ports = glob("/dev/ttyACM*")
#         if not ports:
#             raise RuntimeError("No serial port matching '/dev/ttyACM*' found")
#         return sorted(ports)[0]
#     else:
#         raise RuntimeError("Unsupported platform")

# SENSOR_PORT = find_sensor_port()  # Kept in case you need it elsewhere

# def build_command():
#     spad_pos_str = "[" + ",".join(str(v) for v in SPAD_POSITION) + "]"

#     cmd = [
#         "python", str(CAPTURE_SCRIPT),
#         f"dashboard={DASHBOARD_CONFIG}",
#         f"dashboard.fullscreen={str(DASHBOARD_FULLSCREEN).lower()}",
#         f"logdir={LOGDIR}",
#         f"+object={OBJECT_NAME}",
#         f"+spad_position={spad_pos_str}",
#         f"use_realsense={RGBD_CAPTURE}"
#     ]
#     return cmd

# def main():
#     cmd = build_command()
#     print("Running:", " \\\n  ".join(cmd))
#     subprocess.run(cmd, check=True)

# if __name__ == "__main__":
#     main()
