#!/usr/bin/env python3

import os
os.environ["HYDRA_HYDRA_LOGGING__FILE"] = "false"
os.environ["HYDRA_JOB_LOGGING__FILE"] = "false"

import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple

from cc_hardware.drivers.spads import SPADSensor
from cc_hardware.drivers.spads.spad_wrappers import SPADMergeWrapperConfig
from cc_hardware.drivers.spads.tmf8828 import TMF8828Config, SPADID
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
    dashboard: SPADDashboardConfig,
    logdir: Path,
    object: str,
    spad_position: Tuple[float, float, float],
    use_realsense: bool = False,
):
    logdir.mkdir(parents=True, exist_ok=True)

    wrapped_sensor = TMF8828Config.create(spad_id=SPADID.ID15)
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

    data = spad.accumulate()
    dashboard.update(iter, data=data)
    record = {"iter": iter, **data}

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

        record["realsense_data"] = {
            "raw_depth_image": raw_depth_image,
            "raw_rgb_image": raw_color_image,
            "aligned_depth_image": aligned_depth_image,
            "aligned_rgb_image": aligned_color_image,
        }

        time.sleep(0.5)
    else:
        time.sleep(0.5)

    writer.append(record)

    return True


@register_cli
def spad_capture(
    dashboard: SPADDashboardConfig,
    object: str,
    spad_position: Tuple[float, float, float],
    logdir: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S"),
    use_realsense: bool = False,
):
    _setup = partial(
        setup,
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
