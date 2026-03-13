#!/usr/bin/env python3
"""
full_capture.py - Unified multi-sensor capture for NLOS fusion.

Sensors: SPAD (VL53L8CH), sensor_cam (RealSense RGB-D),
         overhead_cam (eMeet C960), UWB (DWM1001-DEV).

All sensors are initialized and standing by before capture starts.
A shared start gate releases all threads simultaneously.

All configuration is in capture_config.py — no CLI args. Just run:
    python full_capture.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HYDRA_HYDRA_LOGGING__FILE"] = "false"
os.environ["HYDRA_JOB_LOGGING__FILE"] = "false"

import os as _os
import signal
import sys
import time
import threading
from pathlib import Path
import shutil
from datetime import datetime

print("[full_capture] Starting...", flush=True)

import cv2
import numpy as np

print("[full_capture] Loading capture_config...", flush=True)
import capture_config as cfg

print("[full_capture] Loading cc_hardware drivers...", flush=True)
from cc_hardware.drivers.spads import SPADSensor
from cc_hardware.drivers.spads.spad_wrappers import SPADMergeWrapperConfig
from cc_hardware.drivers.spads.vl53l8ch import VL53L8CHConfig8x8, VL53L8CHConfig4x4
from cc_hardware.tools.dashboard.spad_dashboard.pyqtgraph import PyQtGraphDashboardConfig
from cc_hardware.utils.manager import Manager
print("[full_capture] cc_hardware loaded.", flush=True)

# Set matplotlib backend to Qt5Agg BEFORE importing pyplot so it shares
# the same QApplication as PyQtGraph.
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from cameras import RGBDCamera, RealsenseCameraWrapper, USBCameraWrapper
from run_writer import RunWriter

SPAD_CONFIGS        = {"8x8": VL53L8CHConfig8x8, "4x4": VL53L8CHConfig4x4}
SENSOR_CAM_WINDOW   = "sensor_cam  RealSense RGB+Depth"
OVERHEAD_CAM_WINDOW = "overhead_cam  eMeet C960 RGB"

N_SAMPLES  = 1016
RX_ROLES   = ("rx1", "rx2", "rx3")
RX_COLORS  = ("steelblue", "tomato", "seagreen")


def _log(msg: str):
    print(f"[full_capture] {msg}", flush=True)


# ── Graceful shutdown ─────────────────────────────────────────────────
# First Ctrl+C  → sets _stop_event, lets cleanup run normally.
# Second Ctrl+C → force-exits immediately (no waiting for blocked threads).

_stop_event: threading.Event | None = None  # set by main()


def _sigint_handler(signum, frame):
    global _stop_event
    if _stop_event is not None and _stop_event.is_set():
        # Second Ctrl+C — force exit
        _log("Force exit (second Ctrl+C).")
        _os._exit(1)
    if _stop_event is not None:
        _stop_event.set()
    _log("Ctrl+C received — shutting down (press again to force-quit)...")
    # Don't raise KeyboardInterrupt here — Qt's processEvents() swallows it,
    # preventing the except clause in run_loop from ever firing.  Instead,
    # the loop exits via `while not stop.is_set()`.


signal.signal(signal.SIGINT, _sigint_handler)


# ── Thread-safe frame buffer ──────────────────────────────────────────

class LatestFrame:
    """Single-slot buffer holding the most recent frame from a sensor."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = None
        self._timestamp: str | None = None
        self._has_new = threading.Event()

    def put(self, data):
        with self._lock:
            self._data = data
            self._timestamp = datetime.now().isoformat()
        self._has_new.set()

    def get(self):
        with self._lock:
            return self._data, self._timestamp

    def wait_for_new(self, timeout: float = 2.0):
        if not self._has_new.wait(timeout=timeout):
            return None, None
        self._has_new.clear()
        with self._lock:
            return self._data, self._timestamp


# ── SPAD port auto-detection ──────────────────────────────────────────

# STMicroelectronics NUCLEO board (VL53L8CH connects via this)
_SPAD_USB_VID = 0x0483  # STMicroelectronics


def find_spad_port() -> str:
    """Auto-detect the serial port for the VL53L8CH SPAD sensor.

    Identifies the port by USB Vendor ID (STMicroelectronics 0x0483).
    Raises RuntimeError if no matching port or multiple matches found.
    """
    import serial.tools.list_ports

    matches = []
    for p in serial.tools.list_ports.comports():
        if p.vid == _SPAD_USB_VID:
            matches.append(p)

    if not matches:
        raise RuntimeError(
            "No SPAD (STMicro) serial port found. "
            "Is the VL53L8CH NUCLEO board plugged in?"
        )
    if len(matches) > 1:
        ports_str = ", ".join(f"{p.device} ({p.description})" for p in matches)
        raise RuntimeError(
            f"Multiple STMicro serial ports found: {ports_str}. "
            "Unplug extra devices or set SPAD_PORT in capture_config.py."
        )
    port = matches[0]
    _log(f"Auto-detected SPAD on {port.device} ({port.description})")
    return port.device


# ── Sensor Setup ───────────────────────────────────────────────────────

def setup_spad(manager: Manager) -> dict:
    port = getattr(cfg, "SPAD_PORT", None) or find_spad_port()
    _log(f"Initializing SPAD ({cfg.SPAD_RESOLUTION}) on {port}...")
    wrapped_config = SPAD_CONFIGS[cfg.SPAD_RESOLUTION].create(port=port)
    config = SPADMergeWrapperConfig.create(
        wrapped=wrapped_config,
        data_type="HISTOGRAM",
    )
    sensor = SPADSensor.create_from_config(config)
    assert sensor.is_okay, "SPAD sensor failed to initialize"

    # Force-send initial config to firmware.  VL53L8CHSensor.update() only
    # sends config.pack() when super().update() returns True (i.e. something
    # changed).  On first init with no kwargs and no dirty settings, nothing
    # changes → config is never sent → sensor never starts ranging.
    inner = sensor.unwrapped if hasattr(sensor, "unwrapped") else sensor
    if hasattr(inner, "_write_queue"):
        _log("Sending initial config to SPAD firmware...")
        inner._write_queue.put(inner.config.pack())

    manager.add(spad=sensor)
    _log("SPAD initialized.")

    if cfg.SHOW_SPAD_DASHBOARD:
        _log("Setting up SPAD dashboard...")
        dash_cfg = PyQtGraphDashboardConfig.create()
        dashboard = dash_cfg.create_from_registry(config=dash_cfg, sensor=sensor)
        dashboard.setup()
        manager.add(dashboard=dashboard)

        w = dashboard.win
        screen = w.screen().geometry()
        w.move(screen.width() - w.width() - 10, 10)
        w.show()
        _log("SPAD dashboard ready.")

    return {
        "sensor": "VL53L8CH (SPADMergeWrapper)",
        "resolution": f"{wrapped_config.width}x{wrapped_config.height}",
        "ranging_frequency_hz": wrapped_config.ranging_frequency_hz,
        "integration_time_ms": wrapped_config.integration_time_ms,
        "num_bins": wrapped_config.num_bins,
    }


def _setup_camera(cam: RGBDCamera, manager: Manager, key: str) -> dict:
    _log(f"Initializing {key}...")
    metadata = cam.start()
    assert cam.is_okay, f"Camera '{key}' failed to initialize"
    manager.add(**{key: cam})
    _log(f"{key} initialized.")
    return metadata


# ── Qt / display helpers ──────────────────────────────────────────────

def pump_qt():
    try:
        from pyqtgraph.Qt import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app:
            app.processEvents()
    except Exception:
        pass


def _setup_cir_figure():
    """Create a matplotlib figure with 3 subplots for live CIR display."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Live UWB CIR", fontsize=12)
    sample_axis = np.arange(N_SAMPLES)

    lines = {}
    fp_lines = {}
    for ax, role, color in zip(axes, RX_ROLES, RX_COLORS):
        (line,) = ax.plot(sample_axis, np.zeros(N_SAMPLES), lw=0.8,
                          color=color, label=role)
        fp_line = ax.axvline(x=0, color=color, lw=1, ls="--", alpha=0.5)
        ax.set_ylabel(role)
        ax.set_xlim(0, N_SAMPLES - 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        lines[role] = line
        fp_lines[role] = fp_line

    axes[-1].set_xlabel("Sample index (FP-aligned)")
    fig.tight_layout()
    fig.show()
    return fig, axes, lines, fp_lines


# ── Worker Threads ─────────────────────────────────────────────────────

def _spad_accumulate_with_timeout(sensor, timeout: float = 5.0):
    """Call sensor.accumulate() with a timeout.

    accumulate() can block forever if the sensor isn't streaming, so we run
    it in a daemon thread and wait with a timeout.  Returns the data or None.
    """
    result = [None]
    exc = [None]

    def _inner():
        try:
            result[0] = sensor.accumulate()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_inner, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return None, None  # timed out — accumulate is stuck
    if exc[0] is not None:
        return None, exc[0]
    return result[0], None


def _spad_worker(sensor, buf: LatestFrame, stop: threading.Event,
                 start_gate: threading.Event):
    _log("[spad] Worker ready.")
    start_gate.wait()
    _log("[spad] Capturing...")
    _log(f"[spad] sensor.is_okay = {sensor.is_okay}")
    n = 0
    errors = 0
    stall_warned = False
    while not stop.is_set():
        data, exc = _spad_accumulate_with_timeout(sensor, timeout=5.0)

        if exc is not None:
            errors += 1
            if not stop.is_set():
                print(f"  \033[1;31mSPAD worker error #{errors}: "
                      f"{type(exc).__name__}: {exc}\033[0m", flush=True)
            if errors >= 10:
                _log("[spad] Too many errors, stopping worker.")
                break
            continue

        if data is None:
            # accumulate() timed out
            if not stall_warned and not stop.is_set():
                _log("[spad] WARNING: accumulate() timed out — sensor may not be streaming.")
                _log(f"[spad] sensor.is_okay = {sensor.is_okay}")
                stall_warned = True
            continue

        stall_warned = False
        buf.put(data)
        n += 1
        if n == 1:
            _log(f"[spad] First frame received: type={type(data).__name__}")
        elif n % 100 == 0:
            _log(f"[spad] {n} frames accumulated")
    _log(f"[spad] Stopped ({n} frames, {errors} errors).")


def _camera_worker(camera: RGBDCamera, name: str, buf: LatestFrame,
                   stop: threading.Event, start_gate: threading.Event):
    _log(f"[{name}] Worker ready.")
    start_gate.wait()
    _log(f"[{name}] Capturing...")
    n = 0
    while not stop.is_set():
        frame = camera.get_frame(timeout_ms=500)
        if frame is not None:
            buf.put(frame)
            n += 1
            if n == 1:
                shapes = ", ".join(f"{k}: {v.shape}" for k, v in frame.items()
                                  if hasattr(v, "shape"))
                _log(f"[{name}] First frame: {shapes}")
    _log(f"[{name}] Stopped ({n} frames).")


def _start_workers(
    manager: Manager,
    spad_buf: LatestFrame,
    sensor_cam_buf: LatestFrame,
    overhead_cam_buf: LatestFrame,
    stop: threading.Event,
    start_gate: threading.Event,
) -> list[threading.Thread]:
    threads = []
    if cfg.USE_SPAD:
        t = threading.Thread(
            target=_spad_worker,
            args=(manager.components["spad"], spad_buf, stop, start_gate),
            daemon=True, name="spad-worker",
        )
        t.start()
        threads.append(t)
    if cfg.USE_SENSOR_CAM:
        t = threading.Thread(
            target=_camera_worker,
            args=(manager.components["sensor_cam"], "sensor_cam",
                  sensor_cam_buf, stop, start_gate),
            daemon=True, name="sensor-cam-worker",
        )
        t.start()
        threads.append(t)
    if cfg.USE_OVERHEAD_CAM:
        t = threading.Thread(
            target=_camera_worker,
            args=(manager.components["overhead_cam"], "overhead_cam",
                  overhead_cam_buf, stop, start_gate),
            daemon=True, name="overhead-cam-worker",
        )
        t.start()
        threads.append(t)
    _log(f"Started {len(threads)} sensor worker(s) (waiting on start gate).")
    return threads


def _join_workers(stop: threading.Event, threads: list[threading.Thread]):
    stop.set()
    for t in threads:
        t.join(timeout=3.0)


# ── Sensor Health ─────────────────────────────────────────────────────

_STALE_WARN_THRESHOLD = 30  # consecutive iterations before warning


def _check_fresh(name, data, ts, miss, last_ts):
    """Return True if sensor data is fresh (should be written).

    Increments *miss[name]* on None or stale reads, resets on fresh data.
    Prints a bold warning once when the threshold is crossed.
    """
    if data is None or (ts is not None and ts == last_ts.get(name)):
        miss[name] = miss.get(name, 0) + 1
        if miss[name] == _STALE_WARN_THRESHOLD:
            _log(f"\033[1;31m[WARNING] {name}: {miss[name]} consecutive "
                 f"stale/empty frames — sensor may have stopped\033[0m")
        return False
    miss[name] = 0
    last_ts[name] = ts
    return True


# ── Display (main-thread only) ────────────────────────────────────────

def _update_display(manager: Manager, idx: int,
                    spad_data, sensor_cam_data, overhead_cam_data,
                    cir_state=None):
    """Push latest sensor data to dashboards and preview windows."""
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

    # Live CIR update (~2fps so only redraw when new data arrives)
    if cir_state is not None:
        fig, axes, lines, fp_lines, cir_bufs = cir_state
        needs_draw = False
        for role, ax in zip(RX_ROLES, axes):
            buf = cir_bufs.get(role)
            if buf is None:
                continue
            data, _ = buf.get()
            if data is None:
                continue
            cir_mag, fp_index = data
            mag = np.roll(cir_mag, -int(fp_index))
            lines[role].set_ydata(mag)
            ax.set_ylim(0, float(mag.max()) * 1.15 + 1e-6)
            needs_draw = True
        if needs_draw:
            fig.canvas.draw_idle()
            pump_qt()


# ── Loop Modes ─────────────────────────────────────────────────────────

def run_loop(
    manager: Manager, writer: RunWriter,
    spad_buf: LatestFrame, sensor_cam_buf: LatestFrame, overhead_cam_buf: LatestFrame,
    stop: threading.Event, start_gate: threading.Event,
    cir_state=None,
):
    threads = _start_workers(
        manager, spad_buf, sensor_cam_buf, overhead_cam_buf, stop, start_gate,
    )

    if cfg.USE_SENSOR_CAM:
        pace_buf, pace_name = sensor_cam_buf, "sensor_cam"
    elif cfg.USE_OVERHEAD_CAM:
        pace_buf, pace_name = overhead_cam_buf, "overhead_cam"
    else:
        pace_buf, pace_name = spad_buf, "SPAD"

    # Open the gate — all threads start capturing simultaneously.
    _log("All workers ready. Opening start gate...")
    start_gate.set()
    _log(f"Capture running (pacing on {pace_name}). Press Ctrl+C to stop.")

    idx = 0
    t0 = time.perf_counter()
    miss: dict[str, int] = {}
    seen_ts: dict[str, object] = {}

    try:
        while not stop.is_set():
            paced, paced_ts = pace_buf.wait_for_new(timeout=0.1)
            pump_qt()

            if paced is None:
                sc, _ = sensor_cam_buf.get()
                ov, _ = overhead_cam_buf.get()
                sd, _ = spad_buf.get()
                if sc is not None or ov is not None or sd is not None:
                    _update_display(manager, idx, sd, sc, ov, cir_state)
                continue

            spad_data = spad_ts = None
            sensor_cam_data = sensor_cam_ts = None
            overhead_cam_data = overhead_cam_ts = None

            if cfg.USE_SPAD:
                if pace_buf is spad_buf:
                    spad_data, spad_ts = paced, paced_ts
                else:
                    spad_data, spad_ts = spad_buf.get()
                if _check_fresh("spad", spad_data, spad_ts, miss, seen_ts):
                    writer.write_spad(spad_data, spad_ts)

            if cfg.USE_SENSOR_CAM:
                if pace_buf is sensor_cam_buf:
                    sensor_cam_data, sensor_cam_ts = paced, paced_ts
                else:
                    sensor_cam_data, sensor_cam_ts = sensor_cam_buf.get()
                if _check_fresh("sensor_cam", sensor_cam_data, sensor_cam_ts, miss, seen_ts):
                    writer.write_sensor_cam(sensor_cam_data, sensor_cam_ts)

            if cfg.USE_OVERHEAD_CAM:
                if pace_buf is overhead_cam_buf:
                    overhead_cam_data, overhead_cam_ts = paced, paced_ts
                else:
                    overhead_cam_data, overhead_cam_ts = overhead_cam_buf.get()
                if _check_fresh("overhead_cam", overhead_cam_data, overhead_cam_ts, miss, seen_ts):
                    writer.write_overhead_cam(overhead_cam_data, overhead_cam_ts)

            _update_display(manager, idx, spad_data, sensor_cam_data,
                            overhead_cam_data, cir_state)
            idx += 1

            if idx % 30 == 0:
                elapsed = time.perf_counter() - t0
                _log(f"{idx} frames | {idx / elapsed:.1f} fps | {elapsed:.1f}s")
    except KeyboardInterrupt:
        pass

    # Finalize BEFORE joining workers — _join_workers can block on stuck
    # threads, and a second Ctrl+C during join triggers os._exit(1) which
    # skips all finally blocks.  Writing manifest + spad.npz first ensures
    # captured data is saved even if thread cleanup is interrupted.
    writer.finalize()

    _join_workers(stop, threads)
    elapsed = time.perf_counter() - t0
    fps = idx / elapsed if elapsed > 0 else 0
    _log(f"Stopped. {idx} frames in {elapsed:.1f}s ({fps:.1f} fps)")
    return idx


def run_manual(
    manager: Manager, writer: RunWriter,
    spad_buf: LatestFrame, sensor_cam_buf: LatestFrame, overhead_cam_buf: LatestFrame,
    stop: threading.Event, start_gate: threading.Event,
    cir_state=None,
):
    threads = _start_workers(
        manager, spad_buf, sensor_cam_buf, overhead_cam_buf, stop, start_gate,
    )

    _log("All workers ready. Opening start gate...")
    start_gate.set()
    time.sleep(0.5)

    idx = 0
    miss: dict[str, int] = {}
    seen_ts: dict[str, object] = {}
    try:
        while not stop.is_set():
            pump_qt()
            cmd = input(f"[iter {idx}] Enter=capture, q=quit: ").strip().lower()
            if cmd == "q":
                break

            spad_data = sensor_cam_data = overhead_cam_data = None

            if cfg.USE_SPAD:
                spad_data, spad_ts = spad_buf.get()
                if _check_fresh("spad", spad_data, spad_ts, miss, seen_ts):
                    writer.write_spad(spad_data, spad_ts)
                else:
                    _log("Warning: no fresh SPAD data")

            if cfg.USE_SENSOR_CAM:
                sensor_cam_data, sensor_cam_ts = sensor_cam_buf.get()
                if _check_fresh("sensor_cam", sensor_cam_data, sensor_cam_ts, miss, seen_ts):
                    writer.write_sensor_cam(sensor_cam_data, sensor_cam_ts)
                else:
                    _log("Warning: no fresh sensor_cam data")

            if cfg.USE_OVERHEAD_CAM:
                overhead_cam_data, overhead_cam_ts = overhead_cam_buf.get()
                if _check_fresh("overhead_cam", overhead_cam_data, overhead_cam_ts, miss, seen_ts):
                    writer.write_overhead_cam(overhead_cam_data, overhead_cam_ts)
                else:
                    _log("Warning: no fresh overhead_cam data")

            _update_display(manager, idx, spad_data, sensor_cam_data,
                            overhead_cam_data, cir_state)
            _log(f"Captured iter {idx}")
            idx += 1
    except KeyboardInterrupt:
        pass

    writer.finalize()
    _join_workers(stop, threads)
    return idx


# ── Main ───────────────────────────────────────────────────────────────

def main():
    _log("=== Configuration ===")
    _log(f"  SPAD         = {cfg.USE_SPAD}")
    _log(f"  sensor_cam   = {cfg.USE_SENSOR_CAM}")
    _log(f"  overhead_cam = {cfg.USE_OVERHEAD_CAM}")
    _log(f"  UWB          = {cfg.USE_UWB}")
    _log(f"  mode         = {cfg.CAPTURE_MODE}")

    # ── Run naming ────────────────────────────────────────────────────
    default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        run_name = input(f"\n[full_capture] Run name [{default_name}]: ").strip()
    except EOFError:
        run_name = ""
    if not run_name:
        run_name = default_name

    logdir = Path("data/logs") / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(logdir).free / (1 << 30)
    if free_gb < 10:
        _log(f"\033[1;33m[WARNING] Only {free_gb:.1f} GB free on {logdir.resolve().drive}\\ "
             f"— capture may fill disk\033[0m")
    _log(f"  output = {logdir}")

    # ── Shared events ─────────────────────────────────────────────────
    stop = threading.Event()
    start_gate = threading.Event()

    global _stop_event
    _stop_event = stop

    # ── UWB: discover + arm (waits on start_gate before capturing) ────
    uwb_threads: list[threading.Thread] = []
    cir_bufs: dict[str, LatestFrame] = {}
    if cfg.USE_UWB:
        _log("=== UWB Setup ===")
        from capture_uwb import start_uwb_threads

        if cfg.SHOW_UWB_CIR_PREVIEW:
            for role in RX_ROLES:
                cir_bufs[role] = LatestFrame()

        baud = getattr(cfg, "UWB_BAUD", 115200)
        uwb_threads = start_uwb_threads(
            logdir, stop, baud=baud,
            start_gate=start_gate,
            cir_bufs=cir_bufs if cir_bufs else None,
        )
        if not uwb_threads:
            _log("No UWB boards found — continuing without UWB.")
    else:
        _log("UWB disabled.")

    # ── SPAD + cameras ────────────────────────────────────────────────
    _log("=== Sensor Setup ===")

    spad_buf         = LatestFrame()
    sensor_cam_buf   = LatestFrame()
    overhead_cam_buf = LatestFrame()

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
            metadata = {
                "object": cfg.OBJECT,
                "run_name": run_name,
                "start_time": datetime.now().isoformat(),
                "sensors": {},
            }

            if cfg.USE_SPAD:
                metadata["sensors"]["spad"] = setup_spad(manager)
            for key, cam in cameras.items():
                metadata["sensors"][key] = _setup_camera(cam, manager, key)

            if cfg.USE_SENSOR_CAM and cfg.SHOW_SENSOR_CAM_PREVIEW:
                cv2.namedWindow(SENSOR_CAM_WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(SENSOR_CAM_WINDOW, cfg.SENSOR_CAM_WIDTH * 2,
                                 cfg.SENSOR_CAM_HEIGHT)
                cv2.moveWindow(SENSOR_CAM_WINDOW, 10, 10)
            if cfg.USE_OVERHEAD_CAM and cfg.SHOW_OVERHEAD_CAM_PREVIEW:
                cv2.namedWindow(OVERHEAD_CAM_WINDOW, cv2.WINDOW_NORMAL)
                _ov_w = cfg.OVERHEAD_CAM_WIDTH  // 2
                _ov_h = cfg.OVERHEAD_CAM_HEIGHT // 2
                cv2.resizeWindow(OVERHEAD_CAM_WINDOW, _ov_w, _ov_h)
                cv2.moveWindow(OVERHEAD_CAM_WINDOW, 10,
                               10 + cfg.SENSOR_CAM_HEIGHT + 40)

            # Live CIR figure (matplotlib, shares Qt event loop)
            cir_state = None
            if cir_bufs and cfg.SHOW_UWB_CIR_PREVIEW:
                _log("Setting up live CIR plot...")
                fig, axes, lines, fp_lines = _setup_cir_figure()
                cir_state = (fig, axes, lines, fp_lines, cir_bufs)

            writer = RunWriter(logdir, metadata)

            active = [n for n, on in [
                ("SPAD",                      cfg.USE_SPAD),
                ("sensor_cam (RealSense)",    cfg.USE_SENSOR_CAM),
                ("overhead_cam (eMeet C960)", cfg.USE_OVERHEAD_CAM),
                (f"UWB ({len(uwb_threads)} boards)", bool(uwb_threads)),
            ] if on]

            _log("=== Capture ===")
            _log(f"Active sensors: {', '.join(active)}")

            try:
                if cfg.CAPTURE_MODE == "loop":
                    count = run_loop(
                        manager, writer, spad_buf, sensor_cam_buf,
                        overhead_cam_buf, stop, start_gate, cir_state,
                    )
                else:
                    count = run_manual(
                        manager, writer, spad_buf, sensor_cam_buf,
                        overhead_cam_buf, stop, start_gate, cir_state,
                    )
            finally:
                writer.finalize()
                # Close dashboard before Manager.__exit__ to avoid
                # "wrapped C/C++ object has been deleted" RuntimeError.
                if "dashboard" in manager.components:
                    try:
                        manager.components["dashboard"].close()
                    except Exception:
                        pass
                    del manager.components["dashboard"]

    finally:
        for cam in cameras.values():
            cam.close()
        cv2.destroyAllWindows()
        plt.close("all")

    if uwb_threads:
        _log("Waiting for UWB threads to finish writing...")
        for t in uwb_threads:
            t.join(timeout=10)

    counts = writer.counts
    _log(f"Done. {counts} -> {logdir.resolve()}")


if __name__ == "__main__":
    main()
