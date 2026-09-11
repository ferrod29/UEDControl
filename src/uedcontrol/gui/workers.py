"""Threads used by the GUI.

* :class:`DeviceController` - one thread per instrument panel. Every call to
  the instrument runs there, so a slow or unresponsive instrument never
  freezes the window (the original GUIs talked to the hardware from the GUI
  thread and from ad-hoc threads at the same time).
* :class:`AcquisitionWorker` - the camera thread: live view (frame
  averaging, background subtraction, ROI statistics, tracking), background
  captures and pump-probe scans.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..acquisition.delay import DelayLine
from ..acquisition.scan import PumpProbeScan, ScanPoint, ScanSettings, average_frames
from ..acquisition.storage import ScanWriter
from ..analysis.beam import CircularROI, analyze_roi, beam_charge, block_region
from ..analysis.pumpprobe import PumpProbeAccumulator
from ..devices.base import DeviceError
from ..devices.interfaces import Camera
from .qt import QtCore, Signal, Slot

log = logging.getLogger(__name__)


def describe(exc: BaseException) -> str:
    return str(exc) if isinstance(exc, DeviceError) else f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------- devices
class _DeviceWorker(QtCore.QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)
    polled = Signal(object)
    poll_failed = Signal(str)

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self._timer: QtCore.QTimer | None = None
        self._poll_function: Callable[[], Any] | None = None
        self._last_poll_error = ""

    @Slot(int, object)
    def run(self, token: int, function: Callable[[], Any]) -> None:
        try:
            result = function()
        except Exception as exc:
            log.exception("%s: command failed", self.name)
            self.failed.emit(token, describe(exc))
        else:
            self.finished.emit(token, result)

    @Slot(object, int)
    def start_polling(self, function: Callable[[], Any], interval_ms: int) -> None:
        self._poll_function = function
        if self._timer is None:
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self._poll)
        self._timer.start(max(interval_ms, 20))
        self._poll()

    @Slot()
    def stop_polling(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        self._poll_function = None

    def _poll(self) -> None:
        if self._poll_function is None:
            return
        try:
            result = self._poll_function()
        except Exception as exc:
            message = describe(exc)
            if message != self._last_poll_error:  # do not flood the log
                log.warning("%s: reading failed: %s", self.name, message)
                self._last_poll_error = message
            self.poll_failed.emit(message)
        else:
            self._last_poll_error = ""
            self.polled.emit(result)


class DeviceController(QtCore.QObject):
    """Runs calls on a dedicated thread and reports results back in the GUI thread."""

    status = Signal(object)  # result of each periodic poll
    poll_error = Signal(str)
    error = Signal(str)  # failure of a command submitted without a `failed` callback
    _run = Signal(int, object)
    _start = Signal(object, int)
    _stop = Signal()

    def __init__(self, name: str, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QtCore.QThread()
        self._thread.setObjectName(f"device-{name}")
        self._worker = _DeviceWorker(name)
        self._worker.moveToThread(self._thread)
        self._thread.finished.connect(self._worker.deleteLater)
        self._run.connect(self._worker.run)
        self._start.connect(self._worker.start_polling)
        self._stop.connect(self._worker.stop_polling)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.polled.connect(self.status)
        self._worker.poll_failed.connect(self.poll_error)
        self._callbacks: dict[int, tuple[Callable[[Any], None] | None, Callable[[str], None] | None]] = {}
        self._tokens = itertools.count()
        self._thread.start()

    def submit(
        self,
        function: Callable[[], Any],
        done: Callable[[Any], None] | None = None,
        failed: Callable[[str], None] | None = None,
    ) -> None:
        token = next(self._tokens)
        self._callbacks[token] = (done, failed)
        self._run.emit(token, function)

    def start_polling(self, function: Callable[[], Any], interval_s: float) -> None:
        self._start.emit(function, int(interval_s * 1000))

    def stop_polling(self) -> None:
        self._stop.emit()

    def shutdown(self, timeout_ms: int = 3000) -> None:
        self._stop.emit()
        self._thread.quit()
        if not self._thread.wait(timeout_ms):
            log.warning("thread %s did not stop in time", self._thread.objectName())

    def _on_finished(self, token: int, result: Any) -> None:
        done, _failed = self._callbacks.pop(token, (None, None))
        if done is not None:
            done(result)

    def _on_failed(self, token: int, message: str) -> None:
        _done, failed = self._callbacks.pop(token, (None, None))
        if failed is not None:
            failed(message)
        else:
            self.error.emit(message)


# ----------------------------------------------------------------- acquisition
@dataclass(frozen=True)
class LiveSettings:
    frames_to_average: int = 1
    roi: CircularROI | None = None
    subtract_background: bool = False
    block_region: CircularROI | None = None
    track: bool = False
    rep_rate_hz: float = 1000.0
    pulsed: bool = True
    counts_per_electron: float = 1.0


@dataclass(frozen=True)
class ScanProgress:
    """Thread-safe snapshot of a scan after one point."""

    point: ScanPoint
    delays: np.ndarray
    total: np.ndarray
    rms: np.ndarray
    profile: np.ndarray | None
    counts: np.ndarray

    @classmethod
    def snapshot(cls, point: ScanPoint, accumulator: PumpProbeAccumulator) -> ScanProgress:
        return cls(
            point=point,
            delays=accumulator.delays.copy(),
            total=accumulator.mean_total,
            rms=accumulator.mean_rms,
            profile=accumulator.mean_profile,
            counts=accumulator.counts.copy(),
        )


class AcquisitionWorker(QtCore.QObject):
    """Lives in the acquisition thread; slots are invoked through queued signals.

    The methods without ``@Slot`` (``update_settings``, ``request_stop``...)
    are called directly from the GUI thread and are thread-safe.
    """

    frame_ready = Signal(object, object, object)  # image, RoiStatistics | None, BeamCharge | None
    roi_moved = Signal(float, float)
    live_state = Signal(bool)
    captured = Signal(str, object)  # "background" / "reference", image
    scan_progress = Signal(object)  # ScanProgress
    scan_finished = Signal(object, str)  # PumpProbeAccumulator | None, error message ("" if fine)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.camera: Camera | None = None
        self._settings = LiveSettings()
        self._background: np.ndarray | None = None
        self._capture: list[Any] | None = None  # [kind, frames wanted, running sum, count]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._display_pending = False

    # ---------------------------------------------------- called from the GUI
    @property
    def settings(self) -> LiveSettings:
        with self._lock:
            return self._settings

    def update_settings(self, **changes: Any) -> None:
        with self._lock:
            self._settings = replace(self._settings, **changes)

    def set_background(self, image: np.ndarray | None) -> None:
        with self._lock:
            self._background = image

    def request_capture(self, kind: str, frames: int) -> None:
        with self._lock:
            self._capture = [kind, max(1, frames), None, 0]

    def request_stop(self) -> None:
        self._stop.set()

    def frame_consumed(self) -> None:
        self._display_pending = False

    # -------------------------------------------------- acquisition thread
    def _camera_ready(self) -> Camera | None:
        camera = self.camera
        if camera is None or not camera.connected:
            self.error.emit("no camera connected")
            return None
        return camera

    @staticmethod
    def _stop_camera(camera: Camera) -> None:
        try:
            camera.stop_acquisition()
        except Exception as exc:
            log.warning("stopping the camera failed: %s", exc)

    @Slot()
    def run_live(self) -> None:
        camera = self._camera_ready()
        if camera is None:
            self.live_state.emit(False)
            return
        self._stop.clear()
        self._display_pending = False
        self.live_state.emit(True)
        total: np.ndarray | None = None
        count = 0
        try:
            camera.start_acquisition()
            while not self._stop.is_set():
                frame = np.asarray(camera.grab(), dtype=np.float64)
                self._feed_capture(frame)
                with self._lock:
                    settings, background = self._settings, self._background
                if count == 0 or total is None or total.shape != frame.shape:
                    total, count = frame, 1
                else:
                    total, count = total + frame, count + 1
                if count >= settings.frames_to_average:
                    self._publish(total / count, settings, background, camera)
                    count = 0
        except Exception as exc:
            log.exception("live acquisition stopped")
            self.error.emit(describe(exc))
        finally:
            self._stop_camera(camera)
            self.live_state.emit(False)

    def _feed_capture(self, frame: np.ndarray) -> None:
        with self._lock:
            request = self._capture
            if request is None:
                return
            kind, wanted, total, count = request
            total = frame.copy() if total is None else total + frame
            count += 1
            done = total / count if count >= wanted else None
            self._capture = None if done is not None else [kind, wanted, total, count]
        if done is not None:
            self.captured.emit(kind, done)

    @Slot(str, int)
    def run_capture(self, kind: str, frames: int) -> None:
        """Average ``frames`` frames while the live view is off."""
        camera = self._camera_ready()
        if camera is None:
            return
        try:
            camera.start_acquisition()
            self.captured.emit(kind, average_frames(camera, frames))
        except Exception as exc:
            log.exception("capture failed")
            self.error.emit(describe(exc))
        finally:
            self._stop_camera(camera)

    def _publish(self, image: np.ndarray, settings: LiveSettings, background: np.ndarray | None,
                 camera: Camera) -> None:
        if settings.subtract_background and background is not None and background.shape == image.shape:
            image = image - background
        if settings.block_region is not None:
            with contextlib.suppress(ValueError):  # region outside this image
                image = block_region(image, settings.block_region)
        stats = charge = None
        if settings.roi is not None:
            try:
                stats = analyze_roi(image, settings.roi)
            except ValueError:  # ROI outside the image
                stats = None
        if stats is not None:
            try:
                charge = beam_charge(stats.counts, camera.exposure(), settings.rep_rate_hz,
                                     settings.pulsed, settings.counts_per_electron)
            except (ValueError, DeviceError):
                charge = None
            if settings.track:
                with self._lock:
                    if self._settings.roi is not None:
                        self._settings = replace(
                            self._settings, roi=self._settings.roi.moved_to(stats.centroid_x, stats.centroid_y)
                        )
                self.roi_moved.emit(stats.centroid_x, stats.centroid_y)
        if not self._display_pending:  # drop frames the GUI has no time to draw
            self._display_pending = True
            self.frame_ready.emit(image, stats, charge)

    @Slot(object, object, object)
    def run_scan(self, settings: ScanSettings, delay_line: DelayLine | None, writer: ScanWriter | None) -> None:
        camera = self.camera
        if camera is None or not camera.connected:
            self.scan_finished.emit(None, "no camera connected")
            return
        self._stop.clear()
        self._display_pending = False
        scan = PumpProbeScan(camera, delay_line, settings, writer=writer, on_point=self._on_point)
        message = ""
        try:
            camera.start_acquisition()
            scan.run(self._stop)
        except Exception as exc:
            log.exception("scan failed")
            message = describe(exc)
        finally:
            self._stop_camera(camera)
        self.scan_finished.emit(scan.accumulator, message)

    def _on_point(self, point: ScanPoint, accumulator: PumpProbeAccumulator) -> None:
        self.scan_progress.emit(ScanProgress.snapshot(point, accumulator))
        if not self._display_pending:
            self._display_pending = True
            self.frame_ready.emit(point.image, point.stats, None)
