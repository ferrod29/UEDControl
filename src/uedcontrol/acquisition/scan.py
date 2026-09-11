"""Pump-probe scan engine, independent of the GUI.

For every point: move the delay stage, wait, grab and average frames,
subtract the background, analyse the ROI, accumulate, store. The GUI runs
:meth:`PumpProbeScan.run` in a worker thread and receives each point through
``on_point``; scripts can use it directly::

    scan = PumpProbeScan(camera, delay_line, ScanSettings(delays_from_range(-5, 20, 0.5)))
    scan.run(threading.Event())

Without a delay line the scan is a time series: the "delays" are just
dataset numbers (the "continuous" mode of the original software).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..analysis.beam import CircularROI, RoiStatistics, analyze_roi
from ..analysis.pumpprobe import PumpProbeAccumulator, scan_order
from ..devices.base import DeviceError
from ..devices.interfaces import Camera
from .delay import DelayLine

if TYPE_CHECKING:
    from .storage import ScanWriter


@dataclass
class ScanSettings:
    delays_ps: np.ndarray
    frames_per_point: int = 1
    runs: int = 1  # 0 repeats until stopped
    bidirectional: bool = True
    settle_time_s: float = 0.0
    roi: CircularROI | None = None
    background: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.delays_ps = np.asarray(self.delays_ps, dtype=float).ravel()
        if self.delays_ps.size == 0:
            raise ValueError("a scan needs at least one delay")
        if self.frames_per_point < 1:
            raise ValueError("frames_per_point must be >= 1")
        if self.runs < 0:
            raise ValueError("runs must be >= 0")

    @property
    def total_points(self) -> int | None:
        return None if self.runs == 0 else self.runs * self.delays_ps.size


@dataclass(frozen=True)
class ScanPoint:
    number: int  # sequential, from 1
    run: int
    index: int
    delay_ps: float
    image: np.ndarray
    stats: RoiStatistics
    timestamp: float


PointCallback = Callable[[ScanPoint, PumpProbeAccumulator], None]


def average_frames(camera: Camera, count: int, stop: threading.Event | None = None) -> np.ndarray:
    """Mean of ``count`` frames (fewer if ``stop`` is set meanwhile)."""
    total: np.ndarray | None = None
    grabbed = 0
    for _ in range(max(1, count)):
        if grabbed and stop is not None and stop.is_set():
            break
        frame = np.asarray(camera.grab(), dtype=np.float64)
        total = frame if total is None else total + frame
        grabbed += 1
    assert total is not None
    return total / grabbed


class PumpProbeScan:
    def __init__(
        self,
        camera: Camera,
        delay_line: DelayLine | None,
        settings: ScanSettings,
        *,
        writer: ScanWriter | None = None,
        on_point: PointCallback | None = None,
    ) -> None:
        self.camera = camera
        self.delay_line = delay_line
        self.settings = settings
        self.writer = writer
        self.on_point = on_point
        self.accumulator = PumpProbeAccumulator(settings.delays_ps)

    def check_delays(self) -> None:
        """Fail before the first point, not halfway through, if a delay is out of the stage range."""
        if self.delay_line is None:
            return
        low, high = self.delay_line.delay_limits()
        delays = self.settings.delays_ps
        outside = delays[(delays < low) | (delays > high)]
        if outside.size:
            raise DeviceError(
                f"{outside.size} delay(s) outside the stage range {low:.1f}..{high:.1f} ps "
                f"(e.g. {outside[0]:g} ps); adjust the delays or T0"
            )

    def run(self, stop: threading.Event) -> PumpProbeAccumulator:
        settings = self.settings
        self.check_delays()
        order = scan_order(settings.delays_ps.size, settings.runs or None, settings.bidirectional)
        if self.writer is not None:
            self.writer.open(settings)
        try:
            for number, (run, index) in enumerate(order, start=1):
                if stop.is_set():
                    break
                delay = float(settings.delays_ps[index])
                if self.delay_line is not None:
                    self.delay_line.move_to_delay(delay, wait=True)
                if settings.settle_time_s > 0 and stop.wait(settings.settle_time_s):
                    break
                image = average_frames(self.camera, settings.frames_per_point, stop)
                background = settings.background
                if background is not None and background.shape == image.shape:
                    image = image - background
                stats = analyze_roi(image, settings.roi)
                self.accumulator.add(index, stats.counts.total, stats.rms_contrast, stats.correlation)
                point = ScanPoint(number, run, index, delay, image, stats, time.time())
                if self.writer is not None:
                    self.writer.write_point(point)
                if self.on_point is not None:
                    self.on_point(point, self.accumulator)
        finally:
            if self.writer is not None:
                self.writer.write_summary(self.accumulator)
                self.writer.close()
        return self.accumulator
