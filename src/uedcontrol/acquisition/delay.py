"""Pump-probe delay line: conversion between delays (ps) and stage positions.

A retro-reflector on the stage changes the optical path by ``passes`` times
the stage displacement, so ``Δt = passes · Δz / c``. With the default of two
passes, 1 ps corresponds to 0.1499 mm; the original software used 0.15 mm/ps
(153 mm / 1020 ps) for the Mercury stage and 1/6.6666 mm/ps for the SMC100.
Unlike the original, setting T0 now really changes where delays are measured
from (the old code kept using the hard-coded 153 mm).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..constants import SPEED_OF_LIGHT_MM_PER_PS
from ..devices.base import DeviceError
from ..devices.interfaces import Positioner


class DelayLine:
    def __init__(self, stage: Positioner, t0_position: float, passes: float = 2.0, direction: int = 1) -> None:
        if passes <= 0:
            raise ValueError("passes must be positive")
        if direction not in (1, -1):
            raise ValueError("direction must be +1 or -1")
        self.stage = stage
        self.t0_position = float(t0_position)
        self.passes = float(passes)
        self.direction = direction

    @property
    def mm_per_ps(self) -> float:
        return SPEED_OF_LIGHT_MM_PER_PS / self.passes

    def position_for(self, delay_ps: float) -> float:
        return self.t0_position + self.direction * delay_ps * self.mm_per_ps

    def delay_for(self, position: float) -> float:
        return (position - self.t0_position) / (self.direction * self.mm_per_ps)

    def delay_limits(self) -> tuple[float, float]:
        low, high = self.stage.limits()
        a, b = self.delay_for(low), self.delay_for(high)
        return (min(a, b), max(a, b))

    def current_delay(self) -> float:
        return self.delay_for(self.stage.position())

    def move_to_delay(self, delay_ps: float, wait: bool = True) -> None:
        low, high = self.delay_limits()
        if not low <= delay_ps <= high:
            raise DeviceError(f"delay {delay_ps:g} ps is outside the stage range {low:.1f}..{high:.1f} ps")
        self.stage.move_to(self.position_for(delay_ps), wait=wait)

    def set_t0_here(self) -> float:
        """Define the current stage position as time zero and return it."""
        self.t0_position = self.stage.position()
        return self.t0_position


def delays_from_range(start: float, stop: float, step: float) -> np.ndarray:
    """Delays from ``start`` to ``stop`` (inclusive when it falls on the grid)."""
    if step == 0:
        raise ValueError("step must not be zero")
    if (stop - start) * step < 0:
        raise ValueError("step has the wrong sign for this range")
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    return np.round(start + step * np.arange(count), 6)


def delays_from_segments(segments: Iterable[tuple[float, float, float]]) -> np.ndarray:
    """Union of several ranges, e.g. fine steps around T0 and coarse steps later."""
    parts = [delays_from_range(*segment) for segment in segments]
    if not parts:
        raise ValueError("no segments given")
    return np.unique(np.concatenate(parts))


def parse_segments(text: str) -> list[tuple[float, float, float]]:
    """Parse ``start:stop:step`` segments separated by ``;`` or new lines.

    Commas are not separators, so that a decimal comma (``0,5``) is reported
    as an error instead of silently splitting a segment.
    """
    segments = []
    for part in re.split(r"[;\n]+", text):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 3:
            raise ValueError(f"segment {part!r} is not start:stop:step")
        try:
            start, stop, step = (float(field) for field in fields)
        except ValueError:
            raise ValueError(f"segment {part!r} contains something that is not a number") from None
        segments.append((start, stop, step))
    return segments


def delays_from_file(path: str | Path) -> np.ndarray:
    """One delay per line (first column if there are several)."""
    data = np.loadtxt(path, ndmin=2)
    if data.size == 0:
        raise ValueError(f"{path} contains no delays")
    return data[:, 0].astype(float)
