"""Safety and supervision logic used by the device panels (pure Python, testable).

* :class:`PowerInterlock` - trips the RF amplifier when the reflected power
  exceeds a limit (the "Cavity Power Interlock" of the original software).
* :class:`VoltageRamp` - rate-limited high-voltage changes (kV/min).
* :class:`BreakdownDetector` - flags sudden voltage drops (arcs).
* :class:`RippleMeter` - short-term voltage ripple.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class PowerInterlock:
    """Latching over-limit detector.

    :meth:`check` returns True exactly once, when an armed interlock first
    sees a value above the limit; it then stays tripped until :meth:`reset`.
    """

    def __init__(self, limit: float, armed: bool = False) -> None:
        self.limit = float(limit)
        self.armed = armed
        self.tripped = False

    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def reset(self) -> None:
        self.tripped = False

    def check(self, value: float) -> bool:
        if self.armed and not self.tripped and value > self.limit:
            self.tripped = True
            return True
        return False


class VoltageRamp:
    """Steps a set point from ``start`` to ``target`` at ``rate_per_min``.

    One call to :meth:`next_setpoint` per ``interval_s`` (5 s in the original
    software). Unlike the original, the last step stops exactly at the target
    instead of overshooting it, and ramps down work too.
    """

    def __init__(self, start: float, target: float, rate_per_min: float, interval_s: float = 5.0) -> None:
        if rate_per_min <= 0 or interval_s <= 0:
            raise ValueError("rate and interval must be positive")
        self.current = float(start)
        self.target = float(target)
        self.interval_s = float(interval_s)
        self.step = rate_per_min * interval_s / 60.0

    @property
    def done(self) -> bool:
        return self.current == self.target

    def next_setpoint(self) -> float | None:
        """The next value to program, or None when the target has been reached."""
        if self.done:
            return None
        remaining = self.target - self.current
        if abs(remaining) <= self.step:
            self.current = self.target
        else:
            self.current += self.step if remaining > 0 else -self.step
        return self.current


class BreakdownDetector:
    """Detects a voltage drop of at least ``drop`` between consecutive readings."""

    def __init__(self, drop: float = 1.0) -> None:
        self.drop = float(drop)
        self._previous: float | None = None

    def update(self, voltage: float, output_on: bool = True) -> bool:
        previous = self._previous
        self._previous = voltage if output_on else None
        return output_on and previous is not None and previous - voltage >= self.drop


class RippleMeter:
    """Standard deviation of the last ``window`` readings."""

    def __init__(self, window: int = 10) -> None:
        self._values: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._values.append(value)
        return float(np.std(self._values)) if len(self._values) > 1 else 0.0
