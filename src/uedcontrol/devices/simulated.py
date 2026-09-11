"""Simulated instruments.

They implement the same interfaces as the real drivers, so the complete
application - pump-probe scans included - runs on a laptop without hardware.
The GUI smoke tests use them too.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

from .base import Action, Device, DeviceError, Reading, Setting
from .interfaces import (
    Camera,
    HighVoltageSupply,
    PiezoMirrorController,
    Positioner,
    PowerSupply,
    RFAmplifier,
    RFAttenuator,
    RFPowerSensor,
)


class SimulatedCamera(Camera):
    """Synthetic detector frames: an electron beam spot, powder rings or Bragg peaks.

    When :attr:`delay_source` is set (the GUI links it to the delay line) the
    diffraction intensity drops after time zero with a 3 ps time constant, so a
    pump-probe scan produces a recognisable transient.
    """

    model = "Simulated camera"
    PATTERNS = ("electron_beam", "diffraction_rings", "diffraction_peaks")

    def __init__(
        self,
        *,
        name: str | None = None,
        pattern: str = "diffraction_rings",
        shape: tuple[int, int] = (512, 512),
        pixel_size_um: float = 5.0,
        exposure_ms: float = 100.0,
        max_frame_rate: float = 20.0,
        noise: float = 0.03,
        realtime: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(name)
        if pattern not in self.PATTERNS:
            raise ValueError(f"pattern must be one of {self.PATTERNS}")
        self.pattern = pattern
        self.shape = (int(shape[0]), int(shape[1]))
        self.pixel_size_um = float(pixel_size_um)
        self.max_frame_rate = float(max_frame_rate)
        self.noise = float(noise)
        self.realtime = bool(realtime)
        self.delay_source: Callable[[], float] | None = None
        self._exposure_ms = float(exposure_ms)
        self._rng = np.random.default_rng(seed)
        self._next_frame = 0.0
        rows, cols = np.indices(self.shape, dtype=np.float32)
        self._y = rows - self.shape[0] / 2
        self._x = cols - self.shape[1] / 2
        self._r = np.hypot(self._x, self._y)
        self._beam, self._diffraction = self._static_patterns()

    def _static_patterns(self) -> tuple[np.ndarray, np.ndarray]:
        size = min(self.shape)
        beam = np.exp(-(self._r**2) / (2 * (0.02 * size) ** 2))
        diffraction = np.zeros(self.shape, dtype=np.float32)
        if self.pattern == "diffraction_rings":
            for fraction, amplitude in ((0.16, 0.6), (0.26, 0.35), (0.36, 0.2)):
                diffraction += amplitude * np.exp(-((self._r - fraction * size) ** 2) / (2 * (0.008 * size) ** 2))
        elif self.pattern == "diffraction_peaks":
            width = 0.012 * size
            for order, amplitude in ((1, 0.6), (2, 0.25)):
                radius = 0.18 * size * order
                for k in range(6):
                    angle = k * math.pi / 3
                    cx, cy = radius * math.cos(angle), radius * math.sin(angle)
                    diffraction += amplitude * np.exp(
                        -((self._x - cx) ** 2 + (self._y - cy) ** 2) / (2 * width**2)
                    )
        return beam.astype(np.float32), diffraction

    def _connect(self) -> None:
        self._next_frame = time.monotonic()

    def _disconnect(self) -> None:
        pass

    def set_exposure(self, exposure_ms: float) -> None:
        if exposure_ms <= 0:
            raise ValueError("exposure must be positive")
        self._exposure_ms = float(exposure_ms)

    def exposure(self) -> float:
        return self._exposure_ms

    def frame_rate(self) -> float:
        return min(self.max_frame_rate, 1000.0 / self._exposure_ms)

    def pump_response(self) -> float:
        if self.delay_source is None:
            return 1.0
        delay = self.delay_source()
        return 1.0 - 0.2 * (1.0 - math.exp(-delay / 3.0)) if delay > 0 else 1.0

    def _signal(self) -> np.ndarray:
        if self.pattern == "electron_beam":
            size = min(self.shape)
            dx, dy = self._rng.normal(0.0, 0.01 * size, 2)
            sigma = 0.06 * size
            return np.exp(-((self._x - dx) ** 2 + (self._y - dy) ** 2) / (2 * sigma**2))
        return self._beam + self.pump_response() * self._diffraction

    def grab(self) -> np.ndarray:
        self.ensure_connected()
        if self.realtime:
            now = time.monotonic()
            if self._next_frame > now:
                time.sleep(self._next_frame - now)
            self._next_frame = max(now, self._next_frame) + 1.0 / self.frame_rate()
        peak_counts = 10.0 * self._exposure_ms
        image = peak_counts * self._signal() + self._rng.normal(0.0, self.noise * peak_counts, self.shape)
        return np.clip(image, 0, None).astype(np.float32)


class SimulatedStage(Positioner):
    """Linear stage that moves at constant speed (instantly if ``realtime`` is False)."""

    model = "Simulated linear stage"

    def __init__(
        self,
        *,
        name: str | None = None,
        position: float = 150.0,
        limits: tuple[float, float] = (0.0, 300.0),
        speed: float = 100.0,
        unit: str = "mm",
        realtime: bool = True,
    ) -> None:
        super().__init__(name)
        self.unit = unit
        self.speed = float(speed)
        self.realtime = bool(realtime)
        self._limits = (float(limits[0]), float(limits[1]))
        self._position = float(position)
        self._motion: tuple[float, float, float, float] | None = None

    def _connect(self) -> None:
        pass

    def _disconnect(self) -> None:
        self.stop()

    def _update(self) -> float:
        if self._motion is not None:
            start, target, t0, duration = self._motion
            elapsed = time.monotonic() - t0
            if elapsed >= duration:
                self._position, self._motion = target, None
            else:
                self._position = start + (target - start) * elapsed / duration
        return self._position

    def position(self) -> float:
        with self.lock:
            return self._update()

    def is_moving(self) -> bool:
        with self.lock:
            self._update()
            return self._motion is not None

    def move_to(self, position: float, wait: bool = True) -> None:
        low, high = self._limits
        if not low <= position <= high:
            raise DeviceError(f"{self.name}: target {position:g} {self.unit} outside {low:g}..{high:g}")
        with self.lock:
            start = self._update()
            duration = abs(position - start) / self.speed if self.realtime else 0.0
            self._motion = (start, float(position), time.monotonic(), duration)
            self._update()
        if wait:
            self.wait_until_stopped()

    def stop(self) -> None:
        with self.lock:
            self._update()
            self._motion = None

    def home(self) -> None:
        low, high = self._limits
        self.move_to(min(max(0.0, low), high))

    def limits(self) -> tuple[float, float]:
        return self._limits


class SimulatedPowerSupply(PowerSupply):
    """Multi-channel supply feeding resistive loads."""

    model = "Simulated power supply"
    has_master_output = True

    def __init__(self, *, name: str | None = None, n_channels: int = 3, load_ohm: float = 5.0) -> None:
        super().__init__(name)
        self.n_channels = int(n_channels)
        self.load_ohm = float(load_ohm)
        self._voltage = [0.0] * self.n_channels
        self._current = [1.0] * self.n_channels
        self._output = [False] * self.n_channels
        self._master = True
        self._rng = np.random.default_rng()

    def _connect(self) -> None:
        pass

    def _disconnect(self) -> None:
        pass

    def _index(self, channel: int) -> int:
        if not 1 <= channel <= self.n_channels:
            raise ValueError(f"channel must be 1..{self.n_channels}")
        return channel - 1

    def set_voltage(self, value: float, channel: int = 1) -> None:
        self._voltage[self._index(channel)] = float(value)

    def set_current(self, value: float, channel: int = 1) -> None:
        self._current[self._index(channel)] = float(value)

    def measure_voltage(self, channel: int = 1) -> float:
        k = self._index(channel)
        if not (self._output[k] and self._master):
            return 0.0
        voltage = min(self._voltage[k], self._current[k] * self.load_ohm)
        return voltage * (1 + 1e-4 * self._rng.standard_normal())

    def measure_current(self, channel: int = 1) -> float:
        return self.measure_voltage(channel) / self.load_ohm

    def voltage_setpoint(self, channel: int = 1) -> float:
        return self._voltage[self._index(channel)]

    def current_setpoint(self, channel: int = 1) -> float:
        return self._current[self._index(channel)]

    def set_output(self, enabled: bool, channel: int = 1) -> None:
        self._output[self._index(channel)] = bool(enabled)

    def output_enabled(self, channel: int = 1) -> bool:
        return self._output[self._index(channel)]

    def set_master_output(self, enabled: bool) -> None:
        self._master = bool(enabled)

    def master_output_enabled(self) -> bool:
        return self._master


class SimulatedHVSupply(HighVoltageSupply):
    """Gun supply with an interlock, a small ripple and a leakage current."""

    model = "Simulated HV supply"
    has_interlock = True

    def __init__(self, *, name: str | None = None, leak_ua_per_kv: float = 0.05, ripple_kv: float = 0.003) -> None:
        super().__init__(name)
        self.leak_ua_per_kv = leak_ua_per_kv
        self.ripple_kv = ripple_kv
        self._setpoint = 0.0
        self._current_limit = 100.0
        self._output = False
        self._interlock = False
        self._rng = np.random.default_rng()

    def _connect(self) -> None:
        pass

    def _disconnect(self) -> None:
        pass

    def set_voltage(self, value: float, channel: int = 1) -> None:
        self._setpoint = float(value)

    def set_current(self, value: float, channel: int = 1) -> None:
        self._current_limit = float(value)

    def voltage_setpoint(self, channel: int = 1) -> float:
        return self._setpoint

    def current_setpoint(self, channel: int = 1) -> float:
        return self._current_limit

    def measure_voltage(self, channel: int = 1) -> float:
        live = self._output and self._interlock
        base = self._setpoint if live else 0.0
        return max(0.0, base + self.ripple_kv * self._rng.standard_normal())

    def measure_current(self, channel: int = 1) -> float:
        return min(self._current_limit, self.leak_ua_per_kv * self.measure_voltage())

    def set_output(self, enabled: bool, channel: int = 1) -> None:
        if enabled and not self._interlock:
            raise DeviceError(f"{self.name}: interlock open, output stays off")
        self._output = bool(enabled)

    def output_enabled(self, channel: int = 1) -> bool:
        return self._output

    def interlock_closed(self) -> bool:
        return self._interlock

    def set_interlock(self, closed: bool) -> None:
        self._interlock = bool(closed)
        if not closed:
            self._output = False


class SimulatedMirrors(PiezoMirrorController):
    """Agilis-like mirror controller that just counts steps."""

    model = "Simulated piezo mirrors"
    jog_speeds = ((1, "slow"), (2, "medium"), (3, "fast"))

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name)
        self.channel = 1
        self.amplitude = 16
        self.steps: dict[tuple[int, int], int] = {}
        self._jogging: set[tuple[int, int]] = set()

    def _connect(self) -> None:
        pass

    def _disconnect(self) -> None:
        self._jogging.clear()

    def select_channel(self, channel: int) -> None:
        if not 1 <= channel <= self.n_channels:
            raise ValueError(f"channel must be 1..{self.n_channels}")
        self.channel = channel

    def set_step_amplitude(self, amplitude: int) -> None:
        self.amplitude = int(amplitude)

    def step(self, axis: int, steps: int) -> None:
        key = (self.channel, axis)
        self.steps[key] = self.steps.get(key, 0) + int(steps)

    def jog(self, axis: int, speed: int) -> None:
        key = (self.channel, axis)
        if speed:
            self._jogging.add(key)
        else:
            self._jogging.discard(key)

    def stop(self, axis: int | None = None) -> None:
        for ax in (axis,) if axis else range(1, self.n_axes + 1):
            self._jogging.discard((self.channel, ax))

    def is_moving(self, axis: int) -> bool:
        return (self.channel, axis) in self._jogging

    def read_status(self) -> dict[str, Reading]:
        status = {"Channel": Reading(self.channel)}
        for axis in range(1, self.n_axes + 1):
            status[f"Axis {axis} steps"] = Reading(self.steps.get((self.channel, axis), 0))
        return status


class SimulatedVacuumGauge(Device):
    model = "Simulated vacuum gauge"

    def __init__(self, *, name: str | None = None, pressure_mbar: float = 2e-8) -> None:
        super().__init__(name)
        self._pressure = pressure_mbar
        self._on = True
        self._rng = np.random.default_rng()

    def _connect(self) -> None:
        pass

    def _disconnect(self) -> None:
        pass

    def read_status(self) -> dict[str, Reading]:
        if not self._on:
            return {"Pressure": Reading("-", "mbar"), "Gauge": Reading(False)}
        pressure = self._pressure * (1 + 0.05 * self._rng.standard_normal())
        return {"Pressure": Reading(pressure, "mbar"), "Gauge": Reading(True)}

    def actions(self) -> list[Action]:
        return [Action("Gauge on", lambda: setattr(self, "_on", True)),
                Action("Gauge off", lambda: setattr(self, "_on", False))]


class SimulatedRFAmplifier(RFAmplifier):
    model = "Simulated RF amplifier"

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name)
        self._enabled = False

    def _connect(self) -> None:
        pass

    def _disconnect(self) -> None:
        pass

    def rf_enabled(self) -> bool:
        return self._enabled

    def set_rf_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def read_status(self) -> dict[str, Reading]:
        return {"RF output": Reading(self._enabled)}


class SimulatedRFAttenuator(RFAttenuator):
    model = "Simulated RF attenuator"

    def __init__(self, *, name: str | None = None, attenuation_db: float = 20.0) -> None:
        super().__init__(name)
        self._attenuation = float(attenuation_db)

    def _connect(self) -> None:
        pass

    def _disconnect(self) -> None:
        pass

    def attenuation(self) -> float:
        return self._attenuation

    def set_attenuation(self, attenuation_db: float) -> None:
        self._attenuation = float(attenuation_db)

    def read_status(self) -> dict[str, Reading]:
        return {"Attenuation": Reading(self._attenuation, "dB")}

    def settings(self) -> list[Setting]:
        return [Setting("attenuation", "Attenuation", self.set_attenuation, unit="dB",
                        minimum=0, maximum=120, decimals=2, initial=self._attenuation)]


class SimulatedRFPowerSensor(RFPowerSensor):
    """Reads ``level_dbm`` (plus noise) while :attr:`source` reports RF on, else the noise floor."""

    model = "Simulated RF power sensor"

    def __init__(
        self, *, name: str | None = None, level_dbm: float = -20.0, coupling_db: float = 30.0
    ) -> None:
        super().__init__(name)
        self.level_dbm = float(level_dbm)
        self.coupling_db = float(coupling_db)
        #: Callable telling whether RF power is present (linked to the amplifier by the GUI).
        self.source: Callable[[], bool] | None = None
        self._rng = np.random.default_rng()

    def _connect(self) -> None:
        pass

    def _disconnect(self) -> None:
        pass

    def power_dbm(self) -> float:
        live = self.source() if self.source is not None else True
        level = self.level_dbm if live else -60.0
        return level + 0.2 * self._rng.standard_normal()
