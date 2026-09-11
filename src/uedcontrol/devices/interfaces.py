"""Category interfaces.

The GUI panels and the acquisition code program against these classes, never
against a concrete driver. Adding a new instrument therefore means writing one
driver that implements the right interface; nothing else has to change.
"""

from __future__ import annotations

import math
import time
from abc import abstractmethod

import numpy as np

from .base import Device, DeviceError, Reading


class Camera(Device):
    """An area detector returning 2-D images indexed ``[row, column]``."""

    #: Physical pixel pitch, used to convert widths to micrometres.
    pixel_size_um: float = 1.0
    #: Orientation corrections applied by :meth:`orient` (set from the configuration).
    transpose: bool = False
    flip_lr: bool = False
    flip_ud: bool = False

    @abstractmethod
    def set_exposure(self, exposure_ms: float) -> None:
        """Set the exposure time in milliseconds."""

    @abstractmethod
    def exposure(self) -> float:
        """Exposure time in milliseconds."""

    def frame_rate(self) -> float:
        """Frames per second the camera currently delivers."""
        exposure = self.exposure()
        return 1000.0 / exposure if exposure > 0 else 0.0

    def start_acquisition(self) -> None:
        """Prepare continuous acquisition. No-op for cameras that grab on demand."""

    def stop_acquisition(self) -> None:
        """Stop continuous acquisition."""

    @abstractmethod
    def grab(self) -> np.ndarray:
        """Block until the next frame is available and return it."""

    def orient(self, image: np.ndarray) -> np.ndarray:
        """Apply the configured corrections to a raw frame: flips first, then transpose."""
        if self.flip_lr:
            image = image[:, ::-1]
        if self.flip_ud:
            image = image[::-1, :]
        if self.transpose:
            image = image.T
        return np.ascontiguousarray(image)

    def read_status(self) -> dict[str, Reading]:
        with self.lock:
            return {
                "Exposure": Reading(self.exposure(), "ms"),
                "Frame rate": Reading(self.frame_rate(), "Hz"),
                "Pixel size": Reading(self.pixel_size_um, "µm"),
            }


class PowerSupply(Device):
    """DC power supply with one or more channels (channels are numbered from 1)."""

    n_channels: int = 1
    voltage_unit: str = "V"
    current_unit: str = "A"
    #: True when the supply has a global output switch in addition to per-channel ones.
    has_master_output: bool = False

    @abstractmethod
    def set_voltage(self, value: float, channel: int = 1) -> None: ...

    @abstractmethod
    def set_current(self, value: float, channel: int = 1) -> None: ...

    @abstractmethod
    def measure_voltage(self, channel: int = 1) -> float: ...

    @abstractmethod
    def measure_current(self, channel: int = 1) -> float: ...

    @abstractmethod
    def set_output(self, enabled: bool, channel: int = 1) -> None: ...

    @abstractmethod
    def output_enabled(self, channel: int = 1) -> bool: ...

    def voltage_setpoint(self, channel: int = 1) -> float | None:
        """Programmed voltage, if the instrument can report it."""
        return None

    def current_setpoint(self, channel: int = 1) -> float | None:
        """Programmed current limit, if the instrument can report it."""
        return None

    def set_master_output(self, enabled: bool) -> None:
        raise NotImplementedError(f"{self.model} has no master output switch")

    def master_output_enabled(self) -> bool:
        raise NotImplementedError(f"{self.model} has no master output switch")

    def read_status(self) -> dict[str, Reading]:
        status: dict[str, Reading] = {}
        with self.lock:
            for channel in range(1, self.n_channels + 1):
                prefix = f"CH{channel} " if self.n_channels > 1 else ""
                status[f"{prefix}Voltage"] = Reading(self.measure_voltage(channel), self.voltage_unit)
                status[f"{prefix}Current"] = Reading(self.measure_current(channel), self.current_unit)
                status[f"{prefix}Output"] = Reading(self.output_enabled(channel))
            if self.has_master_output:
                status["Master output"] = Reading(self.master_output_enabled())
        return status


class HighVoltageSupply(PowerSupply):
    """Single-channel high-voltage supply (electron gun), optionally with an interlock."""

    voltage_unit = "kV"
    current_unit = "µA"
    #: True when :meth:`interlock_closed` / :meth:`set_interlock` are available.
    has_interlock: bool = False
    #: False when the high voltage is switched by the interlock itself (no separate output).
    separate_output: bool = True

    def interlock_closed(self) -> bool:
        raise NotImplementedError(f"{self.model} has no interlock")

    def set_interlock(self, closed: bool) -> None:
        raise NotImplementedError(f"{self.model} has no interlock")


class Positioner(Device):
    """A single motion axis with absolute position read-back.

    ``move_to(..., wait=True)`` must not hold :attr:`lock` while waiting, so
    that :meth:`stop` issued from another thread interrupts the motion.
    """

    unit: str = "mm"
    move_timeout: float = 60.0

    @abstractmethod
    def move_to(self, position: float, wait: bool = True) -> None:
        """Move to an absolute position (in :attr:`unit`)."""

    @abstractmethod
    def position(self) -> float:
        """Current position (in :attr:`unit`)."""

    @abstractmethod
    def stop(self) -> None: ...

    def move_by(self, delta: float, wait: bool = True) -> None:
        self.move_to(self.position() + delta, wait=wait)

    def is_moving(self) -> bool:
        return False

    def home(self) -> None:
        raise NotImplementedError(f"{self.model} cannot home")

    def limits(self) -> tuple[float, float]:
        return (-math.inf, math.inf)

    def wait_until_stopped(self, timeout: float | None = None, poll: float = 0.05) -> None:
        deadline = time.monotonic() + (self.move_timeout if timeout is None else timeout)
        while self.is_moving():
            if time.monotonic() > deadline:
                raise DeviceError(f"{self.name}: motion did not finish in time")
            time.sleep(poll)

    def read_status(self) -> dict[str, Reading]:
        with self.lock:
            return {
                "Position": Reading(self.position(), self.unit),
                "Moving": Reading(self.is_moving()),
            }


class PiezoMirrorController(Device):
    """Open-loop stepping mirror mounts (e.g. Newport Agilis) without position read-back."""

    n_channels: int = 4
    n_axes: int = 2
    #: Speeds accepted by :meth:`jog`, as (setting, label) pairs.
    jog_speeds: tuple[tuple[int, str], ...] = ()

    @abstractmethod
    def select_channel(self, channel: int) -> None: ...

    @abstractmethod
    def set_step_amplitude(self, amplitude: int) -> None:
        """Step amplitude applied to both axes and both directions."""

    @abstractmethod
    def step(self, axis: int, steps: int) -> None:
        """Relative move by a signed number of steps."""

    @abstractmethod
    def jog(self, axis: int, speed: int) -> None:
        """Continuous motion; the sign of ``speed`` gives the direction, 0 stops."""

    @abstractmethod
    def stop(self, axis: int | None = None) -> None: ...

    @abstractmethod
    def is_moving(self, axis: int) -> bool: ...


class RFPowerSensor(Device):
    """RF power sensor, optionally reading through a directional coupler."""

    #: Coupling factor (dB) of the directional coupler in front of the sensor.
    coupling_db: float = 0.0

    @abstractmethod
    def power_dbm(self) -> float:
        """Power at the sensor, in dBm."""

    def line_power_w(self, power_dbm: float) -> float:
        """Power in the main line, corrected for the coupling factor, in W."""
        return 10 ** ((power_dbm + self.coupling_db) / 10) * 1e-3

    def read_status(self) -> dict[str, Reading]:
        dbm = self.power_dbm()
        status = {"Power": Reading(dbm, "dBm"), "Power (mW)": Reading(10 ** (dbm / 10), "mW")}
        if self.coupling_db:
            status["Line power"] = Reading(self.line_power_w(dbm), "W")
        return status


class RFAmplifier(Device):
    """RF amplifier driving the bunch-compression cavity."""

    @abstractmethod
    def rf_enabled(self) -> bool: ...

    @abstractmethod
    def set_rf_enabled(self, enabled: bool) -> None: ...


class RFAttenuator(Device):
    """Programmable RF attenuator."""

    @abstractmethod
    def attenuation(self) -> float:
        """Attenuation in dB."""

    @abstractmethod
    def set_attenuation(self, attenuation_db: float) -> None: ...
