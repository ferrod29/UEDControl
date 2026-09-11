"""The instrument as a whole: every configured device plus the pump-probe delay line.

This is the GUI-independent entry point for scripts and notebooks::

    from uedcontrol.config import load_config
    from uedcontrol.instrument import Instrument

    with Instrument.from_config(load_config("config/local.yaml")) as ued:
        ued.connect("delay_stage", "detector")
        ued.delay_line.move_to_delay(2.5)
        camera = ued.cameras()["detector"]
        camera.start_acquisition()
        image = camera.grab()

The GUI builds its panels from the same object.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from .acquisition.delay import DelayLine
from .config import ConfigError, InstrumentConfig
from .devices.base import Device
from .devices.interfaces import Camera, Positioner, RFAmplifier, RFAttenuator, RFPowerSensor

log = logging.getLogger(__name__)

D = TypeVar("D", bound=Device)


class Instrument:
    def __init__(self, config: InstrumentConfig, devices: dict[str, Device], delay_line: DelayLine | None) -> None:
        self.config = config
        self.devices = devices
        self.delay_line = delay_line

    @classmethod
    def from_config(cls, config: InstrumentConfig) -> Instrument:
        """Instantiate every device (nothing is connected yet) and check the device roles."""
        devices: dict[str, Device] = {}
        for name, spec in config.devices.items():
            try:
                devices[name] = spec.create()
            except Exception as exc:
                raise ConfigError(f"device {name!r} ({spec.driver}): {exc}") from exc

        delay_line = None
        line = config.delay_line
        if line.stage is not None:
            stage = _expect(devices, line.stage, Positioner, "delay_line.stage")
            try:
                delay_line = DelayLine(stage, line.t0_position, line.passes, line.direction)
            except ValueError as exc:
                raise ConfigError(f"delay_line: {exc}") from exc

        rf = config.rf_interlock
        if rf is not None:
            _expect(devices, rf.amplifier, RFAmplifier, "rf_interlock.amplifier")
            _expect(devices, rf.reflected_sensor, RFPowerSensor, "rf_interlock.reflected_sensor")
            if rf.forward_sensor is not None:
                _expect(devices, rf.forward_sensor, RFPowerSensor, "rf_interlock.forward_sensor")
            if rf.attenuator is not None:
                _expect(devices, rf.attenuator, RFAttenuator, "rf_interlock.attenuator")

        instrument = cls(config, devices, delay_line)
        instrument._link_simulated_devices()
        return instrument

    # ------------------------------------------------------------------ access
    def of_type(self, kind: type[D]) -> dict[str, D]:
        return {name: device for name, device in self.devices.items() if isinstance(device, kind)}

    def cameras(self) -> dict[str, Camera]:
        return self.of_type(Camera)

    def __getitem__(self, name: str) -> Device:
        return self.devices[name]

    # -------------------------------------------------------------- lifecycle
    def connect(self, *names: str) -> None:
        """Connect the named devices (all of them if no name is given)."""
        for name in names or tuple(self.devices):
            self.devices[name].connect()

    def disconnect_all(self) -> None:
        """Disconnect every connected device; failures are logged, never raised."""
        for name, device in self.devices.items():
            if device.connected:
                try:
                    device.disconnect()
                except Exception:
                    log.exception("disconnecting %s failed", name)

    def __enter__(self) -> Instrument:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.disconnect_all()

    # ------------------------------------------------------------- simulation
    def _link_simulated_devices(self) -> None:
        """Make the simulated devices react to each other like the real set-up.

        The simulated detector sees the pump-probe transient when it is moved
        by the simulated delay line, and the simulated RF sensors only read
        power while the simulated amplifier is on.
        """
        from .devices.simulated import SimulatedCamera, SimulatedRFPowerSensor, SimulatedStage

        if self.delay_line is not None and isinstance(self.delay_line.stage, SimulatedStage):
            for camera in self.of_type(SimulatedCamera).values():
                camera.delay_source = self.delay_line.current_delay
        rf = self.config.rf_interlock
        if rf is not None:
            amplifier = self.devices[rf.amplifier]
            for name in (rf.forward_sensor, rf.reflected_sensor):
                sensor = self.devices.get(name) if name else None
                if isinstance(sensor, SimulatedRFPowerSensor):
                    sensor.source = amplifier.rf_enabled


def _expect(devices: dict[str, Device], name: str, kind: type[D], role: str) -> D:
    device = devices[name]
    if not isinstance(device, kind):
        raise ConfigError(f"{role}: device {name!r} is a {device.model}, not a {kind.__name__}")
    return device
