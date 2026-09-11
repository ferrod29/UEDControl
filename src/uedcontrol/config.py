"""Instrument configuration, loaded from YAML.

Every entry under ``devices`` names a driver from
:mod:`uedcontrol.devices.registry`. The keys ``driver``, ``label``, ``group``,
``autoconnect`` and ``panel`` are interpreted by the application; all other
keys are passed to the driver's constructor (``port``, ``serial_number``...).
See ``config/example_instrument.yaml`` for a complete, commented example.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .devices.base import Device
from .devices.registry import DRIVERS, create_device


class ConfigError(ValueError):
    """The configuration is invalid."""


_DEVICE_KEYS = {"driver", "label", "group", "autoconnect", "panel"}
_TOP_LEVEL_KEYS = {"name", "data_dir", "log_dir", "devices", "delay_line", "beam", "rf_interlock"}


@dataclass
class DeviceConfig:
    name: str
    driver: str
    label: str
    group: str = "Devices"
    autoconnect: bool = False
    panel: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    def create(self) -> Device:
        return create_device(self.name, self.driver, **self.options)


@dataclass
class DelayLineConfig:
    stage: str | None = None
    t0_position: float = 150.0
    passes: float = 2.0
    direction: int = 1


@dataclass
class BeamConfig:
    rep_rate_khz: float = 1.0
    pulsed: bool = True
    counts_per_electron: float = 1.0


@dataclass
class RFInterlockConfig:
    amplifier: str
    reflected_sensor: str
    forward_sensor: str | None = None
    attenuator: str | None = None
    limit_dbm: float = 0.0


@dataclass
class InstrumentConfig:
    name: str = "UED"
    data_dir: Path = field(default_factory=lambda: Path.home() / "UEDData")
    log_dir: Path = field(default_factory=lambda: Path.home() / "UEDData" / "logs")
    devices: dict[str, DeviceConfig] = field(default_factory=dict)
    delay_line: DelayLineConfig = field(default_factory=DelayLineConfig)
    beam: BeamConfig = field(default_factory=BeamConfig)
    rf_interlock: RFInterlockConfig | None = None
    source: Path | None = None


def _path(value: Any, base: Path | None) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


def _section(data: Mapping[str, Any], key: str, cls: type) -> Any:
    raw = data.get(key) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'{key}' must be a mapping")
    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown key(s) in '{key}': {', '.join(sorted(unknown))}")
    try:
        return cls(**raw)
    except TypeError as exc:
        raise ConfigError(f"invalid '{key}' section: {exc}") from exc


def parse_config(data: Mapping[str, Any], source: Path | None = None) -> InstrumentConfig:
    if not isinstance(data, Mapping):
        raise ConfigError("the configuration must be a mapping")
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"unknown top-level key(s): {', '.join(sorted(unknown))}")
    base = source.parent if source is not None else None

    devices: dict[str, DeviceConfig] = {}
    for name, spec in (data.get("devices") or {}).items():
        if not isinstance(spec, Mapping) or "driver" not in spec:
            raise ConfigError(f"device {name!r} needs a 'driver'")
        driver = spec["driver"]
        if driver not in DRIVERS:
            raise ConfigError(f"device {name!r}: unknown driver {driver!r} (known: {', '.join(sorted(DRIVERS))})")
        devices[str(name)] = DeviceConfig(
            name=str(name),
            driver=driver,
            label=str(spec.get("label", name)),
            group=str(spec.get("group", "Devices")),
            autoconnect=bool(spec.get("autoconnect", False)),
            panel=dict(spec.get("panel") or {}),
            options={k: v for k, v in spec.items() if k not in _DEVICE_KEYS},
        )

    delay_line = _section(data, "delay_line", DelayLineConfig)
    if delay_line.stage is not None and delay_line.stage not in devices:
        raise ConfigError(f"delay_line.stage refers to unknown device {delay_line.stage!r}")

    rf_interlock = None
    if data.get("rf_interlock"):
        rf_interlock = _section(data, "rf_interlock", RFInterlockConfig)
        for role in ("amplifier", "reflected_sensor", "forward_sensor", "attenuator"):
            ref = getattr(rf_interlock, role)
            if ref is not None and ref not in devices:
                raise ConfigError(f"rf_interlock.{role} refers to unknown device {ref!r}")

    data_dir = _path(data.get("data_dir", Path.home() / "UEDData"), base)
    return InstrumentConfig(
        name=str(data.get("name", "UED")),
        data_dir=data_dir,
        log_dir=_path(data.get("log_dir", data_dir / "logs"), base),
        devices=devices,
        delay_line=delay_line,
        beam=_section(data, "beam", BeamConfig),
        rf_interlock=rf_interlock,
        source=source,
    )


def load_config(path: str | Path) -> InstrumentConfig:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    return parse_config(data, source=path.resolve())


SIMULATED_CONFIG = """
name: UED (simulated)
data_dir: ~/UEDData/simulated

devices:
  detector:
    driver: simulated_camera
    label: Detector (simulated)
    group: Cameras
    pattern: diffraction_rings
    autoconnect: true
  beam_camera:
    driver: simulated_camera
    label: Beam camera (simulated)
    group: Cameras
    pattern: electron_beam
  delay_stage:
    driver: simulated_stage
    label: Delay stage
    group: Delay line
    position: 150.0
    limits: [0.0, 300.0]
    autoconnect: true
  gun_hv:
    driver: simulated_hv_supply
    label: Electron gun HV
    group: Electron gun
  lenses:
    driver: simulated_power_supply
    label: Magnetic lenses
    group: Electron optics
    n_channels: 3
    panel:
      channel_names: [Condenser, Horizontal, Vertical]
  mirrors:
    driver: simulated_mirrors
    label: Pump/probe mirrors
    group: Laser
    panel:
      channel_names: [Probe 1, Probe 2, Pump 1, Pump 2]
  chamber_gauge:
    driver: simulated_vacuum_gauge
    label: Chamber pressure
    group: Vacuum
  rf_amplifier:
    driver: simulated_rf_amplifier
    label: RF amplifier
    group: RF
  rf_forward:
    driver: simulated_rf_sensor
    label: Forward power
    group: RF
    level_dbm: -10
  rf_reflected:
    driver: simulated_rf_sensor
    label: Reflected power
    group: RF
    level_dbm: -25
  rf_attenuator:
    driver: simulated_rf_attenuator
    label: RF attenuator
    group: RF

delay_line:
  stage: delay_stage
  t0_position: 150.0

beam:
  rep_rate_khz: 1.0
  pulsed: true

rf_interlock:
  amplifier: rf_amplifier
  forward_sensor: rf_forward
  reflected_sensor: rf_reflected
  attenuator: rf_attenuator
  limit_dbm: -15
"""


def simulated_config() -> InstrumentConfig:
    """A complete configuration made of simulated instruments (``uedcontrol --simulate``)."""
    return parse_config(yaml.safe_load(SIMULATED_CONFIG))
