"""Instrument panels. :func:`create_panel` picks the right one for a device."""

from __future__ import annotations

from ...acquisition.delay import DelayLine
from ...config import DeviceConfig
from ...devices.base import Device
from ...devices.interfaces import HighVoltageSupply, PiezoMirrorController, Positioner, PowerSupply
from .base import DevicePanel
from .generic import GenericPanel
from .hv_supply import HVSupplyPanel
from .mirrors import MirrorsPanel
from .power_supply import PowerSupplyPanel
from .rf_interlock import RFInterlockPanel
from .stage import StagePanel

__all__ = [
    "DevicePanel",
    "GenericPanel",
    "HVSupplyPanel",
    "MirrorsPanel",
    "PowerSupplyPanel",
    "RFInterlockPanel",
    "StagePanel",
    "create_panel",
]


def create_panel(
    device: Device, spec: DeviceConfig | None = None, *, delay_line: DelayLine | None = None
) -> DevicePanel:
    if isinstance(device, HighVoltageSupply):
        return HVSupplyPanel(device, spec)
    if isinstance(device, PowerSupply):
        return PowerSupplyPanel(device, spec)
    if isinstance(device, Positioner):
        line = delay_line if delay_line is not None and delay_line.stage is device else None
        return StagePanel(device, spec, delay_line=line)
    if isinstance(device, PiezoMirrorController):
        return MirrorsPanel(device, spec)
    return GenericPanel(device, spec)
