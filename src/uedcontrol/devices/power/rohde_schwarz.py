"""Rohde & Schwarz programmable DC supplies (HMP4040, NGE100 series) over SCPI.

Both are used for the magnetic lenses. The two families differ only in the
channel count and in the command that switches a single channel's output.
Connecting does not reset the instrument (the original HMP4040 driver sent
``*RST``, which switched the lens currents off).
"""

from __future__ import annotations

from typing import Any

from ..base import DeviceError
from ..interfaces import PowerSupply
from ..transport import SerialInstrument, Transport


def parse_bool(text: str) -> bool:
    value = text.strip().upper()
    if value in {"1", "ON"}:
        return True
    if value in {"0", "OFF"}:
        return False
    raise DeviceError(f"unexpected boolean reply {text!r}")


class RohdeSchwarzSupply(SerialInstrument, PowerSupply):
    """Common SCPI implementation; use one of the concrete subclasses."""

    has_master_output = True
    write_termination = "\r\n"
    read_termination = b"\n"
    #: SCPI command that switches the selected channel on or off.
    channel_output_command = "OUTP"

    def __init__(
        self,
        port: str | None = None,
        *,
        name: str | None = None,
        transport: Transport | None = None,
        n_channels: int | None = None,
        **serial_options: Any,
    ) -> None:
        super().__init__(port, name=name, transport=transport, **serial_options)
        if n_channels is not None:
            self.n_channels = n_channels
        self._idn = ""

    def _initialize(self) -> None:
        self.write("*CLS")
        self.write("SYST:MIX")  # keep the front panel usable while under remote control
        self._idn = self.query("*IDN?")

    def identify(self) -> str:
        return self._idn or self.model

    def _select(self, channel: int) -> None:
        if not 1 <= channel <= self.n_channels:
            raise ValueError(f"channel must be 1..{self.n_channels}, got {channel}")
        self.write(f"INST:NSEL {channel}")

    def set_voltage(self, value: float, channel: int = 1) -> None:
        with self.lock:
            self._select(channel)
            self.write(f"VOLT {value:.4f}")

    def set_current(self, value: float, channel: int = 1) -> None:
        with self.lock:
            self._select(channel)
            self.write(f"CURR {value:.4f}")

    def measure_voltage(self, channel: int = 1) -> float:
        with self.lock:
            self._select(channel)
            return self.query_float("MEAS:VOLT?")

    def measure_current(self, channel: int = 1) -> float:
        with self.lock:
            self._select(channel)
            return self.query_float("MEAS:CURR?")

    def voltage_setpoint(self, channel: int = 1) -> float:
        with self.lock:
            self._select(channel)
            return self.query_float("VOLT?")

    def current_setpoint(self, channel: int = 1) -> float:
        with self.lock:
            self._select(channel)
            return self.query_float("CURR?")

    def set_output(self, enabled: bool, channel: int = 1) -> None:
        with self.lock:
            self._select(channel)
            self.write(f"{self.channel_output_command} {int(enabled)}")

    def output_enabled(self, channel: int = 1) -> bool:
        with self.lock:
            self._select(channel)
            return parse_bool(self.query(f"{self.channel_output_command}?"))

    def set_master_output(self, enabled: bool) -> None:
        self.write(f"OUTP:GEN {int(enabled)}")

    def master_output_enabled(self) -> bool:
        return parse_bool(self.query("OUTP:GEN?"))


class RohdeSchwarzHMP4040(RohdeSchwarzSupply):
    model = "Rohde & Schwarz HMP4040"
    n_channels = 4
    channel_output_command = "OUTP:SEL"
    serial_defaults = {"baudrate": 921600, "timeout": 0.3, "rtscts": True}


class RohdeSchwarzNGE100(RohdeSchwarzSupply):
    model = "Rohde & Schwarz NGE100"
    n_channels = 3
    channel_output_command = "OUTP"
    serial_defaults = {"baudrate": 9600, "timeout": 0.3, "rtscts": True}
