"""Newport 1935-C optical power meter (pump laser power)."""

from __future__ import annotations

from typing import Any

from ..base import Action, DeviceError, Reading, Setting
from ..transport import SerialInstrument, Transport, parse_float

MODES = (
    "DC continuous",
    "DC single",
    "Integrate",
    "Peak-to-peak continuous",
    "Peak-to-peak single",
    "Pulse continuous",
    "Pulse single",
    "RMS",
)
FILTERS = ("None", "Analog", "Digital", "Analog + digital")


class Newport1935C(SerialInstrument):
    model = "Newport 1935-C"
    serial_defaults = {"baudrate": 38400, "timeout": 0.3}
    write_termination = "\r"
    read_termination = b"\n"

    def __init__(
        self,
        port: str | None = None,
        *,
        name: str | None = None,
        transport: Transport | None = None,
        **serial_options: Any,
    ) -> None:
        super().__init__(port, name=name, transport=transport, **serial_options)
        self._idn = ""

    def query(self, command: str) -> str:
        with self.lock:
            self.write(command)
            reply = self.read_line()
            if reply.upper() == command.upper():  # some firmware versions echo the command
                reply = self.read_line()
        if not reply:
            raise DeviceError(f"{self.name}: no reply to {command!r}")
        return reply

    def _initialize(self) -> None:
        self._idn = self.query("*IDN?")

    def identify(self) -> str:
        return self._idn or self.model

    def power(self) -> float:
        """Measured power in W."""
        return self.query_float("PM:Power?")

    def wavelength(self) -> float:
        return self.query_float("PM:Lambda?")

    def set_wavelength(self, nanometres: float) -> None:
        self.write(f"PM:Lambda {int(round(nanometres))}")

    def mode(self) -> int:
        return int(self.query_float("PM:MODE?"))

    def set_mode(self, mode: int) -> None:
        self.write(f"PM:MODE {int(mode)}")

    def filter(self) -> int:
        return int(self.query_float("PM:FILTer?"))

    def set_filter(self, filter_index: int) -> None:
        self.write(f"PM:FILTer {int(filter_index)}")

    def range(self) -> int:
        return int(self.query_float("PM:RANge?"))

    def set_range(self, range_index: int) -> None:
        self.write(f"PM:RANge {int(range_index)}")

    def set_auto_range(self, enabled: bool = True) -> None:
        self.write(f"PM:AUTO {int(enabled)}")

    def attenuator_in(self) -> bool:
        return int(parse_float(self.query("PM:ATT?"))) == 1

    def detector(self) -> str:
        with self.lock:
            return f"{self.query('PM:DETMODEL?')} S/N {self.query('PM:DETSN?')}"

    def errors(self) -> str:
        return self.query("ERRors?")

    def read_status(self) -> dict[str, Reading]:
        with self.lock:
            return {
                "Power": Reading(self.power(), "W"),
                "Wavelength": Reading(self.wavelength(), "nm"),
            }

    def settings(self) -> list[Setting]:
        with self.lock:
            return [
                Setting("wavelength", "Wavelength", self.set_wavelength, unit="nm",
                        minimum=190, maximum=11000, decimals=0, initial=self.wavelength()),
                Setting("mode", "Mode", lambda label: self.set_mode(MODES.index(label)),
                        kind="choice", choices=MODES, initial=MODES[self.mode()]),
                Setting("filter", "Filter", lambda label: self.set_filter(FILTERS.index(label)),
                        kind="choice", choices=FILTERS, initial=FILTERS[self.filter()]),
                Setting("range", "Range", self.set_range, kind="int", minimum=0, maximum=7,
                        initial=self.range()),
            ]

    def actions(self) -> list[Action]:
        return [Action("Auto range", self.set_auto_range)]
