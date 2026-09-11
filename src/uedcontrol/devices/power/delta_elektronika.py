"""Delta Elektronika SM 70-AR-24 DC supply (ASCII commands, CR terminated)."""

from __future__ import annotations

from typing import Any

from ..interfaces import PowerSupply
from ..transport import SerialInstrument, Transport, parse_float


class DeltaSM70AR24(SerialInstrument, PowerSupply):
    model = "Delta Elektronika SM 70-AR-24"
    serial_defaults = {"baudrate": 9600, "timeout": 0.5}
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

    def read_line(self) -> str:
        # Replies may end with an EOT character.
        return super().read_line().strip("\x04 \r\n")

    def _initialize(self) -> None:
        self.write("SO:FU:RSD 0")  # ignore the remote shut-down input

    def set_voltage(self, value: float, channel: int = 1) -> None:
        self.write(f"SO:VO {value:.4f}")

    def set_current(self, value: float, channel: int = 1) -> None:
        self.write(f"SO:CU {value:.4f}")

    def measure_voltage(self, channel: int = 1) -> float:
        return self.query_float("ME:VO?")

    def measure_current(self, channel: int = 1) -> float:
        return self.query_float("ME:CU?")

    def voltage_setpoint(self, channel: int = 1) -> float:
        return self.query_float("SO:VO?")

    def current_setpoint(self, channel: int = 1) -> float:
        return self.query_float("SO:CU?")

    def set_output(self, enabled: bool, channel: int = 1) -> None:
        self.write(f"SO:FU:OUTP {int(enabled)}")

    def output_enabled(self, channel: int = 1) -> bool:
        return int(parse_float(self.query("SO:FU:OUTP?"))) == 1
