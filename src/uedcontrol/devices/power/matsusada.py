"""Matsusada Precision high-voltage supply through the CO-HVU32 serial controller.

ASCII protocol, commands terminated with CR, RS-485 at 9600 8N1. ``#1`` is the
controller address.

=============  =========================================================
``#1 REN``     remote enable              ``#1 GTL``  back to local
``#1 RST``     reset status/alarms        ``#1 STS``  status word
``#1 VCN x``   voltage set point in kV    ``#1 VM``   measured voltage, "VM=12.34"
``#1 ICN x``   current limit in % of the full-scale current
``#1 IM``      measured current, "IM=1.2" (% of full scale)
``#1 SW1/0``   output on/off              ``#1 SW?``  "SW1" or "SW0"
=============  =========================================================

Replies can arrive late on this bus, so every query checks that the reply
carries the expected prefix and retries otherwise (the original code looped
forever instead).
"""

from __future__ import annotations

from typing import Any

from ..base import DeviceError
from ..interfaces import HighVoltageSupply
from ..transport import SerialInstrument, Transport, parse_float


def _fmt(value: float) -> str:
    return format(round(float(value), 3), "g")


class MatsusadaCOHVU32(SerialInstrument, HighVoltageSupply):
    model = "Matsusada CO-HVU32"
    # write_timeout=0 and RS-485 mode are what made the original set-up work.
    serial_defaults = {"baudrate": 9600, "timeout": 0.2, "write_timeout": 0, "rs485": True}
    write_termination = "\r"
    read_termination = b"\r"

    def __init__(
        self,
        port: str | None = None,
        *,
        name: str | None = None,
        transport: Transport | None = None,
        address: int = 1,
        full_scale_current_ua: float = 1500.0,
        retries: int = 5,
        **serial_options: Any,
    ) -> None:
        super().__init__(port, name=name, transport=transport, **serial_options)
        self.address = address
        self.full_scale_current_ua = full_scale_current_ua
        self.retries = retries

    def _cmd(self, command: str) -> str:
        return f"#{self.address} {command}"

    def _send(self, command: str) -> None:
        self.write(self._cmd(command))

    def _ask(self, command: str, prefix: str) -> str:
        with self.lock:
            for _ in range(self.retries):
                self._send(command)
                for _ in range(2):  # a stale reply may precede the one we want
                    reply = self.read_line()
                    if reply.startswith(prefix):
                        return reply
                    if not reply:
                        break
        raise DeviceError(f"{self.name}: no valid reply to {command!r}")

    # ----------------------------------------------------------------- lifecycle
    def _initialize(self) -> None:
        self._send("REN")
        self._send("RST")

    def _finalize(self) -> None:
        self._send("RST")
        self._send("GTL")

    # ------------------------------------------------------- PowerSupply interface
    def set_voltage(self, value: float, channel: int = 1) -> None:
        self._send(f"VCN {_fmt(value)}")

    def set_current(self, value: float, channel: int = 1) -> None:
        percent = 100.0 * value / self.full_scale_current_ua
        self._send(f"ICN {percent:.1f}")

    def measure_voltage(self, channel: int = 1) -> float:
        return parse_float(self._ask("VM", prefix="VM"))

    def measure_current(self, channel: int = 1) -> float:
        percent = parse_float(self._ask("IM", prefix="IM"))
        return percent * self.full_scale_current_ua / 100.0

    def output_enabled(self, channel: int = 1) -> bool:
        return self._ask("SW?", prefix="SW").strip().endswith("1")

    def set_output(self, enabled: bool, channel: int = 1) -> None:
        with self.lock:
            if enabled:
                self._send("RST")  # clear latched alarms first, as the original GUI did
            self._send("SW1" if enabled else "SW0")

    def status_word(self) -> str:
        with self.lock:
            self._send("STS")
            return self.read_line()
