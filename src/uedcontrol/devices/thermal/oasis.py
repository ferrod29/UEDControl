"""Solid State Cooling Oasis Three chiller (magnetic lens cooling)."""

from __future__ import annotations

from typing import Any

from ..base import Action, Reading, Setting
from ..transport import SerialInstrument, Transport


class OasisThreeChiller(SerialInstrument):
    model = "Solid State Cooling Oasis Three"
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
        self._idn = ""

    def _command(self, command: str) -> str:
        """Send a command and return whatever the chiller answers (possibly nothing)."""
        with self.lock:
            self.write(command)
            return self.read_line()

    def _initialize(self) -> None:
        self._idn = self._command("IDN")

    def identify(self) -> str:
        return self._idn or self.model

    def is_running(self) -> bool:
        return "RUNNING" in self.query("RUN?").upper()

    def run(self) -> None:
        self._command("RUN")

    def stop(self) -> None:
        self._command("STOP")

    def temperature(self) -> float:
        return self.query_float("TEMP?")

    def setpoint(self) -> float:
        return self.query_float("SETTEMP?")

    def set_setpoint(self, celsius: float) -> None:
        self._command(f"SETTEMP {celsius:.1f}")

    def pump_temperature(self) -> float:
        return self.query_float("PUMPTEMP?")

    def read_status(self) -> dict[str, Reading]:
        with self.lock:
            return {
                "Running": Reading(self.is_running()),
                "Temperature": Reading(self.temperature(), "°C"),
                "Set point": Reading(self.setpoint(), "°C"),
            }

    def settings(self) -> list[Setting]:
        return [
            Setting("setpoint", "Set point", self.set_setpoint, unit="°C",
                    minimum=5, maximum=40, decimals=1, initial=self.setpoint()),
        ]

    def actions(self) -> list[Action]:
        return [Action("Run", self.run), Action("Stop", self.stop, confirm="Stop the chiller?")]
