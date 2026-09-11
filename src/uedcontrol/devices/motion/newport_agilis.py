"""Newport Agilis AG-UC8 piezo controller (pump/probe steering mirrors).

Up to four channels with two axes each. Commands are ASCII, CR LF terminated;
set commands produce no reply.
"""

from __future__ import annotations

import time
from typing import Any

from ..base import DeviceError
from ..interfaces import PiezoMirrorController
from ..transport import SerialInstrument, Transport, parse_float


class NewportAgilisUC8(SerialInstrument, PiezoMirrorController):
    model = "Newport Agilis AG-UC8"
    serial_defaults = {"baudrate": 921600, "timeout": 0.2}
    write_termination = "\r\n"
    read_termination = b"\n"
    n_channels = 4
    n_axes = 2
    jog_speeds = (
        (1, "5 steps/s"),
        (2, "100 steps/s"),
        (3, "1700 steps/s"),
        (4, "666 steps/s"),
    )

    def __init__(
        self,
        port: str | None = None,
        *,
        name: str | None = None,
        transport: Transport | None = None,
        command_interval_s: float = 0.02,
        **serial_options: Any,
    ) -> None:
        super().__init__(port, name=name, transport=transport, **serial_options)
        self.command_interval_s = command_interval_s

    def _send(self, command: str) -> None:
        with self.lock:
            self.write(command)
            if self.command_interval_s:
                time.sleep(self.command_interval_s)

    def _initialize(self) -> None:
        self._send("RS")  # reset
        time.sleep(0.1)
        self._send("MR")  # remote mode

    def _finalize(self) -> None:
        self._send("ML")  # back to local mode

    def identify(self) -> str:
        try:
            return self.query("VE")
        except DeviceError:
            return self.model

    def select_channel(self, channel: int) -> None:
        if not 1 <= channel <= self.n_channels:
            raise ValueError(f"channel must be 1..{self.n_channels}")
        self._send(f"CC{channel}")

    def set_step_amplitude(self, amplitude: int) -> None:
        amplitude = int(amplitude)
        if not 1 <= amplitude <= 50:
            raise ValueError("step amplitude must be 1..50")
        with self.lock:
            for axis in (1, 2):
                self._send(f"{axis}SU{amplitude}")  # positive direction
                self._send(f"{axis}SU-{amplitude}")  # negative direction

    def step(self, axis: int, steps: int) -> None:
        self._send(f"{axis}PR{int(steps)}")

    def jog(self, axis: int, speed: int) -> None:
        if not -4 <= speed <= 4:
            raise ValueError("jog speed must be -4..4")
        self._send(f"{axis}JA{int(speed)}")

    def stop(self, axis: int | None = None) -> None:
        with self.lock:
            for ax in (axis,) if axis else (1, 2):
                self._send(f"{ax}ST")

    def axis_status(self, axis: int) -> int:
        """0 ready, 1 stepping, 2 jogging, 3 moving to a limit."""
        return int(self.query(f"{axis}TS").strip()[-1])

    def is_moving(self, axis: int) -> bool:
        return self.axis_status(axis) != 0

    def last_error(self) -> int:
        return int(parse_float(self.query("TE")))
