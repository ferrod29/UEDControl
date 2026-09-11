"""Newport SMC100CC single-axis DC servo controller (delay stage).

Commands are prefixed with the controller address (``1``) and terminated with
CR LF. Queries echo the address and command, e.g. ``1TP`` -> ``1TP12.34560``.
The controller state is the last two hex digits of ``TS``.
"""

from __future__ import annotations

import time
from typing import Any

from ..base import Action, DeviceError, Reading
from ..interfaces import Positioner
from ..transport import SerialInstrument, Transport

NOT_REFERENCED = frozenset({"0A", "0B", "0C", "0D", "0E", "0F", "10", "11"})
CONFIGURATION = frozenset({"14"})
HOMING = frozenset({"1E", "1F"})
MOVING = frozenset({"28"})
READY = frozenset({"32", "33", "34", "35"})
DISABLED = frozenset({"3C", "3D", "3E"})
JOGGING = frozenset({"46", "47"})

_STATE_NAMES = (
    (NOT_REFERENCED, "Not referenced"),
    (CONFIGURATION, "Configuration"),
    (HOMING, "Homing"),
    (MOVING, "Moving"),
    (READY, "Ready"),
    (DISABLED, "Disabled"),
    (JOGGING, "Jogging"),
)


def state_name(code: str) -> str:
    for codes, label in _STATE_NAMES:
        if code in codes:
            return label
    return f"Unknown ({code})"


class NewportSMC100(SerialInstrument, Positioner):
    model = "Newport SMC100CC"
    serial_defaults = {"baudrate": 57600, "timeout": 0.2, "xonxoff": True}
    write_termination = "\r\n"
    read_termination = b"\n"

    def __init__(
        self,
        port: str | None = None,
        *,
        name: str | None = None,
        transport: Transport | None = None,
        address: int = 1,
        velocity: float = 10.0,
        encoder_increment: float | None = None,
        move_timeout: float = 60.0,
        **serial_options: Any,
    ) -> None:
        super().__init__(port, name=name, transport=transport, **serial_options)
        self.address = address
        self.velocity = velocity
        self.encoder_increment = encoder_increment
        self.move_timeout = move_timeout

    def _send(self, command: str) -> None:
        self.write(f"{self.address}{command}")

    def _ask(self, command: str) -> str:
        prefix = f"{self.address}{command[:2]}"
        reply = self.query(f"{self.address}{command}")
        if not reply.startswith(prefix):
            raise DeviceError(f"{self.name}: unexpected reply {reply!r} to {command!r}")
        return reply[len(prefix):]

    # ----------------------------------------------------------------- state
    def state(self) -> str:
        return self._ask("TS")[-2:].upper()

    def last_error(self) -> str:
        return self._ask("TE")

    def _wait_for(self, states: frozenset[str], timeout: float | None = None) -> str:
        deadline = time.monotonic() + (self.move_timeout if timeout is None else timeout)
        while True:
            try:
                code = self.state()
            except DeviceError:  # the controller does not answer while it reboots
                code = ""
            if code in states:
                return code
            if time.monotonic() > deadline:
                raise DeviceError(f"{self.name}: timed out waiting, state {state_name(code)}")
            time.sleep(0.05)

    # ------------------------------------------------------- Positioner interface
    def position(self) -> float:
        return float(self._ask("TP"))

    def move_to(self, position: float, wait: bool = True) -> None:
        self._send(f"PA{position:.5f}")
        if wait:
            self.wait_until_stopped()

    def move_by(self, delta: float, wait: bool = True) -> None:
        self._send(f"PR{delta:.5f}")
        if wait:
            self.wait_until_stopped()

    def stop(self) -> None:
        self._send("ST")

    def is_moving(self) -> bool:
        return self.state() in MOVING | HOMING | JOGGING

    def home(self) -> None:
        self._send("OR")
        self._wait_for(READY)

    def limits(self) -> tuple[float, float]:
        with self.lock:
            return float(self._ask("SL?")), float(self._ask("SR?"))

    def initialize_stage(self) -> None:
        """Reset, configure the velocity and home the stage. The stage moves!"""
        with self.lock:
            if self.state() in DISABLED:
                self._send("MM1")
                self._wait_for(READY)
            self._send("RS")
            time.sleep(0.5)
            self._wait_for(NOT_REFERENCED, timeout=10)
            self._send("PW1")
            self._wait_for(CONFIGURATION, timeout=10)
            if self.encoder_increment is not None:
                self._send(f"SU{self.encoder_increment:g}")
            self._send(f"VA{self.velocity:g}")
            self._send("PW0")  # stores the configuration, takes a few seconds
            self._wait_for(NOT_REFERENCED, timeout=20)
            self._send("OR")
            self._wait_for(READY)

    def read_status(self) -> dict[str, Reading]:
        with self.lock:
            return {
                "Position": Reading(self.position(), self.unit),
                "State": Reading(state_name(self.state())),
            }

    def actions(self) -> list[Action]:
        return [
            Action("Home", self.home),
            Action(
                "Reset && initialize",
                self.initialize_stage,
                confirm="Reset the controller and home the stage? The stage will move.",
            ),
        ]
