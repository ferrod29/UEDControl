"""Agilent TwisTorr 304/305 FS turbo pump controller (Agilent "window" protocol).

Frame: ``STX ADDR WIN COM DATA ETX CRC`` where ``WIN`` is the window number
as three ASCII digits, ``COM`` is ``0`` (read) or ``1`` (write) and ``CRC``
is the XOR of every byte after STX up to and including ETX, sent as two
upper-case hex digits. (The original driver produced the CRC with ``hex()``,
which broke for values below 0x10 and used lower-case digits.)
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import Any

from ..base import Action, DeviceError, Reading
from ..transport import SerialInstrument, Transport

STX, ETX, ACK, NACK = 0x02, 0x03, 0x06, 0x15
REPLY_ERRORS = {
    NACK: "command failed",
    0x32: "unknown window",
    0x33: "data type error",
    0x34: "value out of range",
    0x35: "window disabled",
}

WIN_START_STOP = 0
WIN_SERIAL_MODE = 8
WIN_SOFT_START = 100
WIN_POWER = 202
WIN_FREQUENCY = 203
WIN_TEMPERATURE = 204
WIN_STATUS = 205
WIN_ERROR = 206
WIN_PRESSURE = 224

PUMP_STATUS = {
    0: "Stop",
    1: "Waiting interlock",
    2: "Starting",
    3: "Auto-tuning",
    4: "Braking",
    5: "Normal",
    6: "Fail",
}


def checksum(body: bytes) -> bytes:
    return f"{reduce(operator.xor, body, 0):02X}".encode("ascii")


def build_frame(address: int, window: int, write: bool, data: str = "") -> bytes:
    body = (
        bytes([address])
        + f"{window:03d}".encode("ascii")
        + (b"1" if write else b"0")
        + data.encode("ascii")
        + bytes([ETX])
    )
    return bytes([STX]) + body + checksum(body)


def parse_frame(frame: bytes) -> bytes:
    """Validate a reply and return what lies between the address and ETX."""
    if len(frame) < 5 or frame[0] != STX or frame[-3] != ETX:
        raise DeviceError(f"malformed reply {frame!r}")
    body = frame[1:-2]
    if checksum(body) != frame[-2:].upper():
        raise DeviceError(f"checksum mismatch in reply {frame!r}")
    return body[1:-1]


class AgilentTwisTorr(SerialInstrument):
    model = "Agilent TwisTorr 304/305 FS"
    serial_defaults = {"baudrate": 9600, "timeout": 0.5}

    def __init__(
        self,
        port: str | None = None,
        *,
        name: str | None = None,
        transport: Transport | None = None,
        address: int = 0,
        **serial_options: Any,
    ) -> None:
        super().__init__(port, name=name, transport=transport, **serial_options)
        self.address = 0x80 + address

    def _transact(self, window: int, write: bool = False, data: str = "") -> str:
        with self.lock:
            self.transport.reset_input_buffer()
            self.transport.write(build_frame(self.address, window, write, data))
            reply = self.transport.read_until(bytes([ETX]))
            reply += self.transport.read(2)
        payload = parse_frame(reply)
        if len(payload) == 1:
            code = payload[0]
            if code == ACK:
                return ""
            raise DeviceError(f"{self.name}: window {window}: {REPLY_ERRORS.get(code, hex(code))}")
        if payload[:3] != f"{window:03d}".encode("ascii"):
            raise DeviceError(f"{self.name}: reply for the wrong window: {reply!r}")
        return payload[4:].decode("ascii").strip()

    def read_window(self, window: int) -> str:
        return self._transact(window)

    def write_window(self, window: int, data: str) -> None:
        self._transact(window, write=True, data=data)

    def start(self) -> None:
        self.write_window(WIN_START_STOP, "1")

    def stop(self) -> None:
        self.write_window(WIN_START_STOP, "0")

    def set_serial_mode(self) -> None:
        """Take control over the serial line (instead of the front panel/remote input)."""
        self.write_window(WIN_SERIAL_MODE, "0")

    def set_soft_start(self, enabled: bool) -> None:
        self.write_window(WIN_SOFT_START, "1" if enabled else "0")

    def status(self) -> str:
        code = int(self.read_window(WIN_STATUS))
        return PUMP_STATUS.get(code, f"Unknown ({code})")

    def error_code(self) -> int:
        return int(self.read_window(WIN_ERROR))

    def temperature(self) -> float:
        return float(self.read_window(WIN_TEMPERATURE))

    def pressure(self) -> float:
        return float(self.read_window(WIN_PRESSURE))

    def power(self) -> float:
        return float(self.read_window(WIN_POWER))

    def frequency(self) -> float:
        return float(self.read_window(WIN_FREQUENCY))

    def read_status(self) -> dict[str, Reading]:
        with self.lock:
            return {
                "Status": Reading(self.status()),
                "Pressure": Reading(self.pressure(), "mbar"),
                "Rotation": Reading(self.frequency(), "Hz"),
                "Power": Reading(self.power(), "W"),
                "Temperature": Reading(self.temperature(), "°C"),
            }

    def actions(self) -> list[Action]:
        return [
            Action("Serial control", self.set_serial_mode),
            Action("Start pump", self.start, confirm="Start the turbo pump?"),
            Action("Stop pump", self.stop, confirm="Stop the turbo pump?"),
            Action("Soft start on", lambda: self.set_soft_start(True)),
            Action("Soft start off", lambda: self.set_soft_start(False)),
        ]
