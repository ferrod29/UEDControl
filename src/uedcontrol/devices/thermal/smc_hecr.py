"""SMC Thermo-con HECR series chiller.

.. warning::
   **Unverified.** The original driver (``InstrumentControl/.../HECR02lib.py``)
   was unfinished and could not run. This implementation follows the frame
   layout that code was aiming at: reads are ``ENQ cmd CHK CR``, writes are
   ``STX cmd DATA ETX CHK CR`` with four ASCII digits of data, and the two
   checksum characters are ``0x30 + nibble`` of the low byte of the sum of the
   command and data bytes. Check it against the SMC communication manual
   before use.
"""

from __future__ import annotations

from typing import Any

from ..base import DeviceError, Reading, Setting
from ..transport import SerialInstrument, Transport

STX, ETX, ENQ, ACK, CR = 0x02, 0x03, 0x05, 0x06, 0x0D

CMD_SETPOINT = 0x31
CMD_INTERNAL_SENSOR = 0x32
CMD_EXTERNAL_SENSOR = 0x33
CMD_ALARM = 0x34
CMD_OFFSET = 0x36
CMD_SETPOINT_FRAM = 0x37
CMD_OFFSET_FRAM = 0x38


def checksum(payload: bytes) -> bytes:
    total = sum(payload) & 0xFF
    return bytes([0x30 + (total >> 4), 0x30 + (total & 0x0F)])


def encode_temperature(celsius: float) -> bytes:
    """25.00 -> b"2500" (0.00 ... 99.99 °C)."""
    value = int(round(celsius * 100))
    if not 0 <= value <= 9999:
        raise ValueError("temperature must be between 0 and 99.99 °C")
    return f"{value:04d}".encode("ascii")


def decode_temperature(data: bytes) -> float:
    return int(data[:4].decode("ascii")) / 100.0


def encode_offset(offset: float) -> bytes:
    """-1.25 -> b"-125", 0.5 -> b"0050" (range ±9.99)."""
    value = int(round(abs(offset) * 100))
    if value > 999:
        raise ValueError("offset must be within ±9.99 °C")
    return (b"-" if offset < 0 else b"0") + f"{value:03d}".encode("ascii")


def decode_offset(data: bytes) -> float:
    magnitude = int(data[1:4].decode("ascii")) / 100.0
    return -magnitude if data[:1] == b"-" else magnitude


class SMCThermoChillerHECR(SerialInstrument):
    model = "SMC Thermo-con HECR"
    serial_defaults = {"baudrate": 9600, "timeout": 0.5}

    def __init__(
        self,
        port: str | None = None,
        *,
        name: str | None = None,
        transport: Transport | None = None,
        **serial_options: Any,
    ) -> None:
        super().__init__(port, name=name, transport=transport, **serial_options)

    def _read(self, command: int) -> bytes:
        frame = bytes([ENQ, command]) + checksum(bytes([command])) + bytes([CR])
        with self.lock:
            self.transport.reset_input_buffer()
            self.transport.write(frame)
            reply = self.transport.read_until(bytes([CR]))
        if len(reply) < 7 or reply[0] != STX or reply[1] != command:
            raise DeviceError(f"{self.name}: bad reply {reply!r}")
        return reply[2:6]

    def _write(self, command: int, data: bytes) -> None:
        frame = bytes([STX, command]) + data + bytes([ETX]) + checksum(bytes([command]) + data) + bytes([CR])
        with self.lock:
            self.transport.reset_input_buffer()
            self.transport.write(frame)
            reply = self.transport.read(1)
        if reply != bytes([ACK]):
            raise DeviceError(f"{self.name}: command 0x{command:02X} not acknowledged ({reply!r})")

    def setpoint(self) -> float:
        return decode_temperature(self._read(CMD_SETPOINT))

    def set_setpoint(self, celsius: float, persistent: bool = False) -> None:
        self._write(CMD_SETPOINT_FRAM if persistent else CMD_SETPOINT, encode_temperature(celsius))

    def internal_temperature(self) -> float:
        return decode_temperature(self._read(CMD_INTERNAL_SENSOR))

    def external_temperature(self) -> float:
        return decode_temperature(self._read(CMD_EXTERNAL_SENSOR))

    def offset(self) -> float:
        return decode_offset(self._read(CMD_OFFSET))

    def set_offset(self, offset: float, persistent: bool = False) -> None:
        self._write(CMD_OFFSET_FRAM if persistent else CMD_OFFSET, encode_offset(offset))

    def read_status(self) -> dict[str, Reading]:
        with self.lock:
            return {
                "Internal sensor": Reading(self.internal_temperature(), "°C"),
                "Set point": Reading(self.setpoint(), "°C"),
            }

    def settings(self) -> list[Setting]:
        return [
            Setting("setpoint", "Set point", self.set_setpoint, unit="°C",
                    minimum=0, maximum=99.99, decimals=2),
            Setting("offset", "Offset", self.set_offset, unit="°C",
                    minimum=-9.99, maximum=9.99, decimals=2),
        ]
