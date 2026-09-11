"""Heinzinger PNChp 100-1 high-voltage supply through the MPSD-built control box.

Binary protocol at 38400 8N1. Every frame, in both directions, is 6 bytes::

    [command, ~command & 0xFF, data_lo, data_hi, 0x00, 0x00]

The high voltage is enabled by closing the interlock of the control box, so
this supply has no separate output switch.

The conversion factors are copied verbatim from the original driver
(``UEDControlSystem/Devices/pnchp100.py``). They come from the control box
design; re-check them against its documentation if readings look wrong.
"""

from __future__ import annotations

from typing import Any

from ..base import DeviceError
from ..interfaces import HighVoltageSupply
from ..transport import SerialInstrument, Transport

READ_VOLTAGE = 0x10
READ_CURRENT = 0x11
SET_CURRENT = 0x22
SET_VOLTAGE = 0x23
READ_INTERLOCK = 0x30
SET_INTERLOCK = 0x41

INTERLOCK_OPEN = 0xFF
INTERLOCK_CLOSED = 0xFE

VOLTAGE_READ_FULL_SCALE_KV = 1000 / 10.012
CURRENT_READ_FULL_SCALE_UA = 10000 / 10.012
VOLTAGE_SET_FACTOR = 10 / 1024  # raw = factor * kV * 65535
CURRENT_SET_FACTOR = 1 / 1024  # raw = factor * µA * 65535


def frame(command: int, data: int = 0) -> bytes:
    """Build a 6-byte control-box frame."""
    return bytes([command, ~command & 0xFF]) + (data & 0xFFFF).to_bytes(2, "little") + b"\x00\x00"


def _clamp16(value: float) -> int:
    return max(0, min(0xFFFF, int(value)))


class HeinzingerPNChp100(SerialInstrument, HighVoltageSupply):
    model = "Heinzinger PNChp 100-1 (MPSD control box)"
    serial_defaults = {"baudrate": 38400, "timeout": 0.2}
    has_interlock = True
    separate_output = False

    def __init__(
        self,
        port: str | None = None,
        *,
        name: str | None = None,
        transport: Transport | None = None,
        **serial_options: Any,
    ) -> None:
        super().__init__(port, name=name, transport=transport, **serial_options)

    def _exchange(self, command: int, data: int = 0, check: bool = True) -> bytes:
        with self.lock:
            self.transport.reset_input_buffer()
            self.transport.write(frame(command, data))
            reply = self.transport.read(6)
        if check and (len(reply) != 6 or reply[0] != command):
            raise DeviceError(f"{self.name}: bad reply {reply.hex(' ')} to command 0x{command:02X}")
        return reply

    def _raw(self, command: int) -> int:
        return int.from_bytes(self._exchange(command)[2:4], "little")

    def measure_voltage(self, channel: int = 1) -> float:
        return self._raw(READ_VOLTAGE) * VOLTAGE_READ_FULL_SCALE_KV / 65535

    def measure_current(self, channel: int = 1) -> float:
        return self._raw(READ_CURRENT) * CURRENT_READ_FULL_SCALE_UA / 65535

    def set_voltage(self, value: float, channel: int = 1) -> None:
        self._exchange(SET_VOLTAGE, _clamp16(VOLTAGE_SET_FACTOR * value * 65535), check=False)

    def set_current(self, value: float, channel: int = 1) -> None:
        self._exchange(SET_CURRENT, _clamp16(CURRENT_SET_FACTOR * value * 65535), check=False)

    def interlock_closed(self) -> bool:
        state = self._exchange(READ_INTERLOCK)[2]
        if state == INTERLOCK_CLOSED:
            return True
        if state == INTERLOCK_OPEN:
            return False
        raise DeviceError(f"{self.name}: unknown interlock state 0x{state:02X}")

    def set_interlock(self, closed: bool) -> None:
        self._exchange(SET_INTERLOCK, 0x10 if closed else 0x00, check=False)

    # The interlock is the output switch of this supply.
    def output_enabled(self, channel: int = 1) -> bool:
        return self.interlock_closed()

    def set_output(self, enabled: bool, channel: int = 1) -> None:
        self.set_interlock(enabled)
