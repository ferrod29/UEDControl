"""Thorlabs KDC101 K-Cube DC servo controller (APT binary protocol over USB-VCP).

Every message starts with a 6-byte header ``<msg_id:u16> <p1:u8> <p2:u8>
<dest:u8> <src:u8>``. When bit 0x80 of ``dest`` is set, ``p1 | p2 << 8`` is
the length of a data packet that follows the header.

Differences from the original driver, all per the APT protocol reference:
messages carrying data now set the 0x80 flag, "disable channel" uses 0x02
(not 0x00), the brushed-DC status request 0x0490 is used (its reply layout
is the one the old code tried to parse), and waiting for a move polls the
status bits so :meth:`stop` can interrupt it.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Any

from ..base import Action, DeviceError, Reading
from ..interfaces import Positioner
from ..transport import SerialInstrument, Transport

HOST = 0x01
DEVICE = 0x50  # generic USB hardware unit
HAS_DATA = 0x80

MGMSG_HW_REQ_INFO = 0x0005
MGMSG_HW_GET_INFO = 0x0006
MGMSG_MOD_SET_CHANENABLESTATE = 0x0210
MGMSG_MOD_REQ_CHANENABLESTATE = 0x0211
MGMSG_MOD_GET_CHANENABLESTATE = 0x0212
MGMSG_MOT_MOVE_HOME = 0x0443
MGMSG_MOT_MOVE_HOMED = 0x0444
MGMSG_MOT_MOVE_RELATIVE = 0x0448
MGMSG_MOT_MOVE_ABSOLUTE = 0x0453
MGMSG_MOT_MOVE_COMPLETED = 0x0464
MGMSG_MOT_MOVE_STOP = 0x0465
MGMSG_MOT_REQ_DCSTATUSUPDATE = 0x0490
MGMSG_MOT_GET_DCSTATUSUPDATE = 0x0491

STATUS_BITS = {
    0x00000001: "CW hardware limit",
    0x00000002: "CCW hardware limit",
    0x00000004: "CW software limit",
    0x00000008: "CCW software limit",
    0x00000010: "Moving CW",
    0x00000020: "Moving CCW",
    0x00000040: "Jogging CW",
    0x00000080: "Jogging CCW",
    0x00000100: "Motor connected",
    0x00000200: "Homing",
    0x00000400: "Homed",
    0x00004000: "Position error",
    0x80000000: "Channel enabled",
}
MOTION_BITS = 0x10 | 0x20 | 0x40 | 0x80 | 0x200

#: Encoder counts per degree of the PRM1-Z8 rotation stage used on the instrument.
PRM1_Z8_COUNTS_PER_DEGREE = 1919.6418578623391


def short_message(msg_id: int, param1: int = 0, param2: int = 0) -> bytes:
    return struct.pack("<HBBBB", msg_id, param1, param2, DEVICE, HOST)


def long_message(msg_id: int, data: bytes) -> bytes:
    return struct.pack("<HHBB", msg_id, len(data), DEVICE | HAS_DATA, HOST) + data


@dataclass(frozen=True)
class AptMessage:
    msg_id: int
    param1: int
    param2: int
    data: bytes


class ThorlabsKDC101(SerialInstrument, Positioner):
    model = "Thorlabs KDC101"
    serial_defaults = {"baudrate": 115200, "timeout": 0.5, "rtscts": True}

    def __init__(
        self,
        port: str | None = None,
        *,
        name: str | None = None,
        transport: Transport | None = None,
        channel: int = 1,
        counts_per_unit: float = PRM1_Z8_COUNTS_PER_DEGREE,
        unit: str = "deg",
        tolerance_counts: int = 20,
        move_timeout: float = 60.0,
        **serial_options: Any,
    ) -> None:
        super().__init__(port, name=name, transport=transport, **serial_options)
        self.channel = channel
        self.counts_per_unit = counts_per_unit
        self.unit = unit
        self.tolerance_counts = tolerance_counts
        self.move_timeout = move_timeout

    # ------------------------------------------------------------- messaging
    def _read_message(self) -> AptMessage:
        header = self.transport.read(6)
        if len(header) < 6:
            raise DeviceError(f"{self.name}: no reply from controller")
        msg_id, p1, p2, dest, _source = struct.unpack("<HBBBB", header)
        data = b""
        if dest & HAS_DATA:
            length = p1 | (p2 << 8)
            data = self.transport.read(length)
            if len(data) < length:
                raise DeviceError(f"{self.name}: truncated message 0x{msg_id:04X}")
        return AptMessage(msg_id, p1, p2, data)

    def _await(self, msg_id: int, timeout: float) -> AptMessage:
        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self._read_message()
            except DeviceError:
                if time.monotonic() > deadline:
                    raise
                continue
            if message.msg_id == msg_id:
                return message
            self.log.debug("ignoring APT message 0x%04X", message.msg_id)
            if time.monotonic() > deadline:
                raise DeviceError(f"{self.name}: no message 0x{msg_id:04X} within {timeout} s")

    def _send(self, packet: bytes) -> None:
        with self.lock:
            self.transport.reset_input_buffer()
            self.transport.write(packet)

    def _request(self, request_id: int, reply_id: int, param1: int | None = None) -> AptMessage:
        with self.lock:
            self._send(short_message(request_id, self.channel if param1 is None else param1))
            return self._await(reply_id, timeout=2.0)

    # ------------------------------------------------------------- queries
    def identify(self) -> str:
        info = self._request(MGMSG_HW_REQ_INFO, MGMSG_HW_GET_INFO, param1=0).data
        serial = struct.unpack_from("<I", info, 0)[0]
        model = info[4:12].split(b"\0")[0].decode("ascii", errors="replace")
        return f"{model} S/N {serial}"

    def _dc_status(self) -> tuple[int, int, int]:
        data = self._request(MGMSG_MOT_REQ_DCSTATUSUPDATE, MGMSG_MOT_GET_DCSTATUSUPDATE).data
        _chan, counts, velocity, _reserved, bits = struct.unpack("<HlHHI", data[:14])
        return counts, velocity, bits

    def status_flags(self) -> list[str]:
        bits = self._dc_status()[2]
        return [label for mask, label in STATUS_BITS.items() if bits & mask]

    def is_enabled(self) -> bool:
        return self._request(MGMSG_MOD_REQ_CHANENABLESTATE, MGMSG_MOD_GET_CHANENABLESTATE).param2 == 0x01

    def set_enabled(self, enabled: bool) -> None:
        self._send(short_message(MGMSG_MOD_SET_CHANENABLESTATE, self.channel, 0x01 if enabled else 0x02))

    # ------------------------------------------------------ Positioner interface
    def position(self) -> float:
        return self._dc_status()[0] / self.counts_per_unit

    def is_moving(self) -> bool:
        return bool(self._dc_status()[2] & MOTION_BITS)

    def _wait_for_target(self, target_counts: int | None) -> None:
        deadline = time.monotonic() + self.move_timeout
        time.sleep(0.1)  # let the controller report the motion
        while True:
            counts, _velocity, bits = self._dc_status()
            settled = not bits & MOTION_BITS
            on_target = target_counts is None or abs(counts - target_counts) <= self.tolerance_counts
            if settled and on_target:
                return
            if time.monotonic() > deadline:
                raise DeviceError(f"{self.name}: move did not complete")
            time.sleep(0.05)

    def move_to(self, position: float, wait: bool = True) -> None:
        counts = int(round(position * self.counts_per_unit))
        self._send(long_message(MGMSG_MOT_MOVE_ABSOLUTE, struct.pack("<Hl", self.channel, counts)))
        if wait:
            self._wait_for_target(counts)

    def move_by(self, delta: float, wait: bool = True) -> None:
        with self.lock:
            start = self._dc_status()[0]
            counts = int(round(delta * self.counts_per_unit))
            self._send(long_message(MGMSG_MOT_MOVE_RELATIVE, struct.pack("<Hl", self.channel, counts)))
        if wait:
            self._wait_for_target(start + counts)

    def stop(self) -> None:
        self._send(short_message(MGMSG_MOT_MOVE_STOP, self.channel, 0x02))  # profiled stop

    def home(self) -> None:
        self._send(short_message(MGMSG_MOT_MOVE_HOME, self.channel))
        self._wait_for_target(None)

    def read_status(self) -> dict[str, Reading]:
        counts, _velocity, bits = self._dc_status()
        flags = [label for mask, label in STATUS_BITS.items() if bits & mask]
        return {
            "Position": Reading(counts / self.counts_per_unit, self.unit),
            "Status": Reading(", ".join(flags) or "-"),
        }

    def actions(self) -> list[Action]:
        return [
            Action("Home", self.home),
            Action("Enable", lambda: self.set_enabled(True)),
            Action("Disable", lambda: self.set_enabled(False)),
        ]
