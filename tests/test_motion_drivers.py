import struct

import pytest

from uedcontrol.devices.motion.newport_agilis import NewportAgilisUC8
from uedcontrol.devices.motion.newport_smc100 import NewportSMC100, state_name
from uedcontrol.devices.motion.thorlabs_kdc101 import (
    PRM1_Z8_COUNTS_PER_DEGREE,
    ThorlabsKDC101,
    long_message,
    short_message,
)
from uedcontrol.devices.transport import FakeTransport


# ------------------------------------------------------------------ SMC100
class FakeSMC100:
    def __init__(self):
        self.position = 1.0
        self.moving_polls = 0

    def __call__(self, data):
        body = data.decode().strip()[1:]
        if body == "TS":
            if self.moving_polls:
                self.moving_polls -= 1
                return b"1TS000028\r\n"
            return b"1TS000033\r\n"
        if body == "TP":
            return f"1TP{self.position:.5f}\r\n".encode()
        if body.startswith("PA"):
            self.position = float(body[2:])
            self.moving_polls = 2
        elif body == "SL?":
            return b"1SL-25\r\n"
        elif body == "SR?":
            return b"1SR25\r\n"
        return None


def test_smc100_move_and_readback():
    transport = FakeTransport(FakeSMC100())
    stage = NewportSMC100(transport=transport)
    stage.connect()
    assert stage.position() == pytest.approx(1.0)
    stage.move_to(3.5)
    assert b"1PA3.50000\r\n" in transport.written
    assert stage.position() == pytest.approx(3.5)
    assert not stage.is_moving()
    assert stage.limits() == (-25.0, 25.0)
    assert stage.read_status()["State"].value == "Ready"


def test_smc100_state_names():
    assert state_name("0A") == "Not referenced"
    assert state_name("10") == "Not referenced"
    assert state_name("28") == "Moving"
    assert state_name("99").startswith("Unknown")


# ------------------------------------------------------------------ Agilis
def test_agilis_commands():
    transport = FakeTransport({b"1TS\r\n": b"1TS0\r\n", b"2TS\r\n": b"2TS1\r\n", b"VE\r\n": b"AG-UC8 v2.2.1\r\n"})
    mirrors = NewportAgilisUC8(transport=transport, command_interval_s=0)
    mirrors.connect()
    assert transport.written == [b"RS\r\n", b"MR\r\n"]
    transport.written.clear()

    mirrors.select_channel(3)
    mirrors.step(1, -50)
    mirrors.jog(2, 3)
    mirrors.stop()
    assert transport.written == [b"CC3\r\n", b"1PR-50\r\n", b"2JA3\r\n", b"1ST\r\n", b"2ST\r\n"]

    transport.written.clear()
    mirrors.set_step_amplitude(20)
    assert transport.written == [b"1SU20\r\n", b"1SU-20\r\n", b"2SU20\r\n", b"2SU-20\r\n"]

    assert mirrors.is_moving(1) is False
    assert mirrors.is_moving(2) is True
    assert mirrors.identify() == "AG-UC8 v2.2.1"
    with pytest.raises(ValueError):
        mirrors.jog(1, 5)
    with pytest.raises(ValueError):
        mirrors.select_channel(5)


# ------------------------------------------------------------------ KDC101
def dc_status_reply(counts, bits):
    data = struct.pack("<HlHHI", 1, counts, 0, 0, bits)
    return struct.pack("<HHBB", 0x0491, len(data), 0x81, 0x50) + data


class FakeKDC101:
    def __init__(self):
        self.counts = 0
        self.enabled = 0x01

    def __call__(self, data):
        msg_id = struct.unpack_from("<H", data)[0]
        if msg_id == 0x0490:
            return dc_status_reply(self.counts, 0x80000400)
        if msg_id == 0x0453:
            _channel, self.counts = struct.unpack("<Hl", data[6:])
        elif msg_id == 0x0211:
            return struct.pack("<HBBBB", 0x0212, 1, self.enabled, 0x01, 0x50)
        elif msg_id == 0x0210:
            self.enabled = data[3]
        return None


def test_kdc101_message_builders():
    assert short_message(0x0443, 1) == bytes([0x43, 0x04, 0x01, 0x00, 0x50, 0x01])
    message = long_message(0x0453, struct.pack("<Hl", 1, 100))
    assert message[:6] == bytes([0x53, 0x04, 0x06, 0x00, 0xD0, 0x01])  # data flag set
    assert len(message) == 12


def test_kdc101_move_position_and_enable():
    controller = FakeKDC101()
    transport = FakeTransport(controller)
    stage = ThorlabsKDC101(transport=transport)
    stage.connect()
    assert stage.position() == 0.0
    stage.move_to(10.0)
    assert controller.counts == round(10.0 * PRM1_Z8_COUNTS_PER_DEGREE)
    assert stage.position() == pytest.approx(10.0, abs=1e-3)
    assert "Homed" in stage.status_flags()
    assert not stage.is_moving()

    stage.set_enabled(False)
    assert transport.written[-1] == bytes([0x10, 0x02, 0x01, 0x02, 0x50, 0x01])
    assert stage.is_enabled() is False
