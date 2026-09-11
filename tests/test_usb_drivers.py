import struct

import pytest

from uedcontrol.devices.rf.minicircuits import MiniCircuitsAttenuator, MiniCircuitsPowerSensor
from uedcontrol.devices.rf.statera import GET, DrXWorksStatera100


class FakeLink:
    def __init__(self, handler):
        self.handler = handler
        self.sent = []
        self.is_open = False

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def exchange(self, payload, response_size=None):
        self.sent.append(bytes(payload))
        return self.handler(bytes(payload))


def text_reply(code, text):
    return (bytes([code]) + text.encode() + b"\0").ljust(64, b"\0")


def test_power_sensor():
    def handler(payload):
        if payload[0] == 102:
            return text_reply(102, "-12.34")
        if payload[0] == 103:
            return text_reply(103, "+25.5")
        return text_reply(payload[0], "12210060144")

    link = FakeLink(handler)
    sensor = MiniCircuitsPowerSensor(link=link, frequency_mhz=3000, coupling_db=30.0)
    sensor.connect()
    assert link.is_open
    assert sensor.power_dbm() == pytest.approx(-12.34)
    assert link.sent[-1] == bytes([102, 3000 // 256, 3000 % 256, ord("M")])
    assert sensor.temperature() == pytest.approx(25.5)
    assert sensor.serial() == "12210060144"
    assert sensor.line_power_w(-12.34) == pytest.approx(10 ** ((-12.34 + 30) / 10) * 1e-3)
    status = sensor.read_status()
    assert set(status) == {"Power", "Power (mW)", "Line power", "Temperature"}
    sensor.disconnect()
    assert not link.is_open


def test_attenuator():
    state = {"att": 0.0}

    def handler(payload):
        command = payload[1:].decode()
        if command == ":ATT?":
            return text_reply(1, f"{state['att']:.2f}")
        if command.startswith(":SETATT="):
            state["att"] = float(command.split("=")[1])
        return text_reply(1, "1")

    link = FakeLink(handler)
    attenuator = MiniCircuitsAttenuator(link=link)
    attenuator.connect()
    attenuator.set_attenuation(15.25)
    assert link.sent[-1] == b"\x01:SETATT=15.25"
    assert attenuator.attenuation() == pytest.approx(15.25)
    with pytest.raises(ValueError):
        attenuator.set_attenuation(130)


def test_statera_amplifier():
    values = {"Swi": 1.0, "Tem": 31.5, "Rem": 1.0}

    def handler(payload):
        code, operation = payload[:3].decode(), payload[3]
        if operation == GET:
            return b"OK\0\0" + struct.pack("<f", values[code])
        values[code] = struct.unpack("<f", payload[4:8])[0]
        return b"OK\0\0\0\0\0\0"

    link = FakeLink(handler)
    amplifier = DrXWorksStatera100(link=link)
    amplifier.connect()
    assert amplifier.rf_enabled() is True
    assert amplifier.temperature() == pytest.approx(31.5)
    amplifier.set_rf_enabled(False)
    assert link.sent[-1] == b"Swi=" + struct.pack("<f", 0.0)
    assert amplifier.rf_enabled() is False
