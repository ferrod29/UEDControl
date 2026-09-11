import pytest

from uedcontrol.devices.base import DeviceError, NotConnectedError, Reading
from uedcontrol.devices.transport import FakeTransport, SerialInstrument, parse_float


class EchoInstrument(SerialInstrument):
    model = "Echo"

    def _initialize(self):
        self.write("HELLO")


def test_fake_transport_mapping_and_read_until():
    transport = FakeTransport({b"A?\n": b"1\r\n2\r\n"})
    transport.open()
    transport.write(b"A?\n")
    assert transport.read_until(b"\n") == b"1\r\n"
    assert transport.read_until(b"\n") == b"2\r\n"
    assert transport.read_until(b"\n") == b""
    assert transport.written == [b"A?\n"]


def test_fake_transport_refuses_writes_when_closed():
    with pytest.raises(NotConnectedError):
        FakeTransport().write(b"x")


def test_serial_instrument_lifecycle_and_query():
    transport = FakeTransport({b"VAL?\r\n": b"VAL=3.25\r\n"})
    device = EchoInstrument(transport=transport, name="echo")
    assert not device.connected
    device.connect()
    assert device.connected and transport.is_open
    assert transport.written == [b"HELLO\r\n"]
    assert device.query_float("VAL?") == pytest.approx(3.25)
    device.disconnect()
    assert not transport.is_open and not device.connected


def test_query_without_reply_raises():
    device = EchoInstrument(transport=FakeTransport())
    device.connect()
    with pytest.raises(DeviceError):
        device.query("NOTHING?")


def test_port_or_transport_is_required():
    with pytest.raises(ValueError):
        EchoInstrument()
    with pytest.raises(TypeError):
        EchoInstrument(transport=FakeTransport(), baudrate=9600)


@pytest.mark.parametrize(
    "text, value",
    [("VM=12.30", 12.3), ("1TP-3.5", -3.5), ("1.2E-05", 1.2e-5), (" 42 ", 42.0), ("SW1", 1.0)],
)
def test_parse_float(text, value):
    assert parse_float(text) == pytest.approx(value)


def test_parse_float_without_number():
    with pytest.raises(DeviceError):
        parse_float("ERR")


def test_reading_formatting():
    assert Reading(True).formatted() == "ON"
    assert Reading(1.23456, "kV").formatted(3) == "1.23 kV"
    assert Reading("Ready").formatted() == "Ready"
