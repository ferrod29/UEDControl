import pytest

from uedcontrol.devices.base import DeviceError
from uedcontrol.devices.sensors.agilent_twistorr import (
    ACK,
    ETX,
    NACK,
    STX,
    AgilentTwisTorr,
    build_frame,
    checksum,
)
from uedcontrol.devices.sensors.edwards_tic import EdwardsTIC
from uedcontrol.devices.sensors.newport_1935c import Newport1935C
from uedcontrol.devices.thermal.smc_hecr import (
    checksum as hecr_checksum,
)
from uedcontrol.devices.thermal.smc_hecr import (
    decode_offset,
    decode_temperature,
    encode_offset,
    encode_temperature,
)
from uedcontrol.devices.timing.delay_generator import frame as delay_frame
from uedcontrol.devices.transport import FakeTransport


# ---------------------------------------------------------------- TwisTorr
def reply_frame(payload: bytes, address: int = 0x80) -> bytes:
    body = bytes([address]) + payload + bytes([ETX])
    return bytes([STX]) + body + checksum(body)


def test_twistorr_frame_matches_the_documented_example():
    # "start" command: STX 0x80 "000" "1" "1" ETX CRC="B3"
    assert build_frame(0x80, 0, True, "1") == bytes([0x02, 0x80, 0x30, 0x30, 0x30, 0x31, 0x31, 0x03]) + b"B3"


def test_twistorr_checksum_is_two_uppercase_digits():
    assert checksum(bytes([0x0A])) == b"0A"
    assert checksum(bytes([0x80, 0x80])) == b"00"


def test_twistorr_read_and_write():
    def responder(data):
        window = data[2:5]
        if data[5:6] == b"1":  # write
            return reply_frame(bytes([ACK]))
        values = {b"205": b"000005", b"224": b"1.2E-07   ", b"204": b"000043"}
        return reply_frame(window + b"0" + values[window])

    transport = FakeTransport(responder)
    pump = AgilentTwisTorr(transport=transport)
    pump.connect()
    assert pump.status() == "Normal"
    assert pump.pressure() == pytest.approx(1.2e-7)
    assert pump.temperature() == 43.0
    pump.start()
    assert transport.written[-1] == build_frame(0x80, 0, True, "1")


def test_twistorr_errors():
    pump = AgilentTwisTorr(transport=FakeTransport(lambda data: reply_frame(bytes([NACK]))))
    pump.connect()
    with pytest.raises(DeviceError):
        pump.start()
    corrupted = AgilentTwisTorr(transport=FakeTransport(lambda data: reply_frame(bytes([ACK]))[:-1] + b"Z"))
    corrupted.connect()
    with pytest.raises(DeviceError):
        corrupted.stop()


# ------------------------------------------------------------------ Edwards
def edwards():
    responses = {
        b"?V913\r": b"=V913 1.00e-05;59;11;0;0\r",
        b"?V904\r": b"=V904 4;0;0\r",
        b"?V905\r": b"=V905 100;0;0\r",
        b"?V906\r": b"=V906 15.5;0;0\r",
        b"!C904 1\r": b"*C904 0\r",
        b"!C913 1\r": b"*C913 5\r",
        b"?S913 5\r": b"=S913 5;10\r",
    }
    tic = EdwardsTIC(transport=FakeTransport(responses), gauges=(1,))
    tic.connect()
    return tic


def test_edwards_gauge_reading():
    reading = edwards().gauge(1)
    assert reading.value == pytest.approx(1e-5)
    assert reading.unit == "Pa"
    assert reading.state == "On"
    assert reading.pressure_mbar == pytest.approx(1e-7)


def test_edwards_turbo_and_errors():
    tic = edwards()
    assert tic.turbo_state() == "Running"
    tic.set_turbo(True)
    with pytest.raises(DeviceError, match="invalid command in current state"):
        tic.gauge_command(1, "on")
    assert tic.gauge_type(1) == "APGXH"
    status = tic.read_status()
    assert status["Gauge 1"].unit == "mbar"
    assert status["Turbo power"].value == pytest.approx(15.5)


# ----------------------------------------------------------- Newport 1935-C
def test_newport_1935c_handles_command_echo():
    responses = {
        b"*IDN?\r": b"NEWPORT 1935-C v1.0\r\n",
        b"PM:Power?\r": b"PM:Power?\r\n1.500E-03\r\n",
        b"PM:Lambda?\r": b"267\r\n",
    }
    meter = Newport1935C(transport=FakeTransport(responses))
    meter.connect()
    assert meter.identify() == "NEWPORT 1935-C v1.0"
    assert meter.power() == pytest.approx(1.5e-3)
    assert meter.wavelength() == 267.0


# --------------------------------------------------- unverified protocols
def test_hecr_data_encoding():
    assert encode_temperature(25.0) == b"2500"
    assert decode_temperature(b"2500") == 25.0
    assert encode_offset(-1.25) == b"-125"
    assert decode_offset(b"-125") == -1.25
    assert decode_offset(encode_offset(0.5)) == 0.5
    assert hecr_checksum(bytes([0x31])) == bytes([0x33, 0x31])
    with pytest.raises(ValueError):
        encode_temperature(120)


def test_delay_generator_frame():
    assert delay_frame(0x11, 1) == bytes([0x11, 0xEE, 1, 0, 0, 0])
    assert delay_frame(0xF3) == bytes([0xF3, 0x0C, 0, 0, 0, 0])
