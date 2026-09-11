import pytest

from uedcontrol.devices.base import DeviceError
from uedcontrol.devices.power.delta_elektronika import DeltaSM70AR24
from uedcontrol.devices.power.heinzinger import (
    READ_CURRENT,
    READ_INTERLOCK,
    READ_VOLTAGE,
    SET_INTERLOCK,
    SET_VOLTAGE,
    HeinzingerPNChp100,
    frame,
)
from uedcontrol.devices.power.matsusada import MatsusadaCOHVU32
from uedcontrol.devices.power.rohde_schwarz import RohdeSchwarzHMP4040, RohdeSchwarzNGE100
from uedcontrol.devices.transport import FakeTransport


def connected(cls, responder, **kwargs):
    transport = FakeTransport(responder)
    device = cls(transport=transport, **kwargs)
    device.connect()
    transport.written.clear()
    return device, transport


# ----------------------------------------------------------------- Matsusada
def test_matsusada_connect_enables_remote_without_touching_the_output():
    transport = FakeTransport()
    MatsusadaCOHVU32(transport=transport).connect()
    assert transport.written == [b"#1 REN\r", b"#1 RST\r"]


def test_matsusada_readings_and_current_scaling():
    hv, _ = connected(MatsusadaCOHVU32, {b"#1 VM\r": b"VM=12.34\r\n", b"#1 IM\r": b"IM=10.0\r\n"})
    assert hv.measure_voltage() == pytest.approx(12.34)
    assert hv.measure_current() == pytest.approx(150.0)  # 10 % of 1500 uA


def test_matsusada_setpoints():
    hv, transport = connected(MatsusadaCOHVU32, {})
    hv.set_voltage(12.5)
    hv.set_current(150)
    assert transport.written == [b"#1 VCN 12.5\r", b"#1 ICN 10.0\r"]


def test_matsusada_output():
    hv, transport = connected(MatsusadaCOHVU32, {b"#1 SW?\r": b"SW1\r\n"})
    assert hv.output_enabled() is True
    transport.written.clear()
    hv.set_output(True)
    assert transport.written == [b"#1 RST\r", b"#1 SW1\r"]


def test_matsusada_skips_a_stale_reply():
    hv, _ = connected(MatsusadaCOHVU32, {b"#1 VM\r": b"SW1\rVM=3.0\r"})
    assert hv.measure_voltage() == pytest.approx(3.0)


def test_matsusada_gives_up_after_retries():
    hv, transport = connected(MatsusadaCOHVU32, {}, retries=3)
    with pytest.raises(DeviceError):
        hv.measure_voltage()
    assert transport.written.count(b"#1 VM\r") == 3


# ---------------------------------------------------------------- Heinzinger
def heinzinger_responder(data):
    command = data[0]
    if command == READ_VOLTAGE:
        return frame(READ_VOLTAGE, 0xFFFF)
    if command == READ_CURRENT:
        return frame(READ_CURRENT, 0x8000)
    if command == READ_INTERLOCK:
        return bytes([0x30, 0xCF, 0xFE, 0, 0, 0])
    return data  # set commands are echoed


def test_heinzinger_frame_layout():
    assert frame(0x10) == bytes([0x10, 0xEF, 0, 0, 0, 0])
    assert frame(0x41, 0x10) == bytes([0x41, 0xBE, 0x10, 0x00, 0, 0])


def test_heinzinger_readings():
    hv, _ = connected(HeinzingerPNChp100, heinzinger_responder)
    assert hv.measure_voltage() == pytest.approx(1000 / 10.012)
    assert hv.measure_current() == pytest.approx(10000 / 10.012 * 0x8000 / 65535)
    assert hv.interlock_closed() is True
    assert hv.output_enabled() is True


def test_heinzinger_setpoints_and_interlock():
    hv, transport = connected(HeinzingerPNChp100, heinzinger_responder)
    hv.set_voltage(50.0)
    assert transport.written[-1] == frame(SET_VOLTAGE, int(10 / 1024 * 50.0 * 65535))
    hv.set_interlock(False)
    assert transport.written[-1] == frame(SET_INTERLOCK, 0x00)


def test_heinzinger_rejects_garbage():
    hv, _ = connected(HeinzingerPNChp100, lambda data: b"\x00\x01")
    with pytest.raises(DeviceError):
        hv.measure_voltage()


# ------------------------------------------------------------ Rohde & Schwarz
class FakeSCPISupply:
    def __init__(self, n_channels):
        self.channel = 1
        self.voltage = dict.fromkeys(range(1, n_channels + 1), 0.0)
        self.current = dict.fromkeys(range(1, n_channels + 1), 0.0)
        self.output = dict.fromkeys(range(1, n_channels + 1), False)
        self.general = False

    def __call__(self, data):
        command = data.decode().strip()
        if command == "*IDN?":
            return b"Rohde&Schwarz,NGE103B,1234,1.0\n"
        name, _, argument = command.partition(" ")
        if name == "INST:NSEL":
            self.channel = int(argument)
        elif name == "VOLT" and argument:
            self.voltage[self.channel] = float(argument)
        elif name == "CURR" and argument:
            self.current[self.channel] = float(argument)
        elif name == "VOLT?":
            return f"{self.voltage[self.channel]:.3f}\n".encode()
        elif name == "CURR?":
            return f"{self.current[self.channel]:.3f}\n".encode()
        elif name == "MEAS:VOLT?":
            live = self.output[self.channel] and self.general
            return f"{self.voltage[self.channel] if live else 0.0:.3f}\n".encode()
        elif name == "MEAS:CURR?":
            live = self.output[self.channel] and self.general
            return f"{self.current[self.channel] if live else 0.0:.4f}\n".encode()
        elif name in ("OUTP?", "OUTP:SEL?"):
            return b"1\n" if self.output[self.channel] else b"0\n"
        elif name == "OUTP:GEN?":
            return b"1\n" if self.general else b"0\n"
        elif name == "OUTP:GEN":
            self.general = argument == "1"
        elif name in ("OUTP", "OUTP:SEL"):
            self.output[self.channel] = argument == "1"
        return None


def test_nge100_channels():
    supply = FakeSCPISupply(3)
    lenses, transport = connected(RohdeSchwarzNGE100, supply)
    assert "NGE103B" in lenses.identify()
    lenses.set_voltage(5.0, channel=2)
    lenses.set_output(True, channel=2)
    lenses.set_master_output(True)
    assert b"OUTP 1\r\n" in transport.written
    assert lenses.voltage_setpoint(2) == pytest.approx(5.0)
    assert lenses.measure_voltage(2) == pytest.approx(5.0)
    assert lenses.output_enabled(2) and not lenses.output_enabled(1)
    status = lenses.read_status()
    assert status["CH2 Voltage"].value == pytest.approx(5.0)
    assert status["Master output"].value is True


def test_nge100_connect_does_not_reset():
    supply = FakeSCPISupply(3)
    transport = FakeTransport(supply)
    RohdeSchwarzNGE100(transport=transport).connect()
    assert b"*RST\r\n" not in transport.written


def test_hmp4040_uses_output_select():
    supply = FakeSCPISupply(4)
    hmp, transport = connected(RohdeSchwarzHMP4040, supply)
    hmp.set_output(True, channel=4)
    assert b"OUTP:SEL 1\r\n" in transport.written
    assert hmp.output_enabled(4)
    with pytest.raises(ValueError):
        hmp.set_voltage(1.0, channel=5)


# ------------------------------------------------------------------- Delta
def test_delta_parses_replies_with_eot():
    responses = {b"ME:VO?\r": b"12.5\x04\r\n", b"ME:CU?\r": b"0.25\x04\r\n", b"SO:FU:OUTP?\r": b"1\r\n"}
    supply, transport = connected(DeltaSM70AR24, responses)
    assert supply.measure_voltage() == pytest.approx(12.5)
    assert supply.measure_current() == pytest.approx(0.25)
    assert supply.output_enabled() is True
    supply.set_current(1.5)
    assert transport.written[-1] == b"SO:CU 1.5000\r"
