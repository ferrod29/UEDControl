import pytest

from uedcontrol.acquisition.safety import BreakdownDetector, PowerInterlock, RippleMeter, VoltageRamp


def test_power_interlock_latches_once():
    interlock = PowerInterlock(limit=-10.0)
    assert not interlock.check(0.0)  # not armed
    interlock.arm()
    assert not interlock.check(-20.0)
    assert interlock.check(-5.0)
    assert interlock.tripped
    assert not interlock.check(-1.0)  # reported only once
    interlock.reset()
    assert interlock.check(-1.0)


def test_voltage_ramp_up_stops_exactly_at_target():
    ramp = VoltageRamp(start=0.0, target=1.0, rate_per_min=6.0, interval_s=5.0)  # 0.5 kV per step
    assert ramp.step == pytest.approx(0.5)
    assert [ramp.next_setpoint(), ramp.next_setpoint()] == [pytest.approx(0.5), pytest.approx(1.0)]
    assert ramp.done and ramp.next_setpoint() is None


def test_voltage_ramp_down_and_uneven_steps():
    ramp = VoltageRamp(start=2.0, target=0.8, rate_per_min=6.0, interval_s=5.0)
    values = []
    while (value := ramp.next_setpoint()) is not None:
        values.append(value)
    assert values == [pytest.approx(1.5), pytest.approx(1.0), pytest.approx(0.8)]
    with pytest.raises(ValueError):
        VoltageRamp(0, 1, rate_per_min=0)


def test_breakdown_detector():
    detector = BreakdownDetector(drop=1.0)
    assert not detector.update(30.0)
    assert not detector.update(29.5)
    assert detector.update(25.0)
    assert not detector.update(20.0, output_on=False)
    assert not detector.update(10.0)  # first reading after the output came back


def test_ripple_meter():
    meter = RippleMeter(window=3)
    assert meter.update(1.0) == 0.0
    meter.update(1.0)
    assert meter.update(1.0) == 0.0
    assert meter.update(4.0) > 0
