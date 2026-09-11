import pytest

from uedcontrol.config import ConfigError, parse_config, simulated_config
from uedcontrol.devices.simulated import SimulatedCamera, SimulatedRFPowerSensor
from uedcontrol.instrument import Instrument


def test_simulated_instrument_is_wired_together():
    instrument = Instrument.from_config(simulated_config())
    assert set(instrument.cameras()) == {"detector", "beam_camera"}
    assert instrument.delay_line is not None and instrument.delay_line.stage is instrument["delay_stage"]
    assert instrument.delay_line.t0_position == 150.0

    # The simulated detector follows the simulated delay line ...
    detector = instrument["detector"]
    assert isinstance(detector, SimulatedCamera)
    assert detector.delay_source() == pytest.approx(0.0)
    # ... and the RF sensors only see power while the amplifier is on.
    sensor = instrument["rf_reflected"]
    assert isinstance(sensor, SimulatedRFPowerSensor)
    amplifier = instrument["rf_amplifier"]
    assert sensor.power_dbm() < -50
    amplifier.set_rf_enabled(True)
    assert sensor.power_dbm() > -30


def test_connect_and_context_manager_disconnect():
    with Instrument.from_config(simulated_config()) as instrument:
        instrument.connect("delay_stage", "detector")
        assert instrument["delay_stage"].connected and instrument["detector"].connected
        assert not instrument["lenses"].connected
    assert not any(device.connected for device in instrument.devices.values())


def test_bad_driver_options_are_reported_as_config_errors():
    config = parse_config({"devices": {"cam": {"driver": "simulated_camera", "pattern": "stripes"}}})
    with pytest.raises(ConfigError, match="'cam'.*pattern"):
        Instrument.from_config(config)
    config = parse_config({"devices": {"cam": {"driver": "simulated_camera", "colour": "blue"}}})
    with pytest.raises(ConfigError, match="'cam'"):
        Instrument.from_config(config)


def test_device_roles_are_type_checked():
    config = parse_config({
        "devices": {"cam": {"driver": "simulated_camera"}},
        "delay_line": {"stage": "cam"},
    })
    with pytest.raises(ConfigError, match="delay_line.stage.*not a Positioner"):
        Instrument.from_config(config)
    config = parse_config({
        "devices": {"amp": {"driver": "simulated_rf_amplifier"}, "stage": {"driver": "simulated_stage"}},
        "rf_interlock": {"amplifier": "amp", "reflected_sensor": "stage"},
    })
    with pytest.raises(ConfigError, match="reflected_sensor.*not a RFPowerSensor"):
        Instrument.from_config(config)
    config = parse_config({
        "devices": {"stage": {"driver": "simulated_stage"}},
        "delay_line": {"stage": "stage", "passes": 0},
    })
    with pytest.raises(ConfigError, match="passes"):
        Instrument.from_config(config)
