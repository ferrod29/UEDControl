from pathlib import Path

import pytest
import yaml

from uedcontrol.config import ConfigError, load_config, parse_config, simulated_config
from uedcontrol.devices.base import Device
from uedcontrol.devices.registry import DRIVERS, create_device, driver_class

ROOT = Path(__file__).resolve().parents[1]


def test_every_registered_driver_imports():
    for key in DRIVERS:
        assert issubclass(driver_class(key), Device)
    with pytest.raises(KeyError):
        driver_class("does_not_exist")


def test_simulated_config_builds_and_connects_every_device():
    config = simulated_config()
    assert config.delay_line.stage == "delay_stage"
    assert config.rf_interlock.reflected_sensor == "rf_reflected"
    for spec in config.devices.values():
        device = spec.create()
        device.connect()
        device.read_status()
        device.disconnect()


def test_example_config_parses_and_instantiates_without_hardware():
    config = load_config(ROOT / "config" / "example_instrument.yaml")
    assert config.devices["lenses"].panel["channel_names"] == ["Condenser", "Horizontal", "Vertical"]
    assert config.devices["gun_hv"].options == {"port": "COM16", "full_scale_current_ua": 1500}
    for spec in config.devices.values():
        assert isinstance(spec.create(), Device)


def test_config_errors():
    with pytest.raises(ConfigError, match="unknown driver"):
        parse_config({"devices": {"x": {"driver": "nope"}}})
    with pytest.raises(ConfigError, match="needs a 'driver'"):
        parse_config({"devices": {"x": {"port": "COM1"}}})
    with pytest.raises(ConfigError, match="unknown top-level"):
        parse_config({"colour": "blue"})
    with pytest.raises(ConfigError, match="unknown device"):
        parse_config({"delay_line": {"stage": "missing"}})
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config({"beam": {"rep_rate": 1}})


def test_relative_paths_resolve_next_to_the_file(tmp_path):
    path = tmp_path / "setup.yaml"
    path.write_text(yaml.safe_dump({"data_dir": "data", "devices": {"cam": {"driver": "simulated_camera"}}}))
    config = load_config(path)
    assert config.data_dir == tmp_path / "data"
    assert config.log_dir == tmp_path / "data" / "logs"
    assert create_device("cam", "simulated_camera").name == "cam"
