"""End-to-end GUI test on the simulated instrument (offscreen, no hardware, no display)."""

import os
import time

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pyqtgraph.Qt", reason="needs pyqtgraph with a Qt binding")

from uedcontrol.config import ConfigError, simulated_config  # noqa: E402
from uedcontrol.gui.app import parse_args, resolve_config  # noqa: E402
from uedcontrol.gui.log_panel import setup_logging  # noqa: E402
from uedcontrol.gui.main_window import MainWindow  # noqa: E402
from uedcontrol.gui.qt import QtCore, QtWidgets  # noqa: E402
from uedcontrol.gui.scan_panel import format_duration, safe_name  # noqa: E402
from uedcontrol.instrument import Instrument  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_until(app, condition, timeout=15.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        time.sleep(0.01)


@pytest.fixture
def window(app, tmp_path):
    fmt, scope = QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope
    QtCore.QSettings.setPath(fmt, scope, str(tmp_path / "settings"))  # never touch the real user settings
    config = simulated_config()
    config.data_dir = tmp_path / "data"
    config.log_dir = tmp_path / "logs"
    win = MainWindow(Instrument.from_config(config), setup_logging(None, "WARNING"))
    win.show()
    yield win
    win.close()
    assert not win.acquisition_thread.isRunning()
    assert not any(device.connected for device in win.instrument.devices.values())


def test_window_layout(window):
    groups = [window.instrument_tabs.tabText(i) for i in range(window.instrument_tabs.count())]
    assert groups == ["RF", "Delay line", "Electron gun", "Electron optics", "Laser", "Vacuum"]
    assert len(window.panels) == len(window.instrument.devices) - 2  # the cameras have no instrument panel
    assert window.rf_panel is not None
    assert window.acquisition_panel.camera_combo.count() == 2


def test_live_view_and_pump_probe_scan(app, window, tmp_path):
    acquisition, scan = window.acquisition_panel, window.scan_panel
    # autoconnect: detector and delay stage
    wait_until(app, lambda: acquisition.camera is not None and window.instrument["delay_stage"].connected)
    acquisition.exposure.setValue(2.0)
    wait_until(app, lambda: acquisition.frame_rate_hz() == pytest.approx(20.0))

    acquisition.start_live()
    wait_until(app, lambda: acquisition.last_image is not None)
    assert acquisition.last_image.shape == (512, 512)
    acquisition.stop_live()
    wait_until(app, lambda: not acquisition.is_live())

    scan.start.setValue(-1.0)
    scan.stop.setValue(4.0)
    scan.step.setValue(1.0)
    scan.frames.setValue(2)
    scan.runs.setValue(2)
    scan.name.setText("gold/film?")  # unsafe characters are replaced
    scan.timestamp.setChecked(False)
    scan.sample.setText("Au")
    assert "6 delays" in scan.preview.text()
    assert scan.start_scan()
    assert window.central_tabs.currentWidget() is window.pumpprobe_view
    assert not acquisition.live_button.isEnabled()
    wait_until(app, lambda: not scan.running, timeout=30)

    assert scan.status.text().startswith("Finished: 12 point(s)"), scan.status.text()
    assert scan.start_button.text() == "Start scan" and acquisition.live_button.isEnabled()
    folder = tmp_path / "data" / "gold_film"
    assert scan.last_scan_dir == folder
    with h5py.File(folder / "gold_film.h5") as handle:
        assert handle.attrs["sample"] == "Au"
        assert handle.attrs["mode"] == "delay scan"
        assert len(handle["images"]) == 12
        total = handle["results/roi_total"][()]
    assert total[-1] < total[0]  # the simulated sample responds after T0
    assert (folder / "pp_avgi_data.txt").exists()


def test_scan_errors_are_reported_without_running(app, window):
    acquisition, scan = window.acquisition_panel, window.scan_panel
    wait_until(app, lambda: acquisition.camera is not None and window.instrument["delay_stage"].connected)
    scan.start.setValue(-5000.0)
    scan.stop.setValue(0.0)
    scan.step.setValue(1000.0)
    assert scan.start_scan()  # accepted, then refused by the engine before moving
    wait_until(app, lambda: not scan.running)
    assert "outside the stage range" in scan.status.text()
    assert scan.last_scan_dir is None

    scan.extra.setText("1:2")
    assert not scan.start_scan()
    assert "start:stop:step" in scan.status.text()


def test_time_series_and_analysis_window(app, window):
    acquisition, scan = window.acquisition_panel, window.scan_panel
    wait_until(app, lambda: acquisition.camera is not None)
    acquisition.exposure.setValue(2.0)
    scan.series_mode.setChecked(True)
    scan.datasets.setValue(3)
    scan.save_box.setChecked(False)
    assert scan.start_scan()
    wait_until(app, lambda: not scan.running, timeout=30)
    assert scan.status.text().startswith("Finished: 3 point(s)"), scan.status.text()

    acquisition._send_to_analysis()
    analysis = window.analysis_window
    assert analysis is not None and analysis.isVisible()
    assert analysis.image.shape == (512, 512)
    analysis.roi_button.click()  # ROI on, centred on the image
    analysis.analyse_beam()
    assert analysis.beam_table.item(0, 1).text() != "–"
    analysis.axis_unit.setCurrentIndex(1)  # radius in px
    analysis.detect_rings()
    radii = sorted(peak.radius_px for peak in analysis.peaks)
    # the simulated rings are at 16 %, 26 % and 36 % of 512 px
    for expected in (0.16 * 512, 0.26 * 512, 0.36 * 512):
        assert any(abs(r - expected) < 3 for r in radii), (expected, radii)
    assert analysis.ring_table.rowCount() == len(analysis.peaks)
    assert np.all(np.diff([p.q for p in sorted(analysis.peaks, key=lambda p: p.radius_px)]) > 0)


def test_connect_and_disconnect_all(app, window):
    window.connect_all()
    others = [panel.device for panel in window.panels]
    wait_until(app, lambda: all(device.connected for device in others))
    wait_until(app, lambda: all(panel.body.isEnabled() for panel in window.panels))
    window.disconnect_all()
    wait_until(app, lambda: not any(device.connected for device in others))


def test_command_line(monkeypatch, tmp_path):
    assert resolve_config(parse_args(["--simulate"])).name == "UED (simulated)"
    monkeypatch.delenv("UEDCONTROL_CONFIG", raising=False)
    with pytest.raises(ConfigError, match="no configuration"):
        resolve_config(parse_args([]))
    path = tmp_path / "setup.yaml"
    path.write_text("name: Lab\ndevices: {}\n")
    monkeypatch.setenv("UEDCONTROL_CONFIG", str(path))
    assert resolve_config(parse_args([])).name == "Lab"


def test_helpers():
    assert safe_name('a/b:c*"d') == "a_b_c_d"
    assert safe_name(" ..") == "scan"
    assert format_duration(59.6) == "1:00"
    assert format_duration(3725) == "1:02:05"
