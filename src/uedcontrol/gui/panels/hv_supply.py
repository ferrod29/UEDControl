"""Panel for the electron-gun high-voltage supply.

Features carried over from the original HVPS GUIs: set points, rate-limited
voltage ramp (steps every 5 s), output and interlock switches, voltage /
ripple / current read-back with time plots, breakdown detection and logging,
and saving of the recorded data. The ramp is cancelled when the interlock
opens.

Panel options: ``max_voltage_kv``, ``max_current_ua``, ``breakdown_drop_kv``.
"""

from __future__ import annotations

import datetime as dt
import time

from ...acquisition.safety import BreakdownDetector, RippleMeter, VoltageRamp
from ...devices.interfaces import HighVoltageSupply
from ..qt import QtCore, QtWidgets
from ..widgets.common import Readout, ToggleButton, save_file_name, spin, write_csv
from ..widgets.time_plot import TimeSeriesPlot
from .base import DevicePanel


class HVSupplyPanel(DevicePanel):
    default_poll_interval_s = 0.5
    ramp_interval_s = 5.0
    history_limit = 200_000

    def build(self, body: QtWidgets.QWidget) -> None:
        supply: HighVoltageSupply = self.device  # type: ignore[assignment]
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)

        settings = QtWidgets.QGridLayout()
        self.voltage = spin(0.0, 0.0, float(self.options.get("max_voltage_kv", 100.0)), 3, 0.1, "kV")
        self.current = spin(100.0, 0.0, float(self.options.get("max_current_ua", 2000.0)), 1, 10.0, "µA")
        self.ramp_enabled = QtWidgets.QCheckBox("Ramp at")
        self.ramp_rate = spin(1.0, 0.01, 100.0, 2, 0.1, "kV/min")
        apply_button = QtWidgets.QPushButton("Apply")
        stop_ramp_button = QtWidgets.QPushButton("Stop ramp")
        apply_button.clicked.connect(self._apply)
        stop_ramp_button.clicked.connect(lambda: self._stop_ramp())
        settings.addWidget(QtWidgets.QLabel("Voltage set point"), 0, 0)
        settings.addWidget(self.voltage, 0, 1)
        settings.addWidget(QtWidgets.QLabel("Current limit"), 1, 0)
        settings.addWidget(self.current, 1, 1)
        settings.addWidget(self.ramp_enabled, 2, 0)
        settings.addWidget(self.ramp_rate, 2, 1)
        settings.addWidget(apply_button, 3, 0)
        settings.addWidget(stop_ramp_button, 3, 1)
        layout.addLayout(settings)

        switches = QtWidgets.QHBoxLayout()
        self.output_button: ToggleButton | None = None
        self.interlock_button: ToggleButton | None = None
        if supply.separate_output:
            self.output_button = ToggleButton("Output ON", "Output OFF")
            self.output_button.clicked.connect(lambda checked: self.run(
                supply.set_output, checked, description=f"HV output {'on' if checked else 'off'}"))
            switches.addWidget(self.output_button)
        if supply.has_interlock:
            self.interlock_button = ToggleButton("Interlock closed", "Interlock open")
            self.interlock_button.clicked.connect(lambda checked: self.run(
                supply.set_interlock, checked, description=f"HV interlock {'closed' if checked else 'opened'}"))
            switches.addWidget(self.interlock_button)
        layout.addLayout(switches)

        readings = QtWidgets.QFormLayout()
        self.voltage_read = Readout()
        self.current_read = Readout()
        self.ripple_read = Readout()
        self.breakdown_count = Readout("0")
        self.breakdown_drop = spin(float(self.options.get("breakdown_drop_kv", 1.0)), 0.01, 100.0, 2, 0.1, "kV")
        readings.addRow("Voltage (kV)", self.voltage_read)
        readings.addRow("Current (µA)", self.current_read)
        readings.addRow("Ripple (V)", self.ripple_read)
        readings.addRow("Breakdowns", self.breakdown_count)
        readings.addRow("Breakdown = drop of", self.breakdown_drop)
        layout.addLayout(readings)
        self.ramp_label = QtWidgets.QLabel()
        layout.addWidget(self.ramp_label)

        buttons = QtWidgets.QHBoxLayout()
        self.plot_button = QtWidgets.QPushButton("Show plots")
        self.plot_button.setCheckable(True)
        self.plot_button.toggled.connect(self._toggle_plots)
        save_button = QtWidgets.QPushButton("Save data…")
        save_button.clicked.connect(self._save)
        clear_button = QtWidgets.QPushButton("Clear data")
        clear_button.clicked.connect(self._clear)
        for button in (self.plot_button, save_button, clear_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.plots = TimeSeriesPlot([("Voltage", "kV"), ("Ripple", "V"), ("Current", "µA")])
        self.plots.setMinimumHeight(420)
        self.plots.hide()
        layout.addWidget(self.plots)

        self.ramp_timer = QtCore.QTimer(self)
        self.ramp_timer.timeout.connect(self._ramp_step)
        self.ramp: VoltageRamp | None = None
        self.ripple = RippleMeter(10)
        self.detector = BreakdownDetector(self.breakdown_drop.value())
        self.records: list[tuple[float, float, float, float]] = []
        self.breakdowns: list[tuple[float, float]] = []
        self._last_voltage: float | None = None

    # ------------------------------------------------------------- set points
    def on_connected(self) -> None:
        supply: HighVoltageSupply = self.device  # type: ignore[assignment]
        self.controller.submit(lambda: (supply.voltage_setpoint(), supply.current_setpoint()),
                               done=self._show_setpoints)

    def _show_setpoints(self, values: tuple[float | None, float | None]) -> None:
        voltage, current = values
        if voltage is not None:
            self.voltage.setValue(voltage)
        if current is not None:
            self.current.setValue(current)

    def _apply(self) -> None:
        supply: HighVoltageSupply = self.device  # type: ignore[assignment]
        target, limit = self.voltage.value(), self.current.value()
        self.run(supply.set_current, limit, description=f"HV current limit -> {limit:g} µA")
        if self.ramp_enabled.isChecked():
            start = self._last_voltage if self._last_voltage is not None else 0.0
            self.ramp = VoltageRamp(start, target, self.ramp_rate.value(), self.ramp_interval_s)
            self.log.info("HV ramp %.3f -> %.3f kV at %.2f kV/min", start, target, self.ramp_rate.value())
            self.ramp_timer.start(int(self.ramp_interval_s * 1000))
            self._ramp_step()
        else:
            self._stop_ramp()
            self.run(supply.set_voltage, target, description=f"HV voltage -> {target:g} kV")

    def _ramp_step(self) -> None:
        if self.ramp is None:
            self.ramp_timer.stop()
            return
        value = self.ramp.next_setpoint()
        if value is None:
            self._stop_ramp(finished=True)
            return
        supply: HighVoltageSupply = self.device  # type: ignore[assignment]
        self.run(supply.set_voltage, value)
        self.ramp_label.setText(f"Ramping: {value:.3f} kV (target {self.ramp.target:.3f} kV)")

    def _stop_ramp(self, finished: bool = False) -> None:
        self.ramp_timer.stop()
        if self.ramp is not None:
            self.log.info("HV ramp %s", "finished" if finished else "stopped")
        self.ramp = None
        self.ramp_label.setText("Ramp finished" if finished else "")

    # ----------------------------------------------------------------- polling
    def poll(self) -> tuple[float, float, bool | None, bool | None]:
        supply: HighVoltageSupply = self.device  # type: ignore[assignment]
        with supply.lock:
            voltage = supply.measure_voltage()
            current = supply.measure_current()
            output = supply.output_enabled() if supply.separate_output else None
            interlock = supply.interlock_closed() if supply.has_interlock else None
        return voltage, current, output, interlock

    def on_status(self, status: tuple[float, float, bool | None, bool | None]) -> None:
        voltage, current, output, interlock = status
        now = time.time()
        self._last_voltage = voltage
        ripple_v = self.ripple.update(voltage) * 1000.0
        live = output if output is not None else (interlock if interlock is not None else True)
        self.detector.drop = self.breakdown_drop.value()
        if self.detector.update(voltage, bool(live)):
            self.breakdowns.append((now, voltage))
            self.breakdown_count.setText(str(len(self.breakdowns)))
            self.log.warning("HV breakdown: voltage dropped to %.3f kV", voltage)
        self.voltage_read.show_value(voltage, "{:.3f}")
        self.current_read.show_value(current, "{:.2f}")
        self.ripple_read.show_value(ripple_v, "{:.1f}")
        if self.output_button is not None and output is not None:
            self.output_button.setChecked(output)
        if self.interlock_button is not None and interlock is not None:
            self.interlock_button.setChecked(interlock)
            if not interlock and self.ramp is not None:
                self._stop_ramp()
                self.log.warning("interlock open: HV ramp cancelled")
        self.records.append((now, voltage, current, ripple_v))
        if len(self.records) > self.history_limit:  # about 28 h at 0.5 s; breakdowns are kept apart
            del self.records[: self.history_limit // 10]
        if self.plots.isVisible():
            self.plots.append(now, (voltage, ripple_v, current))

    # ------------------------------------------------------------------- data
    def _toggle_plots(self, visible: bool) -> None:
        self.plot_button.setText("Hide plots" if visible else "Show plots")
        if visible:
            self.plots.clear_data()
            for timestamp, voltage, current, ripple in self.records[-20000:]:
                self.plots.append(timestamp, (voltage, ripple, current), redraw=False)
            self.plots.redraw()
        self.plots.setVisible(visible)

    def _save(self) -> None:
        if not self.records:
            return
        path = save_file_name(self, "Save HV data", "hv_supply.csv", "CSV files (*.csv)")
        if not path:
            return

        def stamp(t: float) -> str:
            return dt.datetime.fromtimestamp(t).isoformat(timespec="milliseconds")

        write_csv(path, ["time", "voltage_kv", "current_ua", "ripple_v"],
                  [(stamp(t), v, i, r) for t, v, i, r in self.records])
        if self.breakdowns:
            breakdown_path = path[:-4] + "_breakdowns.csv" if path.lower().endswith(".csv") else path + "_breakdowns"
            write_csv(breakdown_path, ["time", "voltage_kv"], [(stamp(t), v) for t, v in self.breakdowns])
        self.log.info("HV data saved to %s", path)

    def _clear(self) -> None:
        self.records.clear()
        self.breakdowns.clear()
        self.breakdown_count.setText("0")
        self.plots.clear_data()

    def shutdown(self) -> None:
        self._stop_ramp()
        super().shutdown()
