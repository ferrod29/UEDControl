"""Panel for (multi-channel) DC supplies, e.g. the magnetic lens supplies.

Panel options (``panel:`` in the configuration):

``channel_names``     list of channel labels
``coil_temperature``  ``{channel: {slope, offset}}`` - coil temperature from its
                      resistance, T = slope * V/I + offset (the HMP4040 lens
                      calibration of the original software: 6.8301 R - 113.16)
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass

from ...devices.interfaces import PowerSupply
from ..qt import QtWidgets
from ..widgets.common import Readout, ToggleButton, save_file_name, spin, write_csv
from .base import DevicePanel


@dataclass
class _ChannelRow:
    name: QtWidgets.QLabel
    voltage: QtWidgets.QDoubleSpinBox
    current: QtWidgets.QDoubleSpinBox
    apply: QtWidgets.QPushButton
    output: ToggleButton
    measured_voltage: Readout
    measured_current: Readout
    coil: Readout | None


class PowerSupplyPanel(DevicePanel):
    history_limit = 200_000

    def build(self, body: QtWidgets.QWidget) -> None:
        supply: PowerSupply = self.device  # type: ignore[assignment]
        names = list(self.options.get("channel_names") or [])
        self.coil = {
            int(channel): (float(cal["slope"]), float(cal["offset"]))
            for channel, cal in (self.options.get("coil_temperature") or {}).items()
        }
        grid = QtWidgets.QGridLayout(body)
        headers = ["", f"Set ({supply.voltage_unit})", f"Limit ({supply.current_unit})", "", "Output",
                   f"V ({supply.voltage_unit})", f"I ({supply.current_unit})"]
        if self.coil:
            headers.append("Coil (°C)")
        for column, text in enumerate(headers):
            grid.addWidget(QtWidgets.QLabel(text), 0, column)

        self.rows: list[_ChannelRow] = []
        for channel in range(1, supply.n_channels + 1):
            row = _ChannelRow(
                name=QtWidgets.QLabel(names[channel - 1] if channel <= len(names) else f"CH{channel}"),
                voltage=spin(0.0, 0.0, 1e4, 3, 0.1),
                current=spin(0.0, 0.0, 1e3, 4, 0.01),
                apply=QtWidgets.QPushButton("Apply"),
                output=ToggleButton("ON", "OFF"),
                measured_voltage=Readout(),
                measured_current=Readout(),
                coil=Readout() if channel in self.coil else None,
            )
            widgets = [row.name, row.voltage, row.current, row.apply, row.output, row.measured_voltage,
                       row.measured_current]
            if self.coil:
                widgets.append(row.coil or QtWidgets.QLabel(""))
            for column, widget in enumerate(widgets):
                grid.addWidget(widget, channel, column)
            row.apply.clicked.connect(lambda _checked=False, ch=channel, r=row: self._apply(ch, r))
            row.output.clicked.connect(lambda checked, ch=channel, r=row: self.run(
                supply.set_output, checked, ch,
                description=f"{self.device.name} {r.name.text()} output {'on' if checked else 'off'}"))
            self.rows.append(row)

        last = supply.n_channels + 1
        self.master: ToggleButton | None = None
        if supply.has_master_output:
            self.master = ToggleButton("Master output ON", "Master output OFF")
            self.master.clicked.connect(lambda checked: self.run(
                supply.set_master_output, checked,
                description=f"{self.device.name} master output {'on' if checked else 'off'}"))
            grid.addWidget(self.master, last, 0, 1, 3)
        save_button = QtWidgets.QPushButton("Save log…")
        save_button.clicked.connect(self._save)
        grid.addWidget(save_button, last, len(headers) - 2, 1, 2)
        self.history: list[list[float]] = []

    def _apply(self, channel: int, row: _ChannelRow) -> None:
        supply: PowerSupply = self.device  # type: ignore[assignment]
        voltage, current = row.voltage.value(), row.current.value()

        def apply() -> None:
            with supply.lock:
                supply.set_current(current, channel)
                supply.set_voltage(voltage, channel)

        self.run(apply, description=f"{self.device.name} {row.name.text()}: {voltage:g} {supply.voltage_unit}, "
                                    f"limit {current:g} {supply.current_unit}")

    def on_connected(self) -> None:
        self.controller.submit(self._read_setpoints, done=self._show_setpoints)

    def _read_setpoints(self) -> list[tuple[float | None, float | None]]:
        supply: PowerSupply = self.device  # type: ignore[assignment]
        with supply.lock:
            return [(supply.voltage_setpoint(ch), supply.current_setpoint(ch))
                    for ch in range(1, supply.n_channels + 1)]

    def _show_setpoints(self, values: list[tuple[float | None, float | None]]) -> None:
        for row, (voltage, current) in zip(self.rows, values, strict=True):
            if voltage is not None:
                row.voltage.setValue(voltage)
            if current is not None:
                row.current.setValue(current)

    def poll(self) -> tuple[list[tuple[float, float, bool]], bool | None]:
        supply: PowerSupply = self.device  # type: ignore[assignment]
        with supply.lock:
            channels = [
                (supply.measure_voltage(ch), supply.measure_current(ch), supply.output_enabled(ch))
                for ch in range(1, supply.n_channels + 1)
            ]
            master = supply.master_output_enabled() if supply.has_master_output else None
        return channels, master

    def on_status(self, status: tuple[list[tuple[float, float, bool]], bool | None]) -> None:
        channels, master = status
        record: list[float] = [time.time()]
        for channel, (row, (voltage, current, enabled)) in enumerate(zip(self.rows, channels, strict=True), start=1):
            row.measured_voltage.show_value(voltage, "{:.3f}")
            row.measured_current.show_value(current, "{:.4f}")
            row.output.setChecked(enabled)
            if row.coil is not None:
                slope, offset = self.coil[channel]
                row.coil.show_value(slope * voltage / current + offset if abs(current) > 1e-6 else None, "{:.2f}")
            record += [voltage, current, float(enabled)]
        if master is not None and self.master is not None:
            self.master.setChecked(master)
        self.history.append(record)
        if len(self.history) > self.history_limit:  # the program may run for weeks
            del self.history[: self.history_limit // 10]

    def _save(self) -> None:
        if not self.history:
            return
        path = save_file_name(self, "Save log", f"{self.device.name}.csv", "CSV files (*.csv)")
        if not path:
            return
        supply: PowerSupply = self.device  # type: ignore[assignment]
        header = ["time"]
        for row in self.rows:
            name = row.name.text()
            header += [f"{name} V ({supply.voltage_unit})", f"{name} I ({supply.current_unit})", f"{name} output"]
        rows = [[dt.datetime.fromtimestamp(r[0]).isoformat(timespec="milliseconds"), *r[1:]] for r in self.history]
        write_csv(path, header, rows)
        self.log.info("log saved to %s", path)
