"""Generic panel built from what a driver declares: readings, settings and commands.

Any instrument without a dedicated panel (vacuum gauges, turbo pumps, power
meters, chillers, RF parts, the delay generator...) gets this one, so a new
driver only has to implement ``read_status``, ``settings`` and ``actions``.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

from ...devices.base import Action, Reading, Setting
from ..qt import QtWidgets
from ..widgets.common import Readout, ask, clear_layout, int_spin, save_file_name, spin, write_csv
from ..widgets.time_plot import TimeSeriesPlot
from .base import DevicePanel

Runner = Callable[..., None]


class SettingsBox(QtWidgets.QGroupBox):
    """Editable fields for a list of :class:`Setting`, applied through ``run``."""

    def __init__(self, run: Runner, owner: str, title: str = "Settings") -> None:
        super().__init__(title)
        self._run = run
        self._owner = owner
        self.form = QtWidgets.QFormLayout(self)
        self.hide()

    def populate(self, settings: Sequence[Setting]) -> None:
        clear_layout(self.form)
        for setting in settings:
            self.form.addRow(setting.label, self._editor(setting))
        self.setVisible(bool(settings))

    def _apply(self, setting: Setting, value: Any) -> None:
        self._run(setting.apply, value, description=f"{self._owner}: {setting.label} -> {value}")

    def _editor(self, setting: Setting) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        if setting.kind == "choice":
            box = QtWidgets.QComboBox()
            box.addItems([str(choice) for choice in setting.choices])
            if setting.initial is not None:
                box.setCurrentText(str(setting.initial))
            box.activated.connect(lambda _index, s=setting, b=box: self._apply(s, b.currentText()))
            row.addWidget(box)
            return container
        if setting.kind == "int":
            editor = int_spin(int(setting.initial or 0), int(setting.minimum), int(setting.maximum),
                              suffix=setting.unit)
            value = editor.value
        elif setting.kind == "text":
            editor = QtWidgets.QLineEdit("" if setting.initial is None else str(setting.initial))
            value = editor.text
        else:
            initial = float(setting.initial) if setting.initial is not None else max(setting.minimum, 0.0)
            editor = spin(initial, setting.minimum, setting.maximum, setting.decimals, suffix=setting.unit)
            value = editor.value
        button = QtWidgets.QPushButton("Set")
        button.clicked.connect(lambda _checked=False, s=setting, get=value: self._apply(s, get()))
        row.addWidget(editor, 1)
        row.addWidget(button)
        return container


class ActionsBox(QtWidgets.QGroupBox):
    """Command buttons for a list of :class:`Action`."""

    def __init__(self, run: Runner, owner: str, title: str = "Commands", columns: int = 3) -> None:
        super().__init__(title)
        self._run = run
        self._owner = owner
        self._columns = columns
        self.grid = QtWidgets.QGridLayout(self)
        self.hide()

    def populate(self, actions: Sequence[Action]) -> None:
        clear_layout(self.grid)
        for index, action in enumerate(actions):
            button = QtWidgets.QPushButton(action.label)
            button.clicked.connect(lambda _checked=False, a=action: self._trigger(a))
            self.grid.addWidget(button, index // self._columns, index % self._columns)
        self.setVisible(bool(actions))

    def _trigger(self, action: Action) -> None:
        label = action.label.replace("&&", "&")
        if action.confirm and not ask(self, label, action.confirm):
            return
        self._run(action.callback, description=f"{self._owner}: {label}")


class GenericPanel(DevicePanel):
    history_limit = 200_000

    def build(self, body: QtWidgets.QWidget) -> None:
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        self.readings = QtWidgets.QFormLayout()
        layout.addLayout(self.readings)
        self._readouts: dict[str, Readout] = {}
        self.settings_box = SettingsBox(self.run, self.device.name)
        self.actions_box = ActionsBox(self.run, self.device.name)
        layout.addWidget(self.settings_box)
        layout.addWidget(self.actions_box)

        buttons = QtWidgets.QHBoxLayout()
        plot_button = QtWidgets.QPushButton("Plot")
        save_button = QtWidgets.QPushButton("Save readings…")
        clear_button = QtWidgets.QPushButton("Clear readings")
        plot_button.clicked.connect(self._show_plot)
        save_button.clicked.connect(self._save)
        clear_button.clicked.connect(self._clear)
        for button in (plot_button, save_button, clear_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.history: list[tuple[float, dict[str, float]]] = []
        self._units: dict[str, str] = {}
        self._plot_keys: list[str] = []
        self.plot_window: TimeSeriesPlot | None = None

    def on_connected(self) -> None:
        self.actions_box.populate(self.device.actions())
        self.controller.submit(self.device.settings, done=self.settings_box.populate)

    def on_status(self, status: dict[str, Reading]) -> None:
        numeric: dict[str, float] = {}
        for key, reading in status.items():
            readout = self._readouts.get(key)
            if readout is None:
                readout = Readout()
                self._readouts[key] = readout
                self.readings.addRow(key, readout)
            readout.setText(reading.formatted())
            if isinstance(reading.value, (int, float)) and not isinstance(reading.value, bool):
                numeric[key] = float(reading.value)
                self._units[key] = reading.unit
        if not numeric:
            return
        now = time.time()
        self.history.append((now, numeric))
        if len(self.history) > self.history_limit:
            del self.history[: self.history_limit // 10]
        if self.plot_window is not None and self.plot_window.isVisible():
            self.plot_window.append(now, [numeric.get(key, math.nan) for key in self._plot_keys])

    def _show_plot(self) -> None:
        if not self.history:
            return
        self._plot_keys = list(self.history[-1][1])
        self.plot_window = TimeSeriesPlot([(key, self._units.get(key, "")) for key in self._plot_keys])
        self.plot_window.setWindowTitle(self.title())
        for timestamp, values in self.history:
            self.plot_window.append(timestamp, [values.get(key, math.nan) for key in self._plot_keys], redraw=False)
        self.plot_window.redraw()
        self.plot_window.resize(800, 160 * len(self._plot_keys) + 80)
        self.plot_window.show()

    def _save(self) -> None:
        if not self.history:
            return
        path = save_file_name(self, "Save readings", f"{self.device.name}.csv", "CSV files (*.csv)")
        if not path:
            return
        keys = sorted({key for _t, values in self.history for key in values})
        header = ["time"] + [f"{key} ({self._units.get(key, '')})".replace(" ()", "") for key in keys]
        rows = [
            [dt.datetime.fromtimestamp(t).isoformat(timespec="milliseconds")] + [values.get(key, "") for key in keys]
            for t, values in self.history
        ]
        write_csv(path, header, rows)
        self.log.info("readings saved to %s", path)

    def _clear(self) -> None:
        self.history.clear()
        if self.plot_window is not None:
            self.plot_window.clear_data()
