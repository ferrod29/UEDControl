"""Panel for motion axes. For the pump-probe delay stage it also handles delays and T0."""

from __future__ import annotations

import math
import threading

from ...acquisition.delay import DelayLine
from ...devices.interfaces import Positioner
from .. import theme
from ..qt import QtWidgets
from ..widgets.common import Readout, spin
from .base import DevicePanel
from .generic import ActionsBox


class StagePanel(DevicePanel):
    default_poll_interval_s = 0.5

    def build(self, body: QtWidgets.QWidget) -> None:
        stage: Positioner = self.device  # type: ignore[assignment]
        self.delay_line: DelayLine | None = self.context.get("delay_line")
        unit = stage.unit
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)

        grid = QtWidgets.QGridLayout()
        self.position = Readout()
        self.moving = QtWidgets.QLabel()
        self.target = spin(0.0, -1e6, 1e6, 4, 1.0, unit)
        self.step = spin(1.0, 0.0, 1e6, 4, 0.1, unit)
        go = QtWidgets.QPushButton("Move")
        back = QtWidgets.QPushButton("◀")
        forward = QtWidgets.QPushButton("▶")
        home = QtWidgets.QPushButton("Home")
        stop = QtWidgets.QPushButton("STOP")
        stop.setStyleSheet(f"QPushButton {{ color: {theme.WARN_COLOR}; font-weight: bold; }}")
        self.limits_label = QtWidgets.QLabel("Limits: –")
        go.clicked.connect(self._move)
        back.clicked.connect(lambda: self._step(-1))
        forward.clicked.connect(lambda: self._step(+1))
        home.clicked.connect(lambda: self.run(stage.home, description=f"{self.device.name}: homing"))
        stop.clicked.connect(self._stop)
        grid.addWidget(QtWidgets.QLabel(f"Position ({unit})"), 0, 0)
        grid.addWidget(self.position, 0, 1)
        grid.addWidget(self.moving, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Move to"), 1, 0)
        grid.addWidget(self.target, 1, 1)
        grid.addWidget(go, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Step"), 2, 0)
        grid.addWidget(self.step, 2, 1)
        steps = QtWidgets.QHBoxLayout()
        steps.addWidget(back)
        steps.addWidget(forward)
        grid.addLayout(steps, 2, 2)
        grid.addWidget(home, 3, 1)
        grid.addWidget(stop, 3, 2)
        grid.addWidget(self.limits_label, 4, 0, 1, 3)
        layout.addLayout(grid)

        if self.delay_line is not None:
            box = QtWidgets.QGroupBox("Pump-probe delay")
            delay_grid = QtWidgets.QGridLayout(box)
            self.delay = Readout()
            self.t0 = spin(self.delay_line.t0_position, -1e6, 1e6, 4, 0.01, unit)
            self.t0.valueChanged.connect(self._t0_edited)
            set_t0 = QtWidgets.QPushButton("Set T0 here")
            go_t0 = QtWidgets.QPushButton("Go to T0")
            self.delay_target = spin(0.0, -1e5, 1e5, 3, 0.1, "ps")
            go_delay = QtWidgets.QPushButton("Move")
            self.delay_range = QtWidgets.QLabel()
            set_t0.clicked.connect(self._set_t0)
            go_t0.clicked.connect(lambda: self.run(stage.move_to, self.delay_line.t0_position, False,
                                                   description="moving to T0"))
            go_delay.clicked.connect(self._move_to_delay)
            delay_grid.addWidget(QtWidgets.QLabel("Delay (ps)"), 0, 0)
            delay_grid.addWidget(self.delay, 0, 1)
            delay_grid.addWidget(QtWidgets.QLabel("T0 position"), 1, 0)
            delay_grid.addWidget(self.t0, 1, 1)
            buttons = QtWidgets.QHBoxLayout()
            buttons.addWidget(set_t0)
            buttons.addWidget(go_t0)
            delay_grid.addLayout(buttons, 1, 2)
            delay_grid.addWidget(QtWidgets.QLabel("Move to delay"), 2, 0)
            delay_grid.addWidget(self.delay_target, 2, 1)
            delay_grid.addWidget(go_delay, 2, 2)
            delay_grid.addWidget(self.delay_range, 3, 0, 1, 3)
            layout.addWidget(box)

        self.actions_box = ActionsBox(self.run, self.device.name)
        layout.addWidget(self.actions_box)

    def on_connected(self) -> None:
        self.actions_box.populate(self.device.actions())
        self.controller.submit(self.device.limits, done=self._show_limits)

    def _show_limits(self, limits: tuple[float, float]) -> None:
        low, high = limits
        if math.isfinite(low) and math.isfinite(high):
            self.target.setRange(low, high)
            self.limits_label.setText(f"Limits: {low:g} … {high:g} {self.device.unit}")
            if self.delay_line is not None:
                a, b = self.delay_line.delay_for(low), self.delay_line.delay_for(high)
                self.delay_range.setText(f"Delay range: {min(a, b):.1f} … {max(a, b):.1f} ps")

    def _move(self) -> None:
        target = self.target.value()
        self.run(self.device.move_to, target, False, description=f"{self.device.name}: move to {target:g}")

    def _step(self, sign: int) -> None:
        delta = sign * self.step.value()
        self.run(self.device.move_by, delta, False, description=f"{self.device.name}: move by {delta:+g}")

    def _stop(self) -> None:
        # Not through the controller: it may be busy with a blocking move or homing.
        threading.Thread(target=self._stop_now, name=f"stop-{self.device.name}", daemon=True).start()

    def _stop_now(self) -> None:
        try:
            self.device.stop()
            self.log.info("%s stopped", self.device.name)
        except Exception as exc:
            self.log.error("stop failed: %s", exc)

    def poll(self) -> tuple[float, bool]:
        stage: Positioner = self.device  # type: ignore[assignment]
        with stage.lock:
            return stage.position(), stage.is_moving()

    def on_status(self, status: tuple[float, bool]) -> None:
        position, moving = status
        self.position.show_value(position, "{:.4f}")
        self.moving.setText("moving" if moving else "")
        self.moving.setStyleSheet(f"color: {theme.OFF_COLOR};" if moving else "")
        if self.delay_line is not None:
            self.delay.show_value(self.delay_line.delay_for(position), "{:.3f}")

    # ------------------------------------------------------------------ delay
    def _set_t0(self) -> None:
        self.controller.submit(self.delay_line.set_t0_here, done=self._t0_set)

    def _t0_set(self, t0: float) -> None:
        self.t0.blockSignals(True)
        self.t0.setValue(t0)
        self.t0.blockSignals(False)
        self.log.info("T0 set at %.4f %s", t0, self.device.unit)

    def _t0_edited(self, value: float) -> None:
        self.delay_line.t0_position = value
        self.log.info("T0 position changed to %.4f %s", value, self.device.unit)

    def _move_to_delay(self) -> None:
        delay = self.delay_target.value()
        self.run(self.delay_line.move_to_delay, delay, False, description=f"moving to {delay:g} ps")
