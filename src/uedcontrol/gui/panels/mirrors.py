"""Panel for the piezo steering mirrors (Newport Agilis): channel, step size and keypad.

In step mode every press moves by the given number of steps (a press while
the axis still moves stops it, as before). In continuous mode the mirror
jogs while the button is held down.
"""

from __future__ import annotations

from ...devices.interfaces import PiezoMirrorController
from .. import theme
from ..qt import QtWidgets
from ..widgets.common import int_spin
from .base import DevicePanel


class MirrorsPanel(DevicePanel):
    def build(self, body: QtWidgets.QWidget) -> None:
        mirrors: PiezoMirrorController = self.device  # type: ignore[assignment]
        names = list(self.options.get("channel_names") or [])
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)

        channels = QtWidgets.QHBoxLayout()
        self.channel_buttons: list[QtWidgets.QRadioButton] = []
        for channel in range(1, mirrors.n_channels + 1):
            radio = QtWidgets.QRadioButton(names[channel - 1] if channel <= len(names) else f"Channel {channel}")
            radio.toggled.connect(lambda on, ch=channel, r=radio: on and self.run(
                mirrors.select_channel, ch, description=f"mirror {r.text()} selected"))
            channels.addWidget(radio)
            self.channel_buttons.append(radio)
        layout.addLayout(channels)

        form = QtWidgets.QGridLayout()
        self.amplitude = int_spin(16, 1, 50)
        set_amplitude = QtWidgets.QPushButton("Set")
        set_amplitude.clicked.connect(lambda: self.run(
            mirrors.set_step_amplitude, self.amplitude.value(),
            description=f"mirror step amplitude -> {self.amplitude.value()}"))
        self.steps = int_spin(10, 1, 100_000)
        self.jog_speed = QtWidgets.QComboBox()
        for speed, label in mirrors.jog_speeds or ((1, "1"),):
            self.jog_speed.addItem(label, speed)
        self.continuous = QtWidgets.QCheckBox("Continuous (hold to move)")
        form.addWidget(QtWidgets.QLabel("Step amplitude"), 0, 0)
        form.addWidget(self.amplitude, 0, 1)
        form.addWidget(set_amplitude, 0, 2)
        form.addWidget(QtWidgets.QLabel("Steps per press"), 1, 0)
        form.addWidget(self.steps, 1, 1)
        form.addWidget(QtWidgets.QLabel("Jog speed"), 2, 0)
        form.addWidget(self.jog_speed, 2, 1)
        form.addWidget(self.continuous, 3, 0, 1, 3)
        layout.addLayout(form)

        pad = QtWidgets.QGridLayout()
        for text, (row, column), (axis, sign) in (
            ("▲", (0, 1), (2, +1)),
            ("◀", (1, 0), (1, -1)),
            ("▶", (1, 2), (1, +1)),
            ("▼", (2, 1), (2, -1)),
        ):
            button = QtWidgets.QPushButton(text)
            button.setMinimumSize(48, 36)
            button.pressed.connect(lambda a=axis, s=sign: self._pressed(a, s))
            button.released.connect(lambda a=axis: self._released(a))
            pad.addWidget(button, row, column)
        stop = QtWidgets.QPushButton("STOP")
        stop.setStyleSheet(f"QPushButton {{ color: {theme.WARN_COLOR}; font-weight: bold; }}")
        stop.clicked.connect(lambda: self.run(mirrors.stop, description="mirrors stopped"))
        pad.addWidget(stop, 1, 1)
        layout.addLayout(pad)
        self.status_label = QtWidgets.QLabel()
        layout.addWidget(self.status_label)

    def _pressed(self, axis: int, sign: int) -> None:
        mirrors: PiezoMirrorController = self.device  # type: ignore[assignment]
        if self.continuous.isChecked():
            self.run(mirrors.jog, axis, sign * int(self.jog_speed.currentData()))
        else:
            self.run(self._step, axis, sign * self.steps.value())

    def _step(self, axis: int, steps: int) -> None:
        mirrors: PiezoMirrorController = self.device  # type: ignore[assignment]
        with mirrors.lock:
            if mirrors.is_moving(axis):
                mirrors.stop(axis)
            else:
                mirrors.step(axis, steps)

    def _released(self, axis: int) -> None:
        if self.continuous.isChecked():
            self.run(self.device.stop, axis)

    def poll(self) -> dict[int, bool]:
        mirrors: PiezoMirrorController = self.device  # type: ignore[assignment]
        with mirrors.lock:
            return {axis: mirrors.is_moving(axis) for axis in range(1, mirrors.n_axes + 1)}

    def on_status(self, moving: dict[int, bool]) -> None:
        self.status_label.setText("   ".join(f"Axis {axis}: {'moving' if m else 'idle'}" for axis, m in moving.items()))
