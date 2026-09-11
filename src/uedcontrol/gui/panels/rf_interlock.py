"""RF power interlock ("Cavity Power Interlock" of the original software).

Watches the reflected power of the compression cavity and switches the RF
amplifier off when it exceeds the limit. The individual RF instruments keep
their own panels; this one only supervises them.
"""

from __future__ import annotations

import logging

from ...acquisition.safety import PowerInterlock
from ...config import RFInterlockConfig
from ...devices.interfaces import RFAmplifier, RFPowerSensor
from .. import theme
from ..qt import QtWidgets
from ..widgets.common import Readout, ToggleButton, spin
from ..workers import DeviceController


class RFInterlockPanel(QtWidgets.QGroupBox):
    poll_interval_s = 0.5

    def __init__(
        self,
        amplifier: RFAmplifier,
        reflected: RFPowerSensor,
        forward: RFPowerSensor | None,
        config: RFInterlockConfig,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__("RF power interlock", parent)
        self.amplifier = amplifier
        self.reflected = reflected
        self.forward = forward
        self.interlock = PowerInterlock(config.limit_dbm)
        self.log = logging.getLogger("uedcontrol.gui.rf_interlock")
        self._alarm: QtWidgets.QMessageBox | None = None

        layout = QtWidgets.QGridLayout(self)
        self.forward_dbm, self.forward_w = Readout(), Readout()
        self.reflected_dbm, self.reflected_w = Readout(), Readout()
        self.limit = spin(config.limit_dbm, -100.0, 60.0, 2, 0.5, "dBm")
        self.limit.valueChanged.connect(self._limit_changed)
        self.arm_button = ToggleButton("Interlock ARMED", "Interlock OFF")
        self.arm_button.clicked.connect(self._toggle_armed)
        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.clicked.connect(self._reset)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(QtWidgets.QLabel(""), 0, 0)
        layout.addWidget(QtWidgets.QLabel("dBm"), 0, 1)
        layout.addWidget(QtWidgets.QLabel("W (line)"), 0, 2)
        layout.addWidget(QtWidgets.QLabel("Forward"), 1, 0)
        layout.addWidget(self.forward_dbm, 1, 1)
        layout.addWidget(self.forward_w, 1, 2)
        layout.addWidget(QtWidgets.QLabel("Reflected"), 2, 0)
        layout.addWidget(self.reflected_dbm, 2, 1)
        layout.addWidget(self.reflected_w, 2, 2)
        layout.addWidget(QtWidgets.QLabel("Reflected power limit"), 3, 0)
        layout.addWidget(self.limit, 3, 1, 1, 2)
        layout.addWidget(self.arm_button, 4, 0, 1, 2)
        layout.addWidget(reset_button, 4, 2)
        layout.addWidget(self.status, 5, 0, 1, 3)

        self.controller = DeviceController("rf-interlock", self)
        self.controller.status.connect(self._on_status)
        self.controller.poll_error.connect(lambda message: self._set_status(f"⚠ {message}", warning=True))
        self.controller.start_polling(self._poll, self.poll_interval_s)

    def _poll(self) -> tuple[float | None, float] | None:
        if not self.reflected.connected:
            return None
        forward = self.forward.power_dbm() if self.forward is not None and self.forward.connected else None
        return forward, self.reflected.power_dbm()

    def _on_status(self, result: tuple[float | None, float] | None) -> None:
        if result is None:
            self._set_status("Reflected-power sensor not connected: the interlock is inactive.", warning=True)
            return
        forward, reflected = result
        self.forward_dbm.show_value(forward, "{:.2f}")
        self.forward_w.show_value(None if forward is None else self.forward.line_power_w(forward), "{:.4g}")
        self.reflected_dbm.show_value(reflected, "{:.2f}")
        self.reflected_w.show_value(self.reflected.line_power_w(reflected), "{:.4g}")
        if self.interlock.check(reflected):
            self._trip(reflected)
        elif not self.interlock.tripped:
            self._set_status("Armed" if self.interlock.armed else "Not armed", warning=False)

    def _trip(self, reflected_dbm: float) -> None:
        amplifier = self.amplifier

        def switch_off() -> None:
            if amplifier.connected:
                amplifier.set_rf_enabled(False)

        self.controller.submit(switch_off,
                               failed=lambda message: self.log.error("could not switch RF off: %s", message))
        message = (f"Reflected power {reflected_dbm:.2f} dBm exceeded the limit of {self.interlock.limit:.2f} dBm: "
                   "the RF amplifier has been switched off.")
        self.log.error(message)
        self._set_status("TRIPPED: " + message, warning=True)
        self.arm_button.setText("Interlock TRIPPED")
        self._alarm = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Warning, "RF power interlock", message,
                                            QtWidgets.QMessageBox.StandardButton.Ok, self)
        self._alarm.setModal(False)
        self._alarm.show()

    def _toggle_armed(self, checked: bool) -> None:
        if checked:
            self.interlock.arm()
            self.log.info("RF interlock armed at %.2f dBm", self.interlock.limit)
        else:
            self.interlock.disarm()
            self.log.info("RF interlock disarmed")

    def _reset(self) -> None:
        self.interlock.reset()
        self.arm_button.setChecked(self.interlock.armed)
        self.arm_button.setText("Interlock ARMED" if self.interlock.armed else "Interlock OFF")
        self._set_status("Reset", warning=False)

    def _limit_changed(self, value: float) -> None:
        self.interlock.limit = value

    def _set_status(self, text: str, warning: bool) -> None:
        self.status.setStyleSheet(f"color: {theme.WARN_COLOR};" if warning else "")
        self.status.setText(text)

    def shutdown(self) -> None:
        self.controller.shutdown()
