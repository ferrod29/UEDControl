"""Base class of the instrument panels."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ...config import DeviceConfig
from ...devices.base import Device
from .. import theme
from ..qt import QtWidgets
from ..widgets.common import ToggleButton
from ..workers import DeviceController


class DevicePanel(QtWidgets.QGroupBox):
    """A framed panel for one instrument.

    Provides the connect button, an identification line, a status line and a
    body that is enabled only while the instrument is connected. Every call
    to the instrument goes through :attr:`controller` (its own thread).
    Subclasses implement :meth:`build`, and usually :meth:`poll` and
    :meth:`on_status`.
    """

    default_poll_interval_s = 1.0

    def __init__(
        self,
        device: Device,
        spec: DeviceConfig | None = None,
        parent: QtWidgets.QWidget | None = None,
        **context: Any,
    ) -> None:
        super().__init__(spec.label if spec is not None else device.name, parent)
        self.device = device
        self.spec = spec
        self.options: dict[str, Any] = dict(spec.panel) if spec is not None else {}
        self.context = context
        self.log = logging.getLogger(f"uedcontrol.gui.{device.name}")
        self.poll_interval_s = float(self.options.get("poll_interval_s", self.default_poll_interval_s))

        self.controller = DeviceController(device.name, self)
        self.controller.status.connect(self._on_status)
        self.controller.poll_error.connect(self._on_poll_error)
        self.controller.error.connect(self.show_error)

        self.connect_button = ToggleButton("Connected", "Connect")
        self.connect_button.clicked.connect(self._toggle_connection)
        self.identity = QtWidgets.QLabel(device.model)
        self.identity.setWordWrap(True)
        self.status_line = QtWidgets.QLabel()
        self.status_line.setWordWrap(True)
        self.body = QtWidgets.QWidget()
        self.body.setEnabled(False)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(self.connect_button)
        header.addWidget(self.identity, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.body)
        layout.addWidget(self.status_line)
        self.build(self.body)

    # ------------------------------------------------------------------ hooks
    def build(self, body: QtWidgets.QWidget) -> None:
        """Create the widgets of the panel body."""

    def poll(self) -> Any:
        """Runs in the device thread every poll interval; the result goes to :meth:`on_status`."""
        return self.device.read_status()

    def on_status(self, status: Any) -> None:
        """Show a poll result (GUI thread)."""

    def on_connected(self) -> None:
        """Called in the GUI thread once the instrument is connected."""

    def on_disconnected(self) -> None:
        """Called in the GUI thread once the instrument is disconnected."""

    # ---------------------------------------------------------------- helpers
    def run(
        self,
        function: Callable[..., Any],
        *args: Any,
        done: Callable[[Any], None] | None = None,
        description: str | None = None,
    ) -> None:
        """Call ``function(*args)`` in the device thread."""
        if description:
            self.log.info(description)
        self.controller.submit(lambda: function(*args), done=done)

    def show_error(self, message: str) -> None:
        self.status_line.setStyleSheet(f"color: {theme.WARN_COLOR};")
        self.status_line.setText(message)

    def clear_status(self) -> None:
        self.status_line.setStyleSheet("")
        self.status_line.clear()

    # ------------------------------------------------------------- connection
    def connect_device(self) -> None:
        if not self.device.connected:
            self.connect_button.setChecked(True)
            self._toggle_connection(True)

    def disconnect_device(self) -> None:
        if self.device.connected:
            self.connect_button.setChecked(False)
            self._toggle_connection(False)

    def _toggle_connection(self, checked: bool) -> None:
        self.connect_button.setEnabled(False)
        if checked:
            self.status_line.setStyleSheet("")
            self.status_line.setText("Connecting…")
            self.controller.submit(self._connect_in_thread, done=self._connected, failed=self._connection_failed)
        else:
            self.controller.stop_polling()
            self.controller.submit(self.device.disconnect, done=lambda _result: self._disconnected(),
                                   failed=self._connection_failed)

    def _connect_in_thread(self) -> str:
        self.device.connect()
        try:
            return self.device.identify()
        except Exception as exc:
            self.log.warning("identification failed: %s", exc)
            return self.device.model

    def _connected(self, identity: str) -> None:
        self.connect_button.setEnabled(True)
        self.connect_button.setChecked(True)
        self.identity.setText(str(identity))
        self.body.setEnabled(True)
        self.clear_status()
        self.on_connected()
        if self.poll_interval_s > 0:
            self.controller.start_polling(self.poll, self.poll_interval_s)

    def _connection_failed(self, message: str) -> None:
        self.connect_button.setEnabled(True)
        self.connect_button.setChecked(self.device.connected)
        self.body.setEnabled(self.device.connected)
        self.show_error(message)

    def _disconnected(self) -> None:
        self.connect_button.setEnabled(True)
        self.connect_button.setChecked(False)
        self.body.setEnabled(False)
        self.clear_status()
        self.on_disconnected()

    def _on_status(self, status: Any) -> None:
        if self.status_line.text().startswith("⚠"):
            self.clear_status()
        self.on_status(status)

    def _on_poll_error(self, message: str) -> None:
        self.show_error(f"⚠ {message}")

    def shutdown(self) -> None:
        self.controller.shutdown()
        if self.device.connected:
            try:
                self.device.disconnect()
            except Exception:
                self.log.exception("disconnect failed")
