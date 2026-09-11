"""Main window: live/pump-probe views, acquisition and scan controls, instrument panels, log.

Layout
------
* centre - tabs with the live detector image and the pump-probe results;
* left dock - detector controls ("Acquisition") and pump-probe scans ("Scan");
* right dock - one tab per instrument group of the configuration, holding
  the panels of its devices (and the RF interlock in the group of the amplifier);
* bottom dock - the application log.

The acquisition worker lives in its own thread; every instrument panel has
its own thread too (see :mod:`uedcontrol.gui.workers`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from .. import __version__
from ..devices.interfaces import Camera, RFAmplifier, RFPowerSensor
from ..instrument import Instrument
from .acquisition_panel import AcquisitionPanel
from .analysis_window import AnalysisWindow
from .log_panel import LogPanel, QtLogHandler
from .panels import DevicePanel, RFInterlockPanel, create_panel
from .pumpprobe_view import PumpProbeView
from .qt import QT_LIB, QtCore, QtGui, QtWidgets
from .scan_panel import ScanPanel
from .widgets.common import app_settings, ask, scroll_area
from .widgets.image_view import ImageView
from .workers import AcquisitionWorker

log = logging.getLogger(__name__)

DockArea = QtCore.Qt.DockWidgetArea


class MainWindow(QtWidgets.QMainWindow):
    acquisition_stop_timeout_ms = 15_000

    def __init__(self, instrument: Instrument, log_handler: QtLogHandler | None = None) -> None:
        super().__init__()
        self.instrument = instrument
        self.config = config = instrument.config
        self.settings = app_settings("main_window")
        self.analysis_window: AnalysisWindow | None = None
        self._shut_down = False
        self.setWindowTitle(f"UED control – {config.name}")

        # ------------------------------------------------ acquisition thread
        self.acquisition_thread = QtCore.QThread(self)
        self.acquisition_thread.setObjectName("acquisition")
        self.worker = AcquisitionWorker()
        self.worker.moveToThread(self.acquisition_thread)
        self.acquisition_thread.finished.connect(self.worker.deleteLater)
        self.acquisition_thread.start()

        # ------------------------------------------------------------ centre
        self.image_view = ImageView()
        self.pumpprobe_view = PumpProbeView()
        self.central_tabs = QtWidgets.QTabWidget()
        self.central_tabs.addTab(self.image_view, "Live view")
        self.central_tabs.addTab(self.pumpprobe_view, "Pump-probe")
        self.setCentralWidget(self.central_tabs)

        # ----------------------------------------------------- left: detector
        cameras = instrument.cameras()
        labels = {key: config.devices[key].label for key in cameras}
        self.acquisition_panel = AcquisitionPanel(cameras, labels, self.worker, self.image_view, config.beam)
        self.scan_panel = ScanPanel(self.worker, self.acquisition_panel, self.pumpprobe_view,
                                    instrument.delay_line, config)
        self.acquisition_panel.send_to_analysis.connect(self._send_to_analysis)
        self.scan_panel.scan_running.connect(self._scan_running)
        self.control_tabs = QtWidgets.QTabWidget()
        self.control_tabs.addTab(scroll_area(self.acquisition_panel), "Acquisition")
        self.control_tabs.addTab(scroll_area(self.scan_panel), "Scan")
        self.control_dock = self._dock("Detector and scans", "controls", self.control_tabs, DockArea.LeftDockWidgetArea)

        # ------------------------------------------------- right: instruments
        self.panels: list[DevicePanel] = []
        self.rf_panel: RFInterlockPanel | None = None
        self.instrument_tabs = QtWidgets.QTabWidget()
        self._build_instrument_tabs()
        self.instrument_dock = self._dock("Instruments", "instruments", self.instrument_tabs,
                                          DockArea.RightDockWidgetArea)

        # --------------------------------------------------------- bottom: log
        self.log_panel = LogPanel()
        if log_handler is not None:
            log_handler.emitter.message.connect(self.log_panel.append_message)
        self.log_dock = self._dock("Log", "log", self.log_panel, DockArea.BottomDockWidgetArea)

        self._build_menus()
        source = str(config.source) if config.source else "built-in simulated configuration"
        status = QtWidgets.QLabel(f"Configuration: {source}   ·   Data: {config.data_dir}")
        status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.statusBar().addPermanentWidget(status)

        self.resize(1700, 1000)
        geometry, state = self.settings.value("geometry"), self.settings.value("state")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state)
        QtCore.QTimer.singleShot(0, self._autoconnect)

    # ================================================================== layout
    def _dock(self, title: str, name: str, widget: QtWidgets.QWidget, area: DockArea) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(name)  # needed by saveState/restoreState
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_instrument_tabs(self) -> None:
        instrument, config = self.instrument, self.config
        groups: dict[str, list[QtWidgets.QWidget]] = {}
        rf = config.rf_interlock
        if rf is not None:
            amplifier = instrument.devices[rf.amplifier]
            reflected = instrument.devices[rf.reflected_sensor]
            forward = instrument.devices[rf.forward_sensor] if rf.forward_sensor else None
            assert isinstance(amplifier, RFAmplifier) and isinstance(reflected, RFPowerSensor)
            assert forward is None or isinstance(forward, RFPowerSensor)
            self.rf_panel = RFInterlockPanel(amplifier, reflected, forward, rf)
            groups.setdefault(config.devices[rf.amplifier].group, []).append(self.rf_panel)

        for name, device in instrument.devices.items():
            if isinstance(device, Camera):
                continue  # cameras are driven from the acquisition panel
            spec = config.devices[name]
            panel = create_panel(device, spec, delay_line=instrument.delay_line)
            self.panels.append(panel)
            groups.setdefault(spec.group, []).append(panel)

        if not groups:
            placeholder = QtWidgets.QLabel("No instruments in the configuration besides the detectors.")
            placeholder.setWordWrap(True)
            self.instrument_tabs.addTab(placeholder, "Instruments")
            return
        for group, widgets in groups.items():
            page = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page)
            for widget in widgets:
                layout.addWidget(widget)
            layout.addStretch(1)
            self.instrument_tabs.addTab(scroll_area(page), group)

    def _build_menus(self) -> None:
        menus = self.menuBar()

        file_menu = menus.addMenu("&File")
        self._action(file_menu, "&Save image…", self.acquisition_panel.save_snapshot, "Ctrl+S")
        self._action(file_menu, "Open &data folder", self._open_data_folder)
        file_menu.addSeparator()
        self._action(file_menu, "&Quit", self.close, "Ctrl+Q")

        instruments_menu = menus.addMenu("&Instruments")
        self._action(instruments_menu, "&Connect all instruments", self.connect_all)
        self._action(instruments_menu, "&Disconnect all instruments", self.disconnect_all)

        view_menu = menus.addMenu("&View")
        for dock in (self.control_dock, self.instrument_dock, self.log_dock):
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        self._action(view_menu, "&Live view", lambda: self.central_tabs.setCurrentWidget(self.image_view), "F2")
        self._action(view_menu, "&Pump-probe", lambda: self.central_tabs.setCurrentWidget(self.pumpprobe_view), "F3")

        tools_menu = menus.addMenu("&Tools")
        self._action(tools_menu, "Image &analysis…", self.show_analysis, "Ctrl+I")

        help_menu = menus.addMenu("&Help")
        self._action(help_menu, "&About", self._about)

    @staticmethod
    def _action(menu: QtWidgets.QMenu, text: str, slot: Callable[[], object], shortcut: str | None = None) -> None:
        action = menu.addAction(text)
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        action.triggered.connect(lambda _checked=False: slot())

    # ============================================================ connections
    def _autoconnect(self) -> None:
        for panel in self.panels:
            if panel.spec is not None and panel.spec.autoconnect:
                panel.connect_device()
        cameras = [key for key in self.instrument.cameras() if self.config.devices[key].autoconnect]
        if cameras:
            # Only one detector is active at a time; connecting several at once would race.
            self.acquisition_panel.connect_camera(cameras[0])
            if len(cameras) > 1:
                log.info("autoconnect: only the first detector (%s) is connected at start-up", cameras[0])

    def connect_all(self) -> None:
        for panel in self.panels:
            panel.connect_device()

    def disconnect_all(self) -> None:
        if self.scan_panel.running:
            log.warning("stop the scan before disconnecting the instruments")
            return
        for panel in self.panels:
            panel.disconnect_device()

    # ================================================================ actions
    def _scan_running(self, running: bool) -> None:
        if running:
            self.central_tabs.setCurrentWidget(self.pumpprobe_view)

    def show_analysis(self) -> AnalysisWindow:
        if self.analysis_window is None:
            self.analysis_window = AnalysisWindow(self.acquisition_panel.pixel_size_um())
        self.analysis_window.show()
        self.analysis_window.raise_()
        self.analysis_window.activateWindow()
        return self.analysis_window

    def _send_to_analysis(self, image: np.ndarray) -> None:
        camera = self.acquisition_panel.camera_name() or "detector"
        self.show_analysis().set_image(image, f"{camera} at {QtCore.QTime.currentTime().toString('HH:mm:ss')}",
                                       self.acquisition_panel.pixel_size_um())

    def _open_data_folder(self) -> None:
        directory = self.config.data_dir
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.error("cannot create %s: %s", directory, exc)
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(directory)))

    def _about(self) -> None:
        source = str(self.config.source) if self.config.source else "built-in simulated configuration"
        QtWidgets.QMessageBox.about(
            self,
            "About UED control",
            f"<b>uedcontrol {__version__}</b><br>Control software for the ultrafast electron diffraction "
            f"instrument.<br><br>Configuration: {source}<br>Devices: {len(self.instrument.devices)}<br>"
            f"Qt binding: {QT_LIB}",
        )

    # =============================================================== shutdown
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.scan_panel.running and not ask(self, "Quit", "A scan is running. Stop it and quit?"):
            event.ignore()
            return
        self.shutdown()
        event.accept()

    def shutdown(self) -> None:
        """Stop every thread and disconnect every instrument (idempotent)."""
        if self._shut_down:
            return
        self._shut_down = True
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("state", self.saveState())
        log.info("shutting down")
        # The acquisition thread first: it may be using a camera or the delay stage.
        self.worker.request_stop()
        self.acquisition_thread.quit()
        if not self.acquisition_thread.wait(self.acquisition_stop_timeout_ms):
            log.warning("the acquisition thread did not stop in time")
        self.acquisition_panel.shutdown()
        if self.rf_panel is not None:
            self.rf_panel.shutdown()
        for panel in self.panels:
            panel.shutdown()
        self.instrument.disconnect_all()
        if self.analysis_window is not None:
            self.analysis_window.close()
