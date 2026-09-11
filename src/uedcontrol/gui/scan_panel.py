"""Pump-probe scan controls: delays, averaging, runs, storage, start/stop and progress.

Two modes, as in the original software:

* **Delay scan** - the delay stage visits every delay of the list, run after
  run (alternating direction when *bidirectional* is on).
* **Time series** - no stage motion; the points are just consecutive
  datasets (the old "continuous" mode), e.g. to monitor the beam.

The scan itself runs in the acquisition thread (:meth:`AcquisitionWorker.run_scan`);
this panel only prepares :class:`ScanSettings` and a :class:`ScanWriter` and
shows the progress.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..acquisition.delay import DelayLine, delays_from_file, delays_from_range, delays_from_segments, parse_segments
from ..acquisition.scan import ScanSettings
from ..acquisition.storage import ScanWriter
from ..config import InstrumentConfig
from . import theme
from .acquisition_panel import AcquisitionPanel
from .pumpprobe_view import PumpProbeView
from .qt import QtCore, QtGui, QtWidgets, Signal
from .widgets.common import ToggleButton, int_spin, open_file_name, spin
from .workers import AcquisitionWorker, ScanProgress

_UNSAFE_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def safe_name(name: str) -> str:
    """A scan name usable as a folder and file name on every platform."""
    cleaned = _UNSAFE_CHARACTERS.sub("_", name).strip(" ._")
    return cleaned or "scan"


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class ScanPanel(QtWidgets.QWidget):
    scan_running = Signal(bool)
    _request_scan = Signal(object, object, object)

    def __init__(
        self,
        worker: AcquisitionWorker,
        acquisition: AcquisitionPanel,
        view: PumpProbeView,
        delay_line: DelayLine | None,
        config: InstrumentConfig,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.worker = worker
        self.acquisition = acquisition
        self.view = view
        self.delay_line = delay_line
        self.config = config
        self.log = logging.getLogger("uedcontrol.gui.scan")
        self.data_dir = Path(config.data_dir)
        self.file_delays: np.ndarray | None = None
        self.running = False
        self.last_scan_dir: Path | None = None
        self._writer: ScanWriter | None = None
        self._time_series = False
        self._total_points: int | None = None
        self._started = 0.0

        self._build()
        self._request_scan.connect(worker.run_scan)
        worker.scan_progress.connect(self._on_progress)
        worker.scan_finished.connect(self._on_finished)
        acquisition.frame_rate_changed.connect(lambda _fps: self._update_preview())
        self._mode_changed()

    # ================================================================== layout
    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # -------------------------------------------------------------- mode
        mode_box = QtWidgets.QGroupBox("Mode")
        mode_row = QtWidgets.QHBoxLayout(mode_box)
        self.delay_mode = QtWidgets.QRadioButton("Delay scan")
        self.series_mode = QtWidgets.QRadioButton("Time series")
        self.series_mode.setToolTip("No stage motion: consecutive datasets (the former 'continuous' mode)")
        if self.delay_line is None:
            self.delay_mode.setEnabled(False)
            self.delay_mode.setToolTip("No delay stage configured (delay_line.stage in the configuration)")
            self.series_mode.setChecked(True)
        else:
            self.delay_mode.setChecked(True)
        for button in (self.delay_mode, self.series_mode):
            button.toggled.connect(lambda _on: self._mode_changed())
            mode_row.addWidget(button)
        layout.addWidget(mode_box)

        # ------------------------------------------------------------ delays
        self.delay_box = QtWidgets.QGroupBox("Delays")
        grid = QtWidgets.QGridLayout(self.delay_box)
        self.start = spin(-5.0, -1e5, 1e5, 3, 0.5, "ps")
        self.stop = spin(20.0, -1e5, 1e5, 3, 0.5, "ps")
        self.step = spin(0.5, -1e4, 1e4, 3, 0.1, "ps")
        self.extra = QtWidgets.QLineEdit()
        self.extra.setPlaceholderText("more segments, e.g. -1:1:0.1; 20:100:5")
        self.extra.setToolTip("start:stop:step segments separated by ';'. They are merged with the range above.")
        self.load_button = QtWidgets.QPushButton("Load list…")
        self.load_button.setToolTip("Text file with one delay (ps) per line; replaces the range")
        self.clear_file_button = QtWidgets.QPushButton("Use range")
        self.clear_file_button.setEnabled(False)
        for widget in (self.start, self.stop, self.step):
            widget.valueChanged.connect(lambda _v: self._update_preview())
        self.extra.textChanged.connect(lambda _t: self._update_preview())
        self.load_button.clicked.connect(self._load_delays)
        self.clear_file_button.clicked.connect(self._clear_file)
        grid.addWidget(QtWidgets.QLabel("From"), 0, 0)
        grid.addWidget(self.start, 0, 1)
        grid.addWidget(QtWidgets.QLabel("to"), 0, 2)
        grid.addWidget(self.stop, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Step"), 1, 0)
        grid.addWidget(self.step, 1, 1)
        grid.addWidget(self.load_button, 1, 2)
        grid.addWidget(self.clear_file_button, 1, 3)
        grid.addWidget(self.extra, 2, 0, 1, 4)
        layout.addWidget(self.delay_box)

        self.series_box = QtWidgets.QGroupBox("Time series")
        series_form = QtWidgets.QFormLayout(self.series_box)
        self.datasets = int_spin(100, 1, 1_000_000)
        self.datasets.valueChanged.connect(lambda _v: self._update_preview())
        series_form.addRow("Datasets per run", self.datasets)
        layout.addWidget(self.series_box)

        # ------------------------------------------------------- acquisition
        acquisition_box = QtWidgets.QGroupBox("Acquisition")
        form = QtWidgets.QFormLayout(acquisition_box)
        self.frames = int_spin(10, 1, 1_000_000)
        self.runs = int_spin(1, 0, 100_000)
        self.runs.setSpecialValueText("until stopped")
        self.bidirectional = QtWidgets.QCheckBox("Alternate the direction of successive runs")
        self.bidirectional.setChecked(True)
        self.settle = spin(0.0, 0.0, 3600.0, 2, 0.1, "s")
        self.settle.setToolTip("Wait after each stage move before grabbing frames")
        for widget in (self.frames, self.runs):
            widget.valueChanged.connect(lambda _v: self._update_preview())
        self.settle.valueChanged.connect(lambda _v: self._update_preview())
        form.addRow("Frames per point", self.frames)
        form.addRow("Runs", self.runs)
        form.addRow("", self.bidirectional)
        form.addRow("Settling time", self.settle)
        self.preview = QtWidgets.QLabel()
        self.preview.setWordWrap(True)
        form.addRow(self.preview)
        layout.addWidget(acquisition_box)

        # ----------------------------------------------------------- storage
        self.save_box = QtWidgets.QGroupBox("Save data")
        self.save_box.setCheckable(True)
        self.save_box.setChecked(True)
        storage = QtWidgets.QGridLayout(self.save_box)
        self.name = QtWidgets.QLineEdit("scan")
        self.timestamp = QtWidgets.QCheckBox("Prefix date and time")
        self.timestamp.setChecked(True)
        self.sample = QtWidgets.QLineEdit()
        self.notes = QtWidgets.QLineEdit()
        self.directory_label = QtWidgets.QLabel()
        self.directory_label.setWordWrap(True)
        change_dir = QtWidgets.QPushButton("Change…")
        change_dir.clicked.connect(self._choose_directory)
        self.save_text = QtWidgets.QCheckBox("Text files (tsteps, pp_avgi, ...)")
        self.save_text.setChecked(True)
        self.save_tiff = QtWidgets.QCheckBox("TIFF per point")
        self.save_png = QtWidgets.QCheckBox("PNG preview per point")
        storage.addWidget(QtWidgets.QLabel("Name"), 0, 0)
        storage.addWidget(self.name, 0, 1)
        storage.addWidget(self.timestamp, 0, 2)
        storage.addWidget(QtWidgets.QLabel("Sample"), 1, 0)
        storage.addWidget(self.sample, 1, 1, 1, 2)
        storage.addWidget(QtWidgets.QLabel("Notes"), 2, 0)
        storage.addWidget(self.notes, 2, 1, 1, 2)
        storage.addWidget(QtWidgets.QLabel("Folder"), 3, 0)
        storage.addWidget(self.directory_label, 3, 1)
        storage.addWidget(change_dir, 3, 2)
        storage.addWidget(QtWidgets.QLabel("HDF5 file, plus:"), 4, 0)
        storage.addWidget(self.save_text, 4, 1, 1, 2)
        storage.addWidget(self.save_tiff, 5, 1)
        storage.addWidget(self.save_png, 5, 2)
        self._show_directory()
        layout.addWidget(self.save_box)

        # ------------------------------------------------------------ control
        control = QtWidgets.QGroupBox("Scan")
        control_grid = QtWidgets.QGridLayout(control)
        self.start_button = ToggleButton("Stop scan", "Start scan")
        self.start_button.clicked.connect(self._toggle_scan)
        self.open_button = QtWidgets.QPushButton("Open data folder")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_folder)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        self.status = QtWidgets.QLabel("Ready")
        self.status.setWordWrap(True)
        control_grid.addWidget(self.start_button, 0, 0)
        control_grid.addWidget(self.open_button, 0, 1)
        control_grid.addWidget(self.progress, 1, 0, 1, 2)
        control_grid.addWidget(self.status, 2, 0, 1, 2)
        layout.addWidget(control)
        layout.addStretch(1)

    # ================================================================= delays
    def delays(self) -> np.ndarray:
        """The delays (ps) of one run, or dataset numbers in time-series mode."""
        if self.series_mode.isChecked():
            return np.arange(self.datasets.value(), dtype=float)
        if self.file_delays is not None:
            return self.file_delays
        segments = [(self.start.value(), self.stop.value(), self.step.value())]
        segments += parse_segments(self.extra.text())
        return delays_from_segments(segments) if len(segments) > 1 else delays_from_range(*segments[0])

    def _mode_changed(self) -> None:
        series = self.series_mode.isChecked()
        self.delay_box.setVisible(not series)
        self.series_box.setVisible(series)
        self.bidirectional.setEnabled(not series)
        self.settle.setEnabled(not series)
        self._update_preview()

    def _load_delays(self) -> None:
        path = open_file_name(self, "Load delays", "Text files (*.txt *.dat *.csv);;All files (*)")
        if not path:
            return
        try:
            self.file_delays = delays_from_file(path)
        except Exception as exc:
            self._show_error(f"cannot read delays from {path}: {exc}")
            return
        for widget in (self.start, self.stop, self.step, self.extra):
            widget.setEnabled(False)
        self.clear_file_button.setEnabled(True)
        self.log.info("%d delays loaded from %s", self.file_delays.size, path)
        self._update_preview()

    def _clear_file(self) -> None:
        self.file_delays = None
        for widget in (self.start, self.stop, self.step, self.extra):
            widget.setEnabled(True)
        self.clear_file_button.setEnabled(False)
        self._update_preview()

    def _update_preview(self) -> None:
        try:
            delays = self.delays()
        except ValueError as exc:
            self.preview.setStyleSheet(f"color: {theme.WARN_COLOR};")
            self.preview.setText(str(exc))
            return
        self.preview.setStyleSheet("")
        if self.series_mode.isChecked():
            text = f"{delays.size} datasets per run"
        else:
            source = " (from file)" if self.file_delays is not None else ""
            text = f"{delays.size} delays{source}, {delays.min():g} … {delays.max():g} ps"
        fps = self.acquisition.frame_rate_hz()
        if fps > 0:
            per_point = self.frames.value() / fps + (0.0 if self.series_mode.isChecked() else self.settle.value())
            text += f" · ≈ {format_duration(per_point * delays.size)} per run"
            if not self.series_mode.isChecked():
                text += " plus stage moves"
        self.preview.setText(text)

    # =============================================================== storage
    def _show_directory(self) -> None:
        self.directory_label.setText(str(self.data_dir))

    def _choose_directory(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Data folder", str(self.data_dir))
        if path:
            self.data_dir = Path(path)
            self._show_directory()

    def scan_name(self) -> str:
        name = safe_name(self.name.text())
        return f"{time.strftime('%Y-%m-%d_%H%M%S')}_{name}" if self.timestamp.isChecked() else name

    def metadata(self) -> dict[str, Any]:
        acquisition = self.acquisition
        info: dict[str, Any] = {
            "software": f"uedcontrol {__version__}",
            "instrument": self.config.name,
            "configuration": str(self.config.source) if self.config.source else "built-in",
            "mode": "time series" if self._time_series else "delay scan",
            "sample": self.sample.text(),
            "notes": self.notes.text(),
            "camera": acquisition.camera_name(),
            "exposure_ms": acquisition.exposure_ms(),
            "frames_to_average_live": acquisition.averaging.value(),
            "pixel_size_um": acquisition.pixel_size_um(),
            "rep_rate_khz": acquisition.rep_rate_khz(),
            "background_subtracted": acquisition.subtract_background_enabled(),
        }
        if self.delay_line is not None and not self._time_series:
            info.update(
                delay_stage=self.delay_line.stage.name,
                t0_position=self.delay_line.t0_position,
                stage_unit=self.delay_line.stage.unit,
                passes=self.delay_line.passes,
                direction=self.delay_line.direction,
            )
        return info

    # ================================================================== scan
    def _toggle_scan(self, checked: bool) -> None:
        if checked:
            self.start_scan()
        else:
            self.stop_scan()

    def start_scan(self) -> bool:
        """Validate the inputs and start the scan; returns False (with a message) if it cannot start."""
        if self.running:
            return False
        started = self._start()
        self.start_button.setChecked(started)
        return started

    def _start(self) -> bool:
        try:
            delays = self.delays()
        except ValueError as exc:
            self._show_error(str(exc))
            return False
        camera = self.acquisition.camera
        if camera is None or not camera.connected:
            self._show_error("connect a detector first (Acquisition tab)")
            return False
        self._time_series = self.series_mode.isChecked()
        delay_line = None if self._time_series else self.delay_line
        if delay_line is not None and not delay_line.stage.connected:
            self._show_error(f"connect the delay stage ({delay_line.stage.name}) first")
            return False

        background = self.acquisition.background if self.acquisition.subtract_background_enabled() else None
        try:
            settings = ScanSettings(
                delays,
                frames_per_point=self.frames.value(),
                runs=self.runs.value(),
                bidirectional=self.bidirectional.isChecked() and not self._time_series,
                settle_time_s=0.0 if self._time_series else self.settle.value(),
                roi=self.acquisition.image_view.roi(),
                background=background,
            )
        except ValueError as exc:
            self._show_error(str(exc))
            return False

        self._writer = None
        if self.save_box.isChecked():
            arrays = {}
            if self.acquisition.reference is not None:
                arrays["reference"] = self.acquisition.reference
            self._writer = ScanWriter(
                self.data_dir,
                self.scan_name(),
                self.metadata(),
                text=self.save_text.isChecked(),
                tiff=self.save_tiff.isChecked(),
                png=self.save_png.isChecked(),
                arrays=arrays,
            )

        self.acquisition.stop_live()
        self.view.reset(delays, time_series=self._time_series)
        self._total_points = settings.total_points
        self._started = time.monotonic()
        if self._total_points:
            self.progress.setRange(0, self._total_points)
        else:
            self.progress.setRange(0, 0)  # busy indicator: runs until stopped
        self.progress.setValue(0)
        self._set_running(True)
        where = f" -> {self._writer.scan_dir}" if self._writer is not None else " (not saved)"
        self.log.info("%s started: %d points per run, %s run(s)%s", "time series" if self._time_series else
                      "delay scan", delays.size, settings.runs or "unlimited", where)
        self._request_scan.emit(settings, delay_line, self._writer)
        return True

    def stop_scan(self) -> None:
        if not self.running:
            return
        self.worker.request_stop()
        self.start_button.setEnabled(False)
        self.start_button.setText("Stopping…")
        self.status.setText("Stopping after the current point…")

    def _set_running(self, running: bool) -> None:
        self.running = running
        button = self.start_button
        button.setEnabled(True)
        button.setChecked(running)
        button.setText(button.on_text if running else button.off_text)  # also undoes "Stopping…"
        for box in (self.delay_box, self.series_box, self.save_box):
            box.setEnabled(not running)
        for widget in (self.delay_mode, self.series_mode, self.frames, self.runs, self.bidirectional, self.settle):
            widget.setEnabled(not running)
        if not running:
            self.delay_mode.setEnabled(self.delay_line is not None)
            self._mode_changed()
        self.acquisition.set_scanning(running)
        if running:
            self.status.setStyleSheet("")
            self.status.setText("Starting…")
        self.scan_running.emit(running)

    def _on_progress(self, progress: ScanProgress) -> None:
        self.view.update_progress(progress)
        point = progress.point
        where = f"dataset {point.index + 1}" if self._time_series else f"{point.delay_ps:g} ps"
        elapsed = time.monotonic() - self._started
        if self._total_points:
            self.progress.setValue(point.number)
            remaining = elapsed / point.number * (self._total_points - point.number)
            timing = f"{format_duration(elapsed)} elapsed, ≈ {format_duration(remaining)} left"
            count = f"{point.number}/{self._total_points}"
        else:
            timing = f"{format_duration(elapsed)} elapsed"
            count = str(point.number)
        self.status.setText(f"Point {count} · run {point.run} · {where} · {timing}")

    def _on_finished(self, accumulator: Any, message: str) -> None:
        points = accumulator.completed_points if accumulator is not None else 0
        duration = format_duration(time.monotonic() - self._started)
        self._set_running(False)
        if self._total_points:
            self.progress.setRange(0, self._total_points)
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(1 if points else 0)
        if self._writer is not None and points:
            self.last_scan_dir = self._writer.scan_dir
            self.open_button.setEnabled(True)
        if message:
            self._show_error(f"Scan stopped with an error after {points} point(s): {message}")
            self.log.error("scan failed after %d point(s): %s", points, message)
        else:
            self.status.setStyleSheet("")
            saved = f", saved in {self.last_scan_dir}" if self._writer is not None and points else ""
            self.status.setText(f"Finished: {points} point(s) in {duration}{saved}")
            self.log.info("scan finished: %d point(s) in %s%s", points, duration, saved)
        self._writer = None

    def _open_folder(self) -> None:
        if self.last_scan_dir is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.last_scan_dir)))

    def _show_error(self, message: str) -> None:
        self.status.setStyleSheet(f"color: {theme.WARN_COLOR};")
        self.status.setText(message)
