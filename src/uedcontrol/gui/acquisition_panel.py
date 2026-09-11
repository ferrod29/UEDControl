"""Live acquisition controls: detector, ROI, background/reference and beam statistics."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..acquisition.storage import load_image, save_image
from ..analysis.beam import BeamCharge, RoiStatistics
from ..config import BeamConfig
from ..devices.interfaces import Camera
from . import theme
from .panels.generic import ActionsBox, SettingsBox
from .qt import QtWidgets, Signal
from .widgets.common import Readout, ToggleButton, int_spin, open_file_name, save_file_name, spin
from .widgets.image_view import ImageView
from .workers import AcquisitionWorker, DeviceController

IMAGE_FILTERS = "TIFF (*.tiff *.tif);;PNG preview (*.png);;NumPy (*.npy)"


class AcquisitionPanel(QtWidgets.QWidget):
    send_to_analysis = Signal(object)
    frame_rate_changed = Signal(float)
    _request_live = Signal()
    _request_capture = Signal(str, int)

    def __init__(
        self,
        cameras: Mapping[str, Camera],
        labels: Mapping[str, str],
        worker: AcquisitionWorker,
        image_view: ImageView,
        beam: BeamConfig,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.cameras = dict(cameras)
        self.labels = dict(labels)
        self.worker = worker
        self.image_view = image_view
        self.log = logging.getLogger("uedcontrol.gui.acquisition")
        self.controllers = {key: DeviceController(f"camera-{key}", self) for key in self.cameras}
        for controller in self.controllers.values():
            controller.error.connect(self._show_error)
        self.camera: Camera | None = None
        self.camera_key: str | None = None
        self.background: np.ndarray | None = None
        self.reference: np.ndarray | None = None
        self.last_image: np.ndarray | None = None
        self.last_stats: RoiStatistics | None = None
        self.last_charge: BeamCharge | None = None
        self._fps = 0.0
        self._track_origin: tuple[float, float] | None = None
        self._counts_per_electron = beam.counts_per_electron

        self._build(beam)
        self._request_live.connect(worker.run_live)
        self._request_capture.connect(worker.run_capture)
        worker.frame_ready.connect(self._show_frame)
        worker.roi_moved.connect(self._roi_moved)
        worker.live_state.connect(self._live_state)
        worker.captured.connect(self._captured)
        worker.error.connect(self._show_error)
        image_view.roi_changed.connect(self._roi_changed)
        self._update_beam_settings()
        if self.cameras:
            self._camera_selected(0)

    # ================================================================== layout
    def _build(self, beam: BeamConfig) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # --------------------------------------------------------- detector
        detector = QtWidgets.QGroupBox("Detector")
        grid = QtWidgets.QGridLayout(detector)
        self.camera_combo = QtWidgets.QComboBox()
        for key in self.cameras:
            self.camera_combo.addItem(self.labels.get(key, key), key)
        self.camera_combo.currentIndexChanged.connect(self._camera_selected)
        self.connect_button = ToggleButton("Connected", "Connect")
        self.connect_button.clicked.connect(self._toggle_connect)
        self.exposure = spin(100.0, 0.01, 1e6, 2, 10.0, "ms")
        self.exposure.valueChanged.connect(self._exposure_changed)
        self.averaging = int_spin(1, 1, 10_000)
        self.averaging.valueChanged.connect(self._averaging_changed)
        self.frame_rate = Readout()
        self.integration = Readout()
        self.pixel_size = spin(1.0, 0.001, 1e4, 3, 0.1, "µm")
        self.live_button = ToggleButton("Live ON", "Live OFF")
        self.live_button.clicked.connect(self._toggle_live)
        auto_levels = QtWidgets.QPushButton("Auto levels")
        auto_levels.clicked.connect(self.image_view.auto_levels)
        save_button = QtWidgets.QPushButton("Save image…")
        save_button.clicked.connect(self.save_snapshot)
        analysis_button = QtWidgets.QPushButton("Send to analysis")
        analysis_button.clicked.connect(self._send_to_analysis)
        grid.addWidget(self.camera_combo, 0, 0, 1, 2)
        grid.addWidget(self.connect_button, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Exposure"), 1, 0)
        grid.addWidget(self.exposure, 1, 1)
        grid.addWidget(self.frame_rate, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Frames averaged"), 2, 0)
        grid.addWidget(self.averaging, 2, 1)
        grid.addWidget(self.integration, 2, 2)
        grid.addWidget(QtWidgets.QLabel("Pixel size"), 3, 0)
        grid.addWidget(self.pixel_size, 3, 1)
        grid.addWidget(self.live_button, 4, 0)
        grid.addWidget(auto_levels, 4, 1)
        grid.addWidget(save_button, 5, 0)
        grid.addWidget(analysis_button, 5, 1, 1, 2)
        layout.addWidget(detector)
        self.camera_settings = SettingsBox(self._run_camera, "detector", "Detector settings")
        self.camera_actions = ActionsBox(self._run_camera, "detector", "Detector commands")
        layout.addWidget(self.camera_settings)
        layout.addWidget(self.camera_actions)

        # -------------------------------------------------------------- ROI
        roi_box = QtWidgets.QGroupBox("Region of interest")
        roi_grid = QtWidgets.QGridLayout(roi_box)
        self.roi_button = ToggleButton("ROI ON", "ROI OFF")
        self.roi_button.clicked.connect(self.image_view.set_roi_visible)
        self.ring_button = ToggleButton("Ring ON", "Ring OFF")
        self.ring_button.clicked.connect(self.image_view.set_ring_visible)
        self.crosshair_button = ToggleButton("Crosshair ON", "Crosshair OFF")
        self.crosshair_button.clicked.connect(self.image_view.set_crosshair_visible)
        self.track_button = ToggleButton("Tracking ON", "Tracking OFF")
        self.track_button.clicked.connect(self._toggle_tracking)
        self.block_button = ToggleButton("Beam block ON", "Beam block OFF")
        self.block_button.setToolTip("Blank the current ROI area in the display and statistics")
        self.block_button.clicked.connect(self._toggle_block)
        self.detector_mask = QtWidgets.QCheckBox("Mask ROI on the detector")
        self.detector_mask.setEnabled(False)
        self.detector_mask.toggled.connect(self._toggle_detector_mask)
        for index, widget in enumerate((self.roi_button, self.ring_button, self.crosshair_button,
                                        self.track_button, self.block_button)):
            roi_grid.addWidget(widget, index // 3, index % 3)
        roi_grid.addWidget(self.detector_mask, 2, 0, 1, 3)
        layout.addWidget(roi_box)

        # ------------------------------------------------------- background
        background = QtWidgets.QGroupBox("Background / reference")
        bg_grid = QtWidgets.QGridLayout(background)
        self.capture_frames = int_spin(10, 1, 100_000)
        self.acquire_background = QtWidgets.QPushButton("Acquire background")
        self.acquire_background.clicked.connect(lambda: self._acquire("background"))
        self.acquire_reference = QtWidgets.QPushButton("Acquire reference")
        self.acquire_reference.clicked.connect(lambda: self._acquire("reference"))
        self.subtract = QtWidgets.QCheckBox("Subtract background")
        self.subtract.toggled.connect(lambda on: self.worker.update_settings(subtract_background=on))
        load_bg = QtWidgets.QPushButton("Load background…")
        load_bg.clicked.connect(self._load_background)
        save_bg = QtWidgets.QPushButton("Save background…")
        save_bg.clicked.connect(self._save_background)
        clear_bg = QtWidgets.QPushButton("Clear")
        clear_bg.clicked.connect(self._clear_background)
        self.background_status = QtWidgets.QLabel("No background")
        bg_grid.addWidget(QtWidgets.QLabel("Frames"), 0, 0)
        bg_grid.addWidget(self.capture_frames, 0, 1)
        bg_grid.addWidget(self.subtract, 0, 2)
        bg_grid.addWidget(self.acquire_background, 1, 0)
        bg_grid.addWidget(self.acquire_reference, 1, 1)
        bg_grid.addWidget(clear_bg, 1, 2)
        bg_grid.addWidget(load_bg, 2, 0)
        bg_grid.addWidget(save_bg, 2, 1)
        bg_grid.addWidget(self.background_status, 3, 0, 1, 3)
        layout.addWidget(background)

        # ------------------------------------------------------------- beam
        beam_box = QtWidgets.QGroupBox("Beam")
        beam_grid = QtWidgets.QGridLayout(beam_box)
        self.rep_rate = spin(beam.rep_rate_khz, 0.001, 1e5, 3, 0.1, "kHz")
        self.rep_rate.valueChanged.connect(lambda _v: self._update_beam_settings())
        self.pulsed = QtWidgets.QCheckBox("Pulsed beam")
        self.pulsed.setChecked(beam.pulsed)
        self.pulsed.toggled.connect(lambda _on: self._update_beam_settings())
        beam_grid.addWidget(QtWidgets.QLabel("Repetition rate"), 0, 0)
        beam_grid.addWidget(self.rep_rate, 0, 1)
        beam_grid.addWidget(self.pulsed, 0, 2)

        self.centre = Readout()
        self.diameter = Readout()
        beam_grid.addWidget(QtWidgets.QLabel("Peak (x, y) px"), 1, 0)
        beam_grid.addWidget(self.centre, 1, 1, 1, 2)
        beam_grid.addWidget(QtWidgets.QLabel("ROI diameter"), 2, 0)
        beam_grid.addWidget(self.diameter, 2, 1, 1, 2)

        widths = QtWidgets.QGridLayout()
        for column, text in enumerate(("", "x (px)", "x (µm)", "y (px)", "y (µm)")):
            widths.addWidget(QtWidgets.QLabel(text), 0, column)
        self.width_readouts: dict[tuple[str, str], Readout] = {}
        for row, (key, label) in enumerate((("rms", "RMS"), ("fwhm", "FWHM"), ("e2", "1/e²")), start=1):
            widths.addWidget(QtWidgets.QLabel(label), row, 0)
            for column, axis_unit in enumerate(("x_px", "x_um", "y_px", "y_um"), start=1):
                readout = Readout()
                self.width_readouts[(key, axis_unit)] = readout
                widths.addWidget(readout, row, column)
        beam_grid.addLayout(widths, 3, 0, 1, 3)

        counts = QtWidgets.QFormLayout()
        self.total_counts = Readout()
        self.counts_per_pixel = Readout()
        self.electrons = Readout()
        self.electrons_rms = Readout()
        self.bunch_charge = Readout()
        self.current = Readout()
        self.drift = Readout()
        counts.addRow("Counts in ROI", self.total_counts)
        counts.addRow("Counts per pixel", self.counts_per_pixel)
        counts.addRow("Electrons per pulse", self.electrons)
        counts.addRow("Electrons in RMS / FWHM / 1/e²", self.electrons_rms)
        counts.addRow("Bunch charge (fC)", self.bunch_charge)
        counts.addRow("Beam current (pA)", self.current)
        counts.addRow("Tracking drift (px)", self.drift)
        beam_grid.addLayout(counts, 4, 0, 1, 3)
        layout.addWidget(beam_box)

        self.error_label = QtWidgets.QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(f"color: {theme.WARN_COLOR};")
        layout.addWidget(self.error_label)
        layout.addStretch(1)

    # ========================================================== accessors
    def is_live(self) -> bool:
        return self.live_button.isChecked()

    def camera_name(self) -> str:
        return self.labels.get(self.camera_key, self.camera_key or "") if self.camera_key else ""

    def exposure_ms(self) -> float:
        return self.exposure.value()

    def frame_rate_hz(self) -> float:
        return self._fps

    def pixel_size_um(self) -> float:
        return self.pixel_size.value()

    def rep_rate_khz(self) -> float:
        return self.rep_rate.value()

    def subtract_background_enabled(self) -> bool:
        return self.subtract.isChecked() and self.background is not None

    # ========================================================= connection
    def connect_camera(self, key: str) -> None:
        index = self.camera_combo.findData(key)
        if index < 0:
            return
        self.camera_combo.setCurrentIndex(index)
        if not self.cameras[key].connected:
            self.connect_button.setChecked(True)
            self._toggle_connect(True)

    def _current_key(self) -> str | None:
        return self.camera_combo.currentData()

    def _camera_selected(self, index: int) -> None:
        key = self.camera_combo.itemData(index)
        if key is None:
            return
        if self.camera_key is not None and key != self.camera_key:
            self.stop_live()
            self.camera = None
            self.camera_key = None
            self.worker.camera = None
        camera = self.cameras[key]
        self.connect_button.setChecked(camera.connected)
        if camera.connected:
            self._activate(key)

    def _activate(self, key: str) -> None:
        self.camera_key = key
        self.camera = self.cameras[key]
        self.worker.camera = self.camera
        self.detector_mask.setEnabled(hasattr(self.camera, "mask_disk"))

    def _toggle_connect(self, checked: bool) -> None:
        key = self._current_key()
        if key is None:
            self.connect_button.setChecked(False)
            return
        camera = self.cameras[key]
        controller = self.controllers[key]
        self.connect_button.setEnabled(False)
        if checked:
            controller.submit(lambda: self._connect_camera(camera),
                              done=lambda state, k=key: self._camera_connected(k, state),
                              failed=self._camera_failed)
        else:
            self.stop_live()
            self.camera = None
            self.camera_key = None
            self.worker.camera = None
            controller.submit(camera.disconnect, done=lambda _result: self._camera_disconnected(),
                              failed=self._camera_failed)

    @staticmethod
    def _connect_camera(camera: Camera) -> tuple[Any, ...]:
        camera.connect()
        try:
            identity = camera.identify()
        except Exception:
            identity = camera.model
        return (identity, camera.exposure(), camera.frame_rate(), camera.pixel_size_um,
                camera.settings(), camera.actions())

    def _camera_connected(self, key: str, state: tuple[Any, ...]) -> None:
        identity, exposure, fps, pixel_size, settings, actions = state
        self.connect_button.setEnabled(True)
        self.connect_button.setChecked(True)
        self._activate(key)
        self.exposure.blockSignals(True)
        self.exposure.setValue(exposure)
        self.exposure.blockSignals(False)
        self.pixel_size.setValue(pixel_size)
        self._show_rate(fps)
        self.camera_settings.populate(settings)
        self.camera_actions.populate(actions)
        self.error_label.clear()
        self.log.info("camera %s connected: %s", key, identity)

    def _camera_failed(self, message: str) -> None:
        key = self._current_key()
        self.connect_button.setEnabled(True)
        self.connect_button.setChecked(key is not None and self.cameras[key].connected)
        self._show_error(message)

    def _camera_disconnected(self) -> None:
        self.connect_button.setEnabled(True)
        self.connect_button.setChecked(False)
        self.camera_settings.populate([])
        self.camera_actions.populate([])

    def _run_camera(self, function: Callable[..., Any], *args: Any, done: Callable[[Any], None] | None = None,
                    description: str | None = None) -> None:
        if self.camera_key is None:
            self._show_error("connect a camera first")
            return
        if description:
            self.log.info(description)
        self.controllers[self.camera_key].submit(lambda: function(*args), done=done)

    # ======================================================== exposure etc.
    def _exposure_changed(self, value: float) -> None:
        camera = self.camera
        if camera is None:
            return
        self._run_camera(lambda: (camera.set_exposure(value), camera.frame_rate())[1], done=self._show_rate,
                         description=f"exposure -> {value:g} ms")

    def _show_rate(self, fps: float) -> None:
        self._fps = float(fps)
        self.frame_rate.show_value(self._fps, "{:.2f} Hz")
        self._update_integration()
        self.frame_rate_changed.emit(self._fps)

    def _averaging_changed(self, frames: int) -> None:
        self.worker.update_settings(frames_to_average=frames)
        self._update_integration()

    def _update_integration(self) -> None:
        value = self.averaging.value() / self._fps if self._fps > 0 else None
        self.integration.show_value(value, "{:.3g} s")

    def _update_beam_settings(self) -> None:
        self.worker.update_settings(
            rep_rate_hz=self.rep_rate.value() * 1e3,
            pulsed=self.pulsed.isChecked(),
            counts_per_electron=self._counts_per_electron,
        )

    # ================================================================ live
    def start_live(self) -> None:
        if self.camera is None or not self.camera.connected:
            self.live_button.setChecked(False)
            self._show_error("connect a camera first")
            return
        self.live_button.setChecked(True)
        self.error_label.clear()
        self._request_live.emit()

    def stop_live(self) -> None:
        self.worker.request_stop()

    def _toggle_live(self, checked: bool) -> None:
        if checked:
            self.start_live()
        else:
            self.stop_live()

    def _live_state(self, running: bool) -> None:
        self.live_button.setChecked(running)

    def set_scanning(self, scanning: bool) -> None:
        for widget in (self.camera_combo, self.connect_button, self.live_button,
                       self.acquire_background, self.acquire_reference):
            widget.setEnabled(not scanning)

    def _show_frame(self, image: np.ndarray, stats: RoiStatistics | None, charge: BeamCharge | None) -> None:
        try:
            self.last_image, self.last_stats = image, stats
            if charge is not None:
                self.last_charge = charge
            self.image_view.set_image(image)
            if stats is not None:
                self.image_view.set_projections(stats.x_profile, stats.y_profile, stats.col0, stats.row0)
                self._show_statistics(stats, charge)
            else:
                self.image_view.set_projections(image.sum(axis=0), image.sum(axis=1))
        finally:
            self.worker.frame_consumed()

    def _show_statistics(self, stats: RoiStatistics, charge: BeamCharge | None) -> None:
        pixel = self.pixel_size.value()
        self.centre.setText(f"{stats.peak_x:.0f}, {stats.peak_y:.0f}")
        self.diameter.setText(f"{stats.diameter} px = {stats.diameter * pixel:.1f} µm")
        for key in ("rms", "fwhm", "e2"):
            x_value, y_value = getattr(stats.x_widths, key), getattr(stats.y_widths, key)
            self.width_readouts[(key, "x_px")].show_value(x_value, "{:.1f}")
            self.width_readouts[(key, "x_um")].show_value(x_value * pixel, "{:.1f}")
            self.width_readouts[(key, "y_px")].show_value(y_value, "{:.1f}")
            self.width_readouts[(key, "y_um")].show_value(y_value * pixel, "{:.1f}")
        self.total_counts.show_value(stats.counts.total, "{:.5g}")
        self.counts_per_pixel.show_value(stats.counts.per_pixel, "{:.4g}")
        if charge is not None:
            self.electrons.show_value(charge.electrons_per_pulse, "{:.4g}")
            self.electrons_rms.setText(
                f"{charge.electrons_rms:.3g} / {charge.electrons_fwhm:.3g} / {charge.electrons_e2:.3g}")
            self.bunch_charge.show_value(charge.bunch_charge_fc, "{:.4g}")
            self.current.show_value(charge.current_pa, "{:.4g}")

    # ================================================================= ROI
    def _roi_changed(self, roi: Any) -> None:
        self.worker.update_settings(roi=roi)
        if roi is None and self.track_button.isChecked():
            self.track_button.setChecked(False)
            self._toggle_tracking(False)

    def _roi_moved(self, cx: float, cy: float) -> None:
        self.image_view.set_roi_center(cx, cy)
        if self._track_origin is not None:
            self.drift.setText(f"{cx - self._track_origin[0]:+.1f}, {cy - self._track_origin[1]:+.1f}")

    def _toggle_tracking(self, checked: bool) -> None:
        roi = self.image_view.roi()
        if checked and roi is None:
            self.track_button.setChecked(False)
            self._show_error("switch the ROI on before tracking")
            return
        if checked and self.block_button.isChecked():
            self.block_button.setChecked(False)
            self._toggle_block(False)
        self._track_origin = (roi.cx, roi.cy) if checked and roi is not None else None
        self.worker.update_settings(track=checked)

    def _toggle_block(self, checked: bool) -> None:
        roi = self.image_view.roi()
        if checked and roi is None:
            self.block_button.setChecked(False)
            self._show_error("place the ROI on the area to block first")
            return
        self.worker.update_settings(block_region=roi if checked else None)

    def _toggle_detector_mask(self, checked: bool) -> None:
        camera = self.camera
        if camera is None or not hasattr(camera, "mask_disk"):
            return
        roi = self.image_view.roi()
        if checked and roi is None:
            self.detector_mask.setChecked(False)
            self._show_error("place the ROI on the area to mask first")
            return
        if checked:
            self._run_camera(camera.mask_disk, roi.cx, roi.cy, roi.radius, True,
                             description="masking the ROI on the detector")
        else:
            self._run_camera(camera.mask_disk, 0.0, 0.0, 1.0, False, description="restoring the detector pixel mask")

    # ======================================================= background
    def _acquire(self, kind: str) -> None:
        frames = self.capture_frames.value()
        if self.is_live():
            self.worker.request_capture(kind, frames)
        elif self.camera is not None and self.camera.connected:
            self._request_capture.emit(kind, frames)
        else:
            self._show_error("connect a camera first")
            return
        self.background_status.setText(f"Acquiring {kind} ({frames} frames)…")

    def _captured(self, kind: str, image: np.ndarray) -> None:
        if kind == "background":
            self.background = image
            self.worker.set_background(image)
        else:
            self.reference = image
        self.background_status.setText(f"{kind.capitalize()} acquired at {time.strftime('%H:%M:%S')}")
        self.log.info("%s acquired", kind)

    def _load_background(self) -> None:
        path = open_file_name(self, "Load background", "Images (*.tif *.tiff *.png *.npy)")
        if path:
            try:
                self._captured("background", load_image(path))
            except Exception as exc:
                self._show_error(f"cannot load {path}: {exc}")

    def _save_background(self) -> None:
        if self.background is None:
            self._show_error("no background to save")
            return
        path = save_file_name(self, "Save background", "background.tiff", IMAGE_FILTERS)
        if path:
            save_image(path, self.background)
            self.log.info("background saved to %s", path)

    def _clear_background(self) -> None:
        self.background = None
        self.reference = None
        self.worker.set_background(None)
        self.background_status.setText("No background")

    # =========================================================== snapshots
    def save_snapshot(self) -> None:
        if self.last_image is None:
            self._show_error("no image to save yet")
            return
        path = save_file_name(self, "Save image", "image.tiff", IMAGE_FILTERS)
        if path:
            save_image(path, self.last_image)
            self.log.info("image saved to %s", path)

    def _send_to_analysis(self) -> None:
        if self.last_image is not None:
            self.send_to_analysis.emit(np.array(self.last_image, copy=True))

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)

    def shutdown(self) -> None:
        self.worker.request_stop()
        for controller in self.controllers.values():
            controller.shutdown()
        for key, camera in self.cameras.items():
            if camera.connected:
                try:
                    camera.disconnect()
                except Exception:
                    self.log.exception("disconnecting camera %s failed", key)
