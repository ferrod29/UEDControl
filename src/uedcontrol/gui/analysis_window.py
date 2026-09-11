"""Image analysis: beam-profile fits and diffraction rings (replaces the old ``ImageAnalysis.py``).

Works on an image sent from the live view ("Send to analysis") or loaded from
a file. All numerics live in :mod:`uedcontrol.analysis`; this window only
arranges the inputs and results.

* **Beam** - ROI statistics and Gaussian/Lorentzian fits of both projections
  (centre, FWHM, RMS and 1/e² widths in px and µm).
* **Diffraction** - centre from the ROI (or its intensity centroid),
  azimuthal average versus radius or scattering vector q, ring detection
  and export of the ring table.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg

from ..acquisition.storage import load_image, save_image
from ..analysis.beam import analyze_roi, block_region
from ..analysis.diffraction import (
    DiffractionPeak,
    electron_wavelength_angstrom,
    find_rings,
    radial_profile,
    scattering_vector,
    symmetrize,
)
from ..analysis.fitting import fit_profile
from . import theme
from .qt import QtCore, QtWidgets
from .widgets.common import (
    Readout,
    ToggleButton,
    app_settings,
    int_spin,
    open_file_name,
    save_file_name,
    spin,
    write_csv,
)
from .widgets.image_view import ImageView

IMAGE_FILTERS = "Images (*.tif *.tiff *.png *.npy);;All files (*)"


class AnalysisWindow(QtWidgets.QMainWindow):
    def __init__(self, pixel_size_um: float = 1.0, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("UED image analysis")
        self.log = logging.getLogger("uedcontrol.gui.analysis")
        self.settings = app_settings("analysis")
        self.original: np.ndarray | None = None  # as received or loaded
        self.image: np.ndarray | None = None  # after symmetrisation / beam block
        self.peaks: list[DiffractionPeak] = []
        self._radial: tuple[np.ndarray, np.ndarray] | None = None

        self.image_view = ImageView()
        self.profile_plot = pg.PlotWidget()
        self.profile_plot.showGrid(x=True, y=True, alpha=0.3)
        self.profile_plot.addLegend(offset=(-10, 10))
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.image_view)
        splitter.addWidget(self.profile_plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.addWidget(self._build_image_box(pixel_size_um))
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_beam_tab(), "Beam")
        self.tabs.addTab(self._build_diffraction_tab(), "Diffraction")
        self.tabs.currentChanged.connect(lambda _i: self._redraw_profiles())
        side_layout.addWidget(self.tabs, 1)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        side_layout.addWidget(self.message)
        dock = QtWidgets.QDockWidget("Analysis", self)
        dock.setObjectName("analysis_controls")
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.resize(1300, 850)

    # ================================================================== layout
    def _build_image_box(self, pixel_size_um: float) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Image")
        grid = QtWidgets.QGridLayout(box)
        load = QtWidgets.QPushButton("Load…")
        save = QtWidgets.QPushButton("Save…")
        load.clicked.connect(self._load)
        save.clicked.connect(self._save)
        self.source_label = QtWidgets.QLabel("No image")
        self.source_label.setWordWrap(True)
        self.log_scale = QtWidgets.QCheckBox("Logarithmic display")
        self.log_scale.toggled.connect(lambda _on: self._show_image())
        self.pixel_size = spin(pixel_size_um, 0.001, 1e4, 3, 0.1, "µm")
        self.pixel_size.valueChanged.connect(lambda _v: self._pixel_size_changed())
        self.roi_button = ToggleButton("ROI ON", "ROI OFF")
        self.roi_button.clicked.connect(self.image_view.set_roi_visible)
        self.ring_button = ToggleButton("Ring ON", "Ring OFF")
        self.ring_button.clicked.connect(self.image_view.set_ring_visible)
        self.crosshair_button = ToggleButton("Crosshair ON", "Crosshair OFF")
        self.crosshair_button.clicked.connect(self.image_view.set_crosshair_visible)
        self.fold = int_spin(6, 1, 12)
        symmetrize_button = QtWidgets.QPushButton("Symmetrise")
        symmetrize_button.setToolTip("Average over n rotations about the ROI centre")
        symmetrize_button.clicked.connect(self._symmetrize)
        block = QtWidgets.QPushButton("Block ROI")
        block.setToolTip("Blank the ROI area (e.g. the undiffracted beam)")
        block.clicked.connect(self._block)
        revert = QtWidgets.QPushButton("Revert")
        revert.clicked.connect(self._revert)
        grid.addWidget(load, 0, 0)
        grid.addWidget(save, 0, 1)
        grid.addWidget(self.log_scale, 0, 2)
        grid.addWidget(self.source_label, 1, 0, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Pixel size"), 2, 0)
        grid.addWidget(self.pixel_size, 2, 1)
        grid.addWidget(self.roi_button, 3, 0)
        grid.addWidget(self.ring_button, 3, 1)
        grid.addWidget(self.crosshair_button, 3, 2)
        grid.addWidget(QtWidgets.QLabel("n-fold"), 4, 0)
        grid.addWidget(self.fold, 4, 1)
        grid.addWidget(symmetrize_button, 4, 2)
        grid.addWidget(block, 5, 1)
        grid.addWidget(revert, 5, 2)
        return box

    def _build_beam_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        row = QtWidgets.QHBoxLayout()
        self.model = QtWidgets.QComboBox()
        self.model.addItems(["gaussian", "lorentzian"])
        analyse = QtWidgets.QPushButton("Analyse ROI")
        analyse.setToolTip("Statistics and fits of the ROI (whole image when the ROI is off)")
        analyse.clicked.connect(self.analyse_beam)
        row.addWidget(QtWidgets.QLabel("Fit"))
        row.addWidget(self.model)
        row.addWidget(analyse)
        layout.addLayout(row)
        self.beam_table = QtWidgets.QTableWidget(2, 6)
        self.beam_table.setHorizontalHeaderLabels(
            ["Centre (px)", "FWHM (px)", "FWHM (µm)", "RMS (µm)", "1/e² (µm)", "Counted FWHM (px)"])
        self.beam_table.setVerticalHeaderLabels(["x", "y"])
        self.beam_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.beam_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.beam_table.setMaximumHeight(110)
        layout.addWidget(self.beam_table)
        form = QtWidgets.QFormLayout()
        self.total_counts = Readout()
        self.peak_position = Readout()
        self.centroid = Readout()
        form.addRow("Counts in ROI", self.total_counts)
        form.addRow("Peak (x, y) px", self.peak_position)
        form.addRow("Centroid (x, y) px", self.centroid)
        layout.addLayout(form)
        layout.addStretch(1)
        self._beam_curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        return tab

    def _build_diffraction_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        grid = QtWidgets.QGridLayout()
        self.center_x = spin(0.0, -1e5, 1e5, 2, 0.5, "px")
        self.center_y = spin(0.0, -1e5, 1e5, 2, 0.5, "px")
        from_roi = QtWidgets.QPushButton("From ROI")
        from_roi.setToolTip("Use the ROI centre")
        centroid = QtWidgets.QPushButton("Centroid")
        centroid.setToolTip("Intensity centroid inside the ROI (e.g. of the undiffracted beam)")
        from_roi.clicked.connect(self._centre_from_roi)
        centroid.clicked.connect(self._centre_from_centroid)
        self.voltage = spin(float(self.settings.value("voltage_kv", 100.0)), 0.1, 10_000.0, 2, 1.0, "kV")
        self.camera_length = spin(float(self.settings.value("camera_length_m", 0.5)), 1e-3, 100.0, 4, 0.01, "m")
        self.wavelength = Readout()
        self.voltage.valueChanged.connect(lambda _v: self._calibration_changed())
        self.camera_length.valueChanged.connect(lambda _v: self._calibration_changed())
        self.bin_width = spin(1.0, 0.1, 100.0, 2, 0.5, "px")
        self.exclude_roi = QtWidgets.QCheckBox("Exclude the ROI area")
        self.axis_unit = QtWidgets.QComboBox()
        self.axis_unit.addItems(["q (1/Å)", "radius (px)"])
        self.axis_unit.currentIndexChanged.connect(lambda _i: self._redraw_profiles())
        self.prominence = spin(5.0, 0.0, 100.0, 1, 1.0, "%")
        self.prominence.setToolTip("Minimum peak prominence, relative to the profile range")
        self.min_distance = spin(5.0, 1.0, 1000.0, 1, 1.0, "px")
        profile_button = QtWidgets.QPushButton("Azimuthal average")
        profile_button.clicked.connect(self.compute_radial_profile)
        rings_button = QtWidgets.QPushButton("Find rings")
        rings_button.clicked.connect(self.detect_rings)
        save_table = QtWidgets.QPushButton("Save table…")
        save_table.clicked.connect(self._save_rings)
        rows = [
            ("Centre x", self.center_x, from_roi),
            ("Centre y", self.center_y, centroid),
            ("Voltage", self.voltage, None),
            ("Camera length", self.camera_length, None),
            ("Wavelength (Å)", self.wavelength, None),
            ("Bin width", self.bin_width, self.exclude_roi),
            ("Axis", self.axis_unit, profile_button),
            ("Prominence", self.prominence, None),
            ("Min. distance", self.min_distance, rings_button),
        ]
        for index, (label, widget, extra) in enumerate(rows):
            grid.addWidget(QtWidgets.QLabel(label), index, 0)
            grid.addWidget(widget, index, 1)
            if extra is not None:
                grid.addWidget(extra, index, 2)
        layout.addLayout(grid)
        self.ring_table = QtWidgets.QTableWidget(0, 5)
        self.ring_table.setHorizontalHeaderLabels(["r (px)", "q (1/Å)", "d (Å)", "Δq FWHM (1/Å)", "Intensity"])
        self.ring_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ring_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.ring_table, 1)
        layout.addWidget(save_table)
        self._calibration_changed()
        return tab

    # ================================================================= image
    def set_image(self, image: np.ndarray, source: str = "", pixel_size_um: float | None = None) -> None:
        self.original = np.asarray(image, dtype=np.float64)
        self.image = self.original.copy()
        rows, cols = self.original.shape
        self.source_label.setText(f"{source} · {cols} × {rows} px" if source else f"{cols} × {rows} px")
        if pixel_size_um is not None:
            self.pixel_size.setValue(pixel_size_um)
        if self.center_x.value() == 0 and self.center_y.value() == 0:
            self.center_x.setValue((cols - 1) / 2)
            self.center_y.setValue((rows - 1) / 2)
        self.image_view.clear()
        self._radial = None
        self.peaks = []
        self._beam_curves = {}
        self._fill_rings()
        self._show_image()
        self._redraw_profiles()
        self._info("")

    def _show_image(self) -> None:
        if self.image is None:
            return
        shown = self.image
        if self.log_scale.isChecked():
            shown = np.log10(np.clip(shown - shown.min(), 0, None) + 1.0)
        self.image_view.set_image(shown, auto_levels=True)

    def _require_image(self) -> np.ndarray | None:
        if self.image is None:
            self._info("load an image or use “Send to analysis” in the acquisition panel first", warning=True)
        return self.image

    def _load(self) -> None:
        path = open_file_name(self, "Load image", IMAGE_FILTERS)
        if path:
            try:
                self.set_image(load_image(path), Path(path).name)
            except Exception as exc:
                self._info(f"cannot load {path}: {exc}", warning=True)

    def _save(self) -> None:
        if self._require_image() is None:
            return
        path = save_file_name(self, "Save image", "analysis.tiff",
                              "TIFF (*.tiff *.tif);;PNG preview (*.png);;NumPy (*.npy)")
        if path:
            save_image(path, self.image)
            self.log.info("analysed image saved to %s", path)

    def _symmetrize(self) -> None:
        image = self._require_image()
        if image is None:
            return
        center = (self.center_x.value(), self.center_y.value())
        self.image = symmetrize(image, self.fold.value(), center)
        self._show_image()
        self._info(f"{self.fold.value()}-fold symmetrised about ({center[0]:.1f}, {center[1]:.1f})")

    def _block(self) -> None:
        image = self._require_image()
        roi = self.image_view.roi()
        if image is None:
            return
        if roi is None:
            self._info("switch the ROI on and place it on the area to block", warning=True)
            return
        self.image = block_region(image, roi)
        self._show_image()

    def _revert(self) -> None:
        if self.original is not None:
            self.image = self.original.copy()
            self._show_image()
            self._info("reverted to the original image")

    def _pixel_size_changed(self) -> None:
        if self._beam_curves:
            self.analyse_beam()
        elif self.peaks:
            self.detect_rings()

    # ================================================================== beam
    def analyse_beam(self) -> None:
        image = self._require_image()
        if image is None:
            return
        try:
            stats = analyze_roi(image, self.image_view.roi())
        except ValueError as exc:
            self._info(str(exc), warning=True)
            return
        pixel = self.pixel_size.value()
        model = self.model.currentText()
        self._beam_curves = {}
        failed = []
        for row, (axis, profile, origin, counted) in enumerate((
            ("x", stats.x_profile, stats.col0, stats.x_widths),
            ("y", stats.y_profile, stats.row0, stats.y_widths),
        )):
            positions = np.arange(profile.size) + origin
            try:
                fit = fit_profile(profile, model, x=positions)
            except ValueError as exc:
                failed.append(f"{axis}: {exc}")
                values = (math.nan,) * 5
            else:
                if not fit.success:
                    failed.append(f"{axis}: the fit did not converge")
                values = (fit.center, fit.fwhm, fit.fwhm * pixel, fit.rms * pixel, fit.e2_width * pixel)
                self._beam_curves[axis] = (positions, profile, fit.evaluate(positions))
            for column, value in enumerate((*values, counted.fwhm)):
                text = "–" if not math.isfinite(value) else f"{value:.4g}"
                self.beam_table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        self.total_counts.show_value(stats.counts.total, "{:.5g}")
        self.peak_position.setText(f"{stats.peak_x:.0f}, {stats.peak_y:.0f}")
        self.centroid.setText(f"{stats.centroid_x:.1f}, {stats.centroid_y:.1f}")
        self._redraw_profiles()
        if failed:
            self._info("; ".join(failed), warning=True)
        else:
            self._info(f"{model} fits done")

    # =========================================================== diffraction
    def _calibration_changed(self) -> None:
        self.settings.setValue("voltage_kv", self.voltage.value())
        self.settings.setValue("camera_length_m", self.camera_length.value())
        self.wavelength.show_value(self._wavelength(), "{:.5f}")
        if self.peaks:
            self.detect_rings()
        else:
            self._redraw_profiles()

    def _wavelength(self) -> float:
        return electron_wavelength_angstrom(self.voltage.value() * 1e3)

    def _to_q(self, radii: np.ndarray) -> np.ndarray:
        return np.asarray(scattering_vector(radii, self.pixel_size.value(), self.camera_length.value(),
                                            self._wavelength()))

    def _centre_from_roi(self) -> None:
        roi = self.image_view.roi()
        if roi is None:
            self._info("switch the ROI on and centre it on the pattern", warning=True)
            return
        self.center_x.setValue(roi.cx)
        self.center_y.setValue(roi.cy)

    def _centre_from_centroid(self) -> None:
        image = self._require_image()
        if image is None:
            return
        try:
            stats = analyze_roi(image, self.image_view.roi())
        except ValueError as exc:
            self._info(str(exc), warning=True)
            return
        self.center_x.setValue(stats.centroid_x)
        self.center_y.setValue(stats.centroid_y)

    def compute_radial_profile(self) -> tuple[np.ndarray, np.ndarray] | None:
        image = self._require_image()
        if image is None:
            return None
        exclude = None
        roi = self.image_view.roi()
        if self.exclude_roi.isChecked() and roi is not None:
            exclude = roi.mask(image.shape)
        center = (self.center_x.value(), self.center_y.value())
        self._radial = radial_profile(image, center, self.bin_width.value(), exclude)
        self.peaks = []
        self._fill_rings()
        self._redraw_profiles()
        return self._radial

    def detect_rings(self) -> None:
        if self._radial is None and self.compute_radial_profile() is None:
            return
        radii, profile = self._radial
        finite = profile[np.isfinite(profile)]
        span = float(np.ptp(finite)) if finite.size else 0.0
        self.peaks = find_rings(
            radii,
            profile,
            pixel_size_um=self.pixel_size.value(),
            camera_length_m=self.camera_length.value(),
            wavelength_angstrom=self._wavelength(),
            prominence=self.prominence.value() / 100 * span,
            min_distance_px=self.min_distance.value(),
        )
        self._fill_rings()
        self._redraw_profiles()
        self._info(f"{len(self.peaks)} ring(s) found")

    def _fill_rings(self) -> None:
        self.ring_table.setRowCount(len(self.peaks))
        for row, peak in enumerate(self.peaks):
            d_spacing = 2 * math.pi / peak.q if peak.q > 0 else math.nan
            for column, value in enumerate((peak.radius_px, peak.q, d_spacing, peak.q_width, peak.intensity)):
                self.ring_table.setItem(row, column, QtWidgets.QTableWidgetItem(f"{value:.5g}"))

    def _save_rings(self) -> None:
        if not self.peaks:
            self._info("find the rings first", warning=True)
            return
        path = save_file_name(self, "Save ring table", "rings.csv", "CSV files (*.csv)")
        if not path:
            return
        rows = [(p.radius_px, p.q, 2 * math.pi / p.q if p.q > 0 else "", p.q_width, p.intensity) for p in self.peaks]
        write_csv(path, ["radius_px", "q_inv_angstrom", "d_angstrom", "q_fwhm_inv_angstrom", "intensity"], rows)
        self.log.info("ring table saved to %s (centre %.2f, %.2f px; %.3f kV; L = %.4f m; pixel %.3f µm)", path,
                      self.center_x.value(), self.center_y.value(), self.voltage.value(), self.camera_length.value(),
                      self.pixel_size.value())

    # ================================================================= plots
    def _redraw_profiles(self) -> None:
        plot = self.profile_plot
        plot.clear()
        if self.tabs.currentIndex() == 0:
            plot.setLabel("bottom", "Position (px)")
            plot.setLabel("left", "Projected counts")
            colours = {"x": "c", "y": theme.OFF_COLOR}
            for axis, (positions, profile, fitted) in self._beam_curves.items():
                plot.plot(positions, profile, pen=None, symbol="o", symbolSize=4, symbolBrush=colours[axis],
                          name=f"{axis} profile")
                plot.plot(positions, fitted, pen=pg.mkPen(colours[axis], width=2), name=f"{axis} fit")
            return
        use_q = self.axis_unit.currentIndex() == 0
        plot.setLabel("bottom", "q (1/Å)" if use_q else "Radius (px)")
        plot.setLabel("left", "Azimuthal average")
        if self._radial is None:
            return
        radii, profile = self._radial
        x = self._to_q(radii) if use_q else radii
        plot.plot(x, profile, pen=pg.mkPen("c", width=1.5), name="radial profile")
        if self.peaks:
            positions = [p.q if use_q else p.radius_px for p in self.peaks]
            plot.plot(positions, [p.intensity for p in self.peaks], pen=None, symbol="t", symbolSize=10,
                      symbolBrush=theme.OFF_COLOR, name="rings")

    def _info(self, text: str, warning: bool = False) -> None:
        self.message.setStyleSheet(f"color: {theme.WARN_COLOR};" if warning else "")
        self.message.setText(text)
