"""Live pump-probe display: ROI profile map versus delay, ROI counts and RMS contrast."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from .qt import QtCore, QtWidgets
from .theme import diverging_colormap
from .workers import ScanProgress


class PumpProbeView(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.relative = QtWidgets.QCheckBox("Show the change relative to negative delays")
        self.relative.toggled.connect(self._redraw)
        self.graphics = pg.GraphicsLayoutWidget()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.relative)
        layout.addWidget(self.graphics, 1)

        self.map_plot = self.graphics.addPlot(row=0, col=0)
        self.map_plot.setLabel("left", "ROI profile position (px)")
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.map_plot.addItem(self.image_item)
        self.histogram = pg.HistogramLUTItem()
        self.histogram.setImageItem(self.image_item)
        self.histogram.gradient.setColorMap(diverging_colormap())
        self.graphics.addItem(self.histogram, row=0, col=1)

        symbol = {"symbol": "o", "symbolSize": 5, "symbolBrush": "c"}
        self.total_plot = self.graphics.addPlot(row=1, col=0)
        self.total_plot.setLabel("left", "ROI counts")
        self.total_plot.showGrid(x=True, y=True, alpha=0.3)
        self.total_curve = self.total_plot.plot(pen=pg.mkPen("c", width=1.5), **symbol)
        self.rms_plot = self.graphics.addPlot(row=2, col=0)
        self.rms_plot.setLabel("left", "RMS contrast")
        self.rms_plot.showGrid(x=True, y=True, alpha=0.3)
        self.rms_plot.setXLink(self.total_plot)
        self.rms_curve = self.rms_plot.plot(pen=pg.mkPen("c", width=1.5), **symbol)
        self._progress: ScanProgress | None = None
        self._axis_label = "Delay (ps)"
        self.reset(np.zeros(0))

    def reset(self, delays: np.ndarray, time_series: bool = False) -> None:
        self._progress = None
        self._axis_label = "Dataset" if time_series else "Delay (ps)"
        for plot in (self.map_plot, self.total_plot, self.rms_plot):
            plot.setLabel("bottom", self._axis_label)
        self.image_item.clear()
        self.total_curve.setData([], [])
        self.rms_curve.setData([], [])

    def update_progress(self, progress: ScanProgress) -> None:
        self._progress = progress
        self._redraw()

    def _redraw(self) -> None:
        progress = self._progress
        if progress is None:
            return
        order = np.argsort(progress.delays)
        delays = progress.delays[order]
        valid = progress.counts[order] > 0
        self.total_curve.setData(delays[valid], progress.total[order][valid])
        self.rms_curve.setData(delays[valid], progress.rms[order][valid])
        if progress.profile is None:
            return

        profile = progress.profile[order]
        relative = self.relative.isChecked()
        if relative:
            reference_rows = profile[(delays < 0) & valid]
            if len(reference_rows):
                reference = np.nanmean(reference_rows, axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    profile = (profile - reference) / np.where(reference == 0, np.nan, reference)
        image = np.nan_to_num(profile.T, nan=0.0)
        if relative:
            limit = float(np.max(np.abs(image))) or 1.0
            self.image_item.setImage(image, levels=(-limit, limit))
        else:
            self.image_item.setImage(image, autoLevels=True)

        n_points, length = delays.size, image.shape[0]
        steps = np.diff(delays)
        if n_points > 1 and np.allclose(steps, steps[0]) and steps[0] > 0:
            x0, width = delays[0] - steps[0] / 2, steps[0] * n_points
            self.map_plot.setLabel("bottom", self._axis_label)
        else:  # irregular grid: one column per point
            x0, width = -0.5, float(n_points)
            self.map_plot.setLabel("bottom", "Delay point #")
        self.image_item.setRect(QtCore.QRectF(x0, -length / 2, width, length))
