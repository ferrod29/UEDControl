"""Image display with histogram, projections, a circular/annular ROI and a crosshair.

Pixel ``(row, col)`` covers ``[col, col+1) x [row, row+1)`` in view
coordinates, so its centre is at ``(col + 0.5, row + 0.5)``. :meth:`ImageView.roi`
converts the graphics ROI to a :class:`~uedcontrol.analysis.beam.CircularROI`
in pixel coordinates, which is all the analysis code ever sees.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ...analysis.beam import CircularROI
from ..qt import QtCore, QtWidgets, Signal


class ImageView(QtWidgets.QWidget):
    roi_changed = Signal(object)  # CircularROI or None
    point_clicked = Signal(float, float)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        colormap: str | pg.ColorMap = "inferno",
        x_label: str = "x (px)",
        y_label: str = "y (px)",
        projections: bool = True,
    ) -> None:
        super().__init__(parent)
        self.graphics = pg.GraphicsLayoutWidget()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.graphics)

        self.plot = self.graphics.addPlot(row=0, col=1)
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        self.plot.setAspectLocked(True)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)
        self.histogram = pg.HistogramLUTItem()
        self.histogram.setImageItem(self.image_item)
        self.graphics.addItem(self.histogram, row=0, col=2)
        self.set_colormap(colormap)

        self.x_plot = self.y_plot = None
        if projections:
            pen = pg.mkPen("c", width=1.5)
            self.x_plot = self.graphics.addPlot(row=1, col=1)
            self.x_plot.setXLink(self.plot)
            self.x_plot.setMaximumHeight(160)
            self.x_plot.showGrid(x=True, y=True, alpha=0.3)
            self.x_curve = self.x_plot.plot(pen=pen)
            self.y_plot = self.graphics.addPlot(row=0, col=0)
            self.y_plot.setYLink(self.plot)
            self.y_plot.setMaximumWidth(160)
            self.y_plot.showGrid(x=True, y=True, alpha=0.3)
            self.y_curve = self.y_plot.plot(pen=pen)

        self.roi_item = pg.CircleROI([0, 0], [100, 100], pen=pg.mkPen("r", width=2))
        dashed = pg.mkPen("r", width=1.5, style=QtCore.Qt.PenStyle.DashLine)
        self.inner_item = pg.CircleROI([25, 25], [50, 50], pen=dashed, movable=False)
        for item, z in ((self.roi_item, 10), (self.inner_item, 11)):
            item.setZValue(z)
            self.plot.addItem(item)
            item.hide()
        self._roi_placed = False
        self._syncing = False
        self.roi_item.sigRegionChanged.connect(self._outer_changed)
        self.inner_item.sigRegionChanged.connect(self._inner_changed)

        line_pen = pg.mkPen("y", width=1)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=line_pen)
        for line in (self.v_line, self.h_line):
            self.plot.addItem(line, ignoreBounds=True)
            line.hide()
        self._crosshair = False
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._mouse_clicked)
        self._has_image = False

    # ------------------------------------------------------------------ image
    @property
    def image(self) -> np.ndarray | None:
        return self.image_item.image

    def set_image(self, image: np.ndarray, auto_levels: bool | None = None) -> None:
        first = not self._has_image
        self.image_item.setImage(np.asarray(image), autoLevels=first if auto_levels is None else auto_levels)
        if first:
            self.plot.autoRange()
            self._has_image = True

    def auto_levels(self) -> None:
        if self._has_image:
            self.image_item.setImage(self.image_item.image, autoLevels=True)

    def clear(self) -> None:
        self.image_item.clear()
        self._has_image = False

    def set_rect(self, x: float, y: float, width: float, height: float) -> None:
        self.image_item.setRect(QtCore.QRectF(x, y, width, height))

    def set_colormap(self, colormap: str | pg.ColorMap) -> None:
        cmap = pg.colormap.get(colormap) if isinstance(colormap, str) else colormap
        self.histogram.gradient.setColorMap(cmap)

    def set_projections(self, x_profile: np.ndarray, y_profile: np.ndarray, x0: float = 0.0, y0: float = 0.0) -> None:
        if self.x_plot is None:
            return
        self.x_curve.setData(np.arange(len(x_profile)) + x0 + 0.5, np.asarray(x_profile))
        self.y_curve.setData(np.asarray(y_profile), np.arange(len(y_profile)) + y0 + 0.5)

    # -------------------------------------------------------------------- ROI
    def roi(self) -> CircularROI | None:
        if not self.roi_item.isVisible():
            return None
        pos, size = self.roi_item.pos(), self.roi_item.size()
        radius = max(size.x() / 2, 0.5)
        inner = 0.0
        if self.inner_item.isVisible():
            inner = min(self.inner_item.size().x() / 2, 0.99 * radius)
        return CircularROI(pos.x() + radius - 0.5, pos.y() + radius - 0.5, radius, inner)

    def set_roi(self, roi: CircularROI) -> None:
        self._syncing = True
        try:
            diameter = 2 * roi.radius
            self.roi_item.setPos((roi.cx + 0.5 - roi.radius, roi.cy + 0.5 - roi.radius), update=False)
            self.roi_item.setSize((diameter, diameter))
            if roi.inner_radius > 0:
                self.inner_item.setSize((2 * roi.inner_radius, 2 * roi.inner_radius), update=False)
            self._center_inner()
        finally:
            self._syncing = False
        self._roi_placed = True
        self._emit_roi()

    def set_roi_center(self, cx: float, cy: float) -> None:
        roi = self.roi()
        if roi is not None:
            self.set_roi(roi.moved_to(cx, cy))

    def set_roi_visible(self, visible: bool) -> None:
        if visible and not self._roi_placed:
            rows, cols = self.image.shape[:2] if self.image is not None else (300, 300)
            radius = max(min(rows, cols) / 6, 5)
            self.roi_item.show()
            self.set_roi(CircularROI((cols - 1) / 2, (rows - 1) / 2, radius))
        self.roi_item.setVisible(visible)
        if not visible:
            self.inner_item.hide()
        self._emit_roi()

    def set_ring_visible(self, visible: bool) -> None:
        if visible and self.roi_item.isVisible():
            self._syncing = True
            try:
                diameter = self.roi_item.size().x() / 2
                self.inner_item.setSize((diameter, diameter), update=False)
                self._center_inner()
            finally:
                self._syncing = False
            self.inner_item.show()
        else:
            self.inner_item.hide()
        self._emit_roi()

    def _center_inner(self) -> None:
        pos, size = self.roi_item.pos(), self.roi_item.size()
        cx, cy = pos.x() + size.x() / 2, pos.y() + size.y() / 2
        diameter = min(self.inner_item.size().x(), 0.95 * size.x())
        self.inner_item.setSize((diameter, diameter), update=False)
        self.inner_item.setPos((cx - diameter / 2, cy - diameter / 2))

    def _outer_changed(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._center_inner()
        finally:
            self._syncing = False
        self._emit_roi()

    def _inner_changed(self) -> None:
        self._outer_changed()

    def _emit_roi(self) -> None:
        self.roi_changed.emit(self.roi())

    # -------------------------------------------------------------- crosshair
    def set_crosshair_visible(self, visible: bool) -> None:
        self._crosshair = visible
        self.v_line.setVisible(visible)
        self.h_line.setVisible(visible)

    def _mouse_moved(self, scene_pos: QtCore.QPointF) -> None:
        if not self._crosshair or not self.plot.sceneBoundingRect().contains(scene_pos):
            return
        point = self.plot.vb.mapSceneToView(scene_pos)
        self.v_line.setPos(point.x())
        self.h_line.setPos(point.y())

    def _mouse_clicked(self, event) -> None:
        if not self._crosshair:
            return
        point = self.plot.vb.mapSceneToView(event.scenePos())
        x, y = point.x() - 0.5, point.y() - 0.5
        self.v_line.setPos(point.x())
        self.h_line.setPos(point.y())
        image = self.image
        if image is not None and 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            self.set_projections(image[int(y), :], image[:, int(x)])  # line cuts through the pixel
        if self.roi_item.isVisible():
            self.set_roi_center(x, y)
        self.point_clicked.emit(x, y)
