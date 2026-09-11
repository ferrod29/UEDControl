"""Stacked time-series plots with a date/time axis (voltages, pressures, powers...)."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg


class TimeSeriesPlot(pg.GraphicsLayoutWidget):
    def __init__(self, series: Sequence[tuple[str, str]], history: int = 20000, parent=None) -> None:
        super().__init__(parent)
        self._times: deque[float] = deque(maxlen=history)
        self._values: list[deque[float]] = [deque(maxlen=history) for _ in series]
        self._curves = []
        pen = pg.mkPen("#ED7117", width=1.5)
        first = None
        for row, (label, unit) in enumerate(series):
            plot = self.addPlot(row=row, col=0, axisItems={"bottom": pg.DateAxisItem()})
            plot.setLabel("left", f"{label} ({unit})" if unit else label)
            plot.showGrid(x=True, y=True, alpha=0.3)
            if first is None:
                first = plot
            else:
                plot.setXLink(first)
            self._curves.append(plot.plot(pen=pen))

    def append(self, timestamp: float, values: Sequence[float | None], redraw: bool = True) -> None:
        self._times.append(float(timestamp))
        for store, value in zip(self._values, values, strict=False):
            store.append(math.nan if value is None else float(value))
        if redraw:
            self.redraw()

    def redraw(self) -> None:
        times = np.fromiter(self._times, dtype=float)
        for curve, store in zip(self._curves, self._values, strict=True):
            curve.setData(times, np.fromiter(store, dtype=float), connect="finite")

    def clear_data(self) -> None:
        self._times.clear()
        for store in self._values:
            store.clear()
        self.redraw()
