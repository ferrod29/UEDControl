"""Look and feel: dark Fusion palette, pyqtgraph defaults and colour maps."""

from __future__ import annotations

import pyqtgraph as pg

from .qt import QtGui, QtWidgets

ON_COLOR = "#00d0d0"  # cyan, as in the original software
OFF_COLOR = "#ff9f1a"  # orange
WARN_COLOR = "#ff5c5c"


def configure_pyqtgraph() -> None:
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)


def dark_palette() -> QtGui.QPalette:
    role = QtGui.QPalette.ColorRole
    disabled = QtGui.QPalette.ColorGroup.Disabled
    palette = QtGui.QPalette()
    for colour_role, rgb in {
        role.Window: (53, 53, 53),
        role.WindowText: (255, 255, 255),
        role.Base: (25, 25, 25),
        role.AlternateBase: (53, 53, 53),
        role.ToolTipBase: (53, 53, 53),
        role.ToolTipText: (255, 255, 255),
        role.Text: (255, 255, 255),
        role.Button: (53, 53, 53),
        role.ButtonText: (255, 255, 255),
        role.BrightText: (255, 0, 0),
        role.Link: (42, 130, 218),
        role.Highlight: (42, 130, 218),
        role.HighlightedText: (0, 0, 0),
    }.items():
        palette.setColor(colour_role, QtGui.QColor(*rgb))
    for colour_role, rgb in {
        role.Base: (49, 49, 49),
        role.Text: (130, 130, 130),
        role.Button: (42, 42, 42),
        role.ButtonText: (110, 110, 110),
        role.Window: (49, 49, 49),
        role.WindowText: (130, 130, 130),
    }.items():
        palette.setColor(disabled, colour_role, QtGui.QColor(*rgb))
    return palette


def apply_dark_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    app.setPalette(dark_palette())


def diverging_colormap() -> pg.ColorMap:
    """Blue - white - red, for signed pump-probe changes."""
    return pg.ColorMap([0.0, 0.5, 1.0], [(0, 0, 255), (255, 255, 255), (255, 0, 0)])
