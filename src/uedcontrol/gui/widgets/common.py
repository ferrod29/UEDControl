"""Small reusable widgets and helpers."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .. import theme
from ..qt import QtCore, QtGui, QtWidgets

_RIGHT = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter


class ToggleButton(QtWidgets.QPushButton):
    """Checkable button whose text and colour show its state (cyan on, orange off).

    Connect user actions to ``clicked``. ``setChecked``, used to mirror the
    instrument state, does not emit ``clicked``.
    """

    def __init__(self, on_text: str, off_text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(off_text, parent)
        self.on_text = on_text
        self.off_text = off_text
        self.setCheckable(True)
        self.toggled.connect(self._refresh)
        self._refresh(False)

    def _refresh(self, checked: bool) -> None:
        self.setText(self.on_text if checked else self.off_text)
        colour = theme.ON_COLOR if checked else theme.OFF_COLOR
        self.setStyleSheet(f"QPushButton {{ color: {colour}; font-weight: bold; }}")


def spin(
    value: float = 0.0,
    minimum: float = -1e9,
    maximum: float = 1e9,
    decimals: int = 3,
    step: float = 1.0,
    suffix: str = "",
) -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setKeyboardTracking(False)
    box.setAlignment(_RIGHT)
    if suffix:
        box.setSuffix(f" {suffix}")
    return box


def int_spin(
    value: int = 0, minimum: int = 0, maximum: int = 1_000_000, step: int = 1, suffix: str = ""
) -> QtWidgets.QSpinBox:
    box = QtWidgets.QSpinBox()
    box.setRange(int(minimum), int(maximum))
    box.setSingleStep(step)
    box.setValue(int(value))
    box.setKeyboardTracking(False)
    box.setAlignment(_RIGHT)
    if suffix:
        box.setSuffix(f" {suffix}")
    return box


class Readout(QtWidgets.QLabel):
    """Right-aligned, selectable label for a measured value."""

    def __init__(self, text: str = "–", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setAlignment(_RIGHT)
        self.setMinimumWidth(80)
        self.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))

    def show_value(self, value: Any, fmt: str = "{:.4g}") -> None:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            self.setText("–")
        else:
            self.setText(fmt.format(value))


def app_settings(name: str) -> QtCore.QSettings:
    """Persistent GUI settings (window layout, last calibration...), stored as an INI file
    in the user's configuration folder, e.g. ``%APPDATA%/uedcontrol/<name>.ini``."""
    return QtCore.QSettings(QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope, "uedcontrol", name)


def ask(parent: QtWidgets.QWidget | None, title: str, question: str) -> bool:
    buttons = QtWidgets.QMessageBox.StandardButton
    answer = QtWidgets.QMessageBox.question(parent, title, question, buttons.Yes | buttons.No, buttons.No)
    return answer == buttons.Yes


def warn(parent: QtWidgets.QWidget | None, title: str, message: str) -> None:
    QtWidgets.QMessageBox.warning(parent, title, message)


def scroll_area(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    area.setWidget(widget)
    return area


def clear_layout(layout: QtWidgets.QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


def save_file_name(parent: QtWidgets.QWidget | None, caption: str, default: str, filters: str) -> str | None:
    path, _selected = QtWidgets.QFileDialog.getSaveFileName(parent, caption, default, filters)
    return path or None


def open_file_name(parent: QtWidgets.QWidget | None, caption: str, filters: str) -> str | None:
    path, _selected = QtWidgets.QFileDialog.getOpenFileName(parent, caption, "", filters)
    return path or None


def write_csv(path: str | Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
