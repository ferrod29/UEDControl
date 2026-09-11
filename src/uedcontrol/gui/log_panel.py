"""Logging for the application: console, rotating log file and the log panel.

This replaces the ad-hoc ``print(..., file=log_file)`` calls of the original
GUIs: every module logs through :mod:`logging`, and the records end up in
``<log_dir>/uedcontrol.log`` and in the log dock of the main window.
"""

from __future__ import annotations

import contextlib
import html
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .qt import QtCore, QtGui, QtWidgets, Signal, Slot

_COLOURS = {
    logging.DEBUG: "#8a8a8a",
    logging.INFO: "#e0e0e0",
    logging.WARNING: "#ffb347",
    logging.ERROR: "#ff6b6b",
    logging.CRITICAL: "#ff3030",
}
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_installed: list[logging.Handler] = []


class _Emitter(QtCore.QObject):
    message = Signal(str, int)


class QtLogHandler(logging.Handler):
    """Forwards log records (from any thread) to the GUI thread through a signal."""

    def __init__(self) -> None:
        super().__init__()
        self.emitter = _Emitter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
        except Exception:
            self.handleError(record)
            return
        with contextlib.suppress(RuntimeError):  # the GUI is being torn down
            self.emitter.message.emit(text, record.levelno)


class LogPanel(QtWidgets.QPlainTextEdit):
    def __init__(self, parent: QtWidgets.QWidget | None = None, max_lines: int = 5000) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_lines)
        self.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))

    @Slot(str, int)
    def append_message(self, text: str, level: int) -> None:
        colour = _COLOURS.get(level, "#ff6b6b" if level >= logging.WARNING else "#e0e0e0")
        self.appendHtml(f'<span style="color:{colour}; white-space:pre">{html.escape(text)}</span>')


def setup_logging(log_dir: Path | None, level: str | int = "INFO") -> QtLogHandler:
    """Configure the ``uedcontrol`` logger; safe to call more than once."""
    logger = logging.getLogger("uedcontrol")
    for handler in _installed:
        logger.removeHandler(handler)
        handler.close()
    _installed.clear()
    logger.setLevel(level)
    logger.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FILE_FORMAT, "%H:%M:%S"))
    _installed.append(console)
    qt_handler = QtLogHandler()
    qt_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    _installed.append(qt_handler)
    file_error = None
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_dir / "uedcontrol.log", maxBytes=5_000_000, backupCount=10, encoding="utf-8"
            )
            file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, "%Y-%m-%d %H:%M:%S"))
            _installed.append(file_handler)
        except OSError as exc:
            file_error = exc
    for handler in _installed:
        logger.addHandler(handler)
    if file_error is not None:
        logger.warning("cannot write the log file in %s: %s", log_dir, file_error)
    return qt_handler
