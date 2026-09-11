"""Single import point for Qt.

Everything uses the binding pyqtgraph selected (PyQt5, PyQt6 or PySide6; set
``PYQTGRAPH_QT_LIB`` to force one), with fully qualified enum names so that
the code runs unchanged on all of them.
"""

from pyqtgraph.Qt import QT_LIB, QtCore, QtGui, QtWidgets

Signal = QtCore.Signal
Slot = QtCore.Slot

__all__ = ["QT_LIB", "QtCore", "QtGui", "QtWidgets", "Signal", "Slot", "exec_app"]


def exec_app(app: QtWidgets.QApplication) -> int:
    run = getattr(app, "exec", None) or app.exec_
    return int(run())
