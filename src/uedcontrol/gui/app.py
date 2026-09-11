"""Command-line entry point: ``uedcontrol --config FILE`` or ``uedcontrol --simulate``.

Without ``--config`` the file named by the ``UEDCONTROL_CONFIG`` environment
variable is used, so a lab PC can start the right configuration from a plain
desktop shortcut.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

from .. import __version__
from ..config import ConfigError, InstrumentConfig, load_config, simulated_config

CONFIG_ENV = "UEDCONTROL_CONFIG"

log = logging.getLogger("uedcontrol")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="uedcontrol", description="Control software for the ultrafast electron diffraction instrument."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("-c", "--config", type=Path, help=f"instrument configuration (YAML); default: ${CONFIG_ENV}")
    source.add_argument("--simulate", action="store_true", help="run with simulated instruments (no hardware)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--version", action="version", version=f"uedcontrol {__version__}")
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> InstrumentConfig:
    if args.simulate:
        return simulated_config()
    path = args.config or os.environ.get(CONFIG_ENV)
    if not path:
        raise ConfigError(f"no configuration given: use --config FILE, set {CONFIG_ENV}, or start with --simulate")
    return load_config(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Qt is imported only now, so that --help and --version work without a display.
    from ..instrument import Instrument
    from . import theme
    from .log_panel import setup_logging
    from .main_window import MainWindow
    from .qt import QtCore, QtWidgets, exec_app

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("uedcontrol")
    app.setOrganizationName("uedcontrol")
    theme.apply_dark_theme(app)
    theme.configure_pyqtgraph()

    try:
        config = resolve_config(args)
        handler = setup_logging(config.log_dir, args.log_level)
        instrument = Instrument.from_config(config)
    except ConfigError as exc:
        setup_logging(None, args.log_level)
        log.error("invalid configuration: %s", exc)
        QtWidgets.QMessageBox.critical(None, "UED control", f"The configuration cannot be used:\n\n{exc}")
        return 2

    log.info("uedcontrol %s: %s, %d device(s)", __version__,
             config.source or "simulated instrument", len(instrument.devices))
    window = MainWindow(instrument, handler)
    window.show()

    # Let Ctrl+C in the console close the window cleanly: Python only handles
    # signals when it runs, so wake it up regularly.
    signal.signal(signal.SIGINT, lambda *_: window.close())
    heartbeat = QtCore.QTimer()
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start(250)
    try:
        return exec_app(app)
    finally:
        window.shutdown()
