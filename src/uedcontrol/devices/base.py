"""Base classes shared by every instrument driver.

A driver subclasses :class:`Device` (usually through one of the category
interfaces in :mod:`uedcontrol.devices.interfaces`) and implements
``_connect`` / ``_disconnect``. The public :meth:`Device.connect` and
:meth:`Device.disconnect` wrappers do the state bookkeeping, logging and
locking, so every driver behaves the same way.

Thread safety
-------------
Every driver method that talks to hardware holds ``self.lock`` for the whole
command sequence (e.g. "select channel 2, then read its voltage"). The lock is
re-entrant, so methods may call each other freely. This is what makes it safe
to poll a device from a GUI worker thread while a pump-probe scan commands the
same device from the acquisition thread.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


class DeviceError(RuntimeError):
    """An instrument reported an error or replied with something unexpected."""


class NotConnectedError(DeviceError):
    """A command was issued to a device that is not connected."""


@dataclass(frozen=True)
class Reading:
    """A single value reported by a device, together with its unit."""

    value: Any
    unit: str = ""

    def formatted(self, precision: int = 4) -> str:
        if isinstance(self.value, bool):
            text = "ON" if self.value else "OFF"
        elif isinstance(self.value, float):
            text = f"{self.value:.{precision}g}"
        else:
            text = str(self.value)
        return f"{text} {self.unit}".strip()


@dataclass
class Setting:
    """A user-adjustable parameter that the generic GUI panel can render."""

    key: str
    label: str
    apply: Callable[[Any], None]
    kind: str = "float"  # "float", "int", "choice" or "text"
    unit: str = ""
    minimum: float = -1e9
    maximum: float = 1e9
    decimals: int = 3
    choices: Sequence[str] = ()
    initial: Any = None


@dataclass
class Action:
    """A command button in the generic GUI panel."""

    label: str
    callback: Callable[[], Any]
    confirm: str | None = None  # if set, the GUI asks this question before running


class Device(ABC):
    """Lifecycle, locking and introspection common to all instruments."""

    #: Human readable model name; subclasses override it.
    model: str = "Device"

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.model
        self.lock = threading.RLock()
        self.log = logging.getLogger(f"uedcontrol.devices.{self.name}")
        self._connected = False

    # ------------------------------------------------------------------ lifecycle
    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        with self.lock:
            if self._connected:
                return
            self._connect()
            self._connected = True
        self.log.info("connected (%s)", self.model)

    def disconnect(self) -> None:
        with self.lock:
            if not self._connected:
                return
            try:
                self._disconnect()
            finally:
                self._connected = False
        self.log.info("disconnected")

    def ensure_connected(self) -> None:
        if not self._connected:
            raise NotConnectedError(f"{self.name} is not connected")

    @abstractmethod
    def _connect(self) -> None:
        """Open the communication channel and bring the instrument to a known state."""

    @abstractmethod
    def _disconnect(self) -> None:
        """Release the communication channel."""

    # -------------------------------------------------------------- introspection
    def identify(self) -> str:
        """Identification string (model, serial number, firmware...)."""
        return self.model

    def read_status(self) -> dict[str, Reading]:
        """Values shown by monitoring panels. Called periodically from a worker thread."""
        return {}

    def settings(self) -> list[Setting]:
        """Parameters the generic panel exposes as editable fields."""
        return []

    def actions(self) -> list[Action]:
        """Commands the generic panel exposes as buttons."""
        return []

    # ------------------------------------------------------------------- helpers
    def __enter__(self) -> Device:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r} connected={self._connected}>"
