"""PI Mercury C-663 stepper controller with an M-531 linear stage (main delay line).

Uses PI's ``pipython`` package (GCS commands), imported on connect.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from ..base import DeviceError, Reading
from ..interfaces import Positioner


class PIMercuryC663(Positioner):
    model = "PI Mercury C-663"
    unit = "mm"

    def __init__(
        self,
        serial_number: str | None = None,
        *,
        name: str | None = None,
        controller: str = "C-663",
        stages: Sequence[str] = ("M-531.2S1",),
        refmodes: str | Sequence[str] | None = None,
        rs232_port: int | None = None,
        baudrate: int = 115200,
        axis: str | None = None,
        move_timeout: float = 120.0,
    ) -> None:
        super().__init__(name)
        if serial_number is None and rs232_port is None:
            raise ValueError("give either the USB 'serial_number' or an 'rs232_port'")
        self.serial_number = serial_number
        self.controller = controller
        self.stages = list(stages) if stages else None
        self.refmodes = refmodes
        self.rs232_port = rs232_port
        self.baudrate = baudrate
        self.axis = axis
        self.move_timeout = move_timeout
        self._dev: Any = None
        self._pitools: Any = None
        self._idn = ""

    def _connect(self) -> None:
        try:
            from pipython import GCSDevice, pitools
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise DeviceError("PIPython is required for the PI stage (pip install PIPython)") from exc
        device = GCSDevice(self.controller)
        try:
            if self.rs232_port is not None:
                device.ConnectRS232(comport=self.rs232_port, baudrate=self.baudrate)
            else:
                device.ConnectUSB(serialnum=self.serial_number)
            pitools.startup(device, stages=self.stages, refmodes=self.refmodes)
        except Exception:
            device.CloseConnection()
            raise
        self._dev, self._pitools = device, pitools
        self.axis = self.axis or device.axes[0]
        self._idn = device.qIDN().strip()

    def _disconnect(self) -> None:
        self._dev.CloseConnection()
        self._dev = None

    def identify(self) -> str:
        return self._idn or self.model

    def position(self) -> float:
        with self.lock:
            return float(self._dev.qPOS(self.axis)[self.axis])

    def move_to(self, position: float, wait: bool = True) -> None:
        low, high = self.limits()
        if not low <= position <= high:
            raise DeviceError(f"{self.name}: target {position} mm outside {low}..{high} mm")
        with self.lock:
            self._dev.MOV(self.axis, position)
        if wait:
            self.wait_until_stopped()

    def is_moving(self) -> bool:
        with self.lock:
            return not self._dev.qONT(self.axis)[self.axis]

    def stop(self) -> None:
        with self.lock:
            self._pitools.stopall(self._dev)

    def home(self) -> None:
        with self.lock:
            self._dev.FRF(self.axis)
        deadline = time.monotonic() + self.move_timeout
        while True:
            with self.lock:
                if self._dev.qFRF(self.axis)[self.axis]:
                    return
            if time.monotonic() > deadline:
                raise DeviceError(f"{self.name}: referencing timed out")
            time.sleep(0.1)

    def limits(self) -> tuple[float, float]:
        with self.lock:
            return (
                float(self._dev.qTMN(self.axis)[self.axis]),
                float(self._dev.qTMX(self.axis)[self.axis]),
            )

    def read_status(self) -> dict[str, Reading]:
        with self.lock:
            return {
                "Position": Reading(self.position(), self.unit),
                "On target": Reading(not self.is_moving()),
            }
