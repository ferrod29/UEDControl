"""attocube ECC100 piezo positioner controller, through the vendor DLL (ctypes).

Each :class:`AttocubeECC100Axis` drives one axis; axes of the same controller
share one connection. Positions are in mm (the DLL uses nm). Ported from
``UEDControlSystem/Devices/ecc100.py`` (itself derived from ScopeFoundry's
attocube driver). The DLL path, hard-coded before, is now a parameter.
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import byref, c_int32
from typing import Any

from ..base import Action, DeviceError, Reading, Setting
from ..interfaces import Positioner

NCB_ERRORS = {
    -1: "unspecified error",
    1: "communication timeout",
    2: "no active connection to the device",
    3: "driver error",
    7: "device is in use by another application",
    9: "parameter out of range",
    10: "feature only available in the Pro version",
}
ACTOR_TYPES = ("linear", "goniometer", "rotator")


def _check(code: int) -> None:
    if code != 0:
        raise DeviceError(f"ECC100 error {code}: {NCB_ERRORS.get(code, 'unknown error')}")


class _Controller:
    """An open controller, shared by all axes that use it."""

    _registry: dict[tuple[str, int], _Controller] = {}
    _registry_lock = threading.Lock()

    def __init__(self, key: tuple[str, int], dll: Any) -> None:
        self.key = key
        self.dll = dll
        self.handle = c_int32()
        self.users = 0
        self.lock = threading.RLock()

    @classmethod
    def acquire(cls, dll_path: str, device_number: int) -> _Controller:
        key = (dll_path, device_number)
        with cls._registry_lock:
            controller = cls._registry.get(key)
            if controller is None:
                try:
                    dll = ctypes.cdll.LoadLibrary(dll_path)
                except OSError as exc:
                    raise DeviceError(f"cannot load the ECC100 library {dll_path!r}: {exc}") from exc
                count = dll.ECC_Check(None)
                if device_number >= count:
                    raise DeviceError(f"ECC100 #{device_number} not found ({count} connected)")
                controller = cls(key, dll)
                _check(dll.ECC_Connect(device_number, byref(controller.handle)))
                cls._registry[key] = controller
            controller.users += 1
            return controller

    def release(self) -> None:
        with self._registry_lock:
            self.users -= 1
            if self.users <= 0:
                self._registry.pop(self.key, None)
                _check(self.dll.ECC_Close(self.handle))


class AttocubeECC100Axis(Positioner):
    model = "attocube ECC100"
    unit = "mm"

    def __init__(
        self,
        axis: int = 0,
        *,
        name: str | None = None,
        device_number: int = 0,
        dll_path: str = "ecc.dll",
        move_timeout: float = 60.0,
    ) -> None:
        super().__init__(name)
        self.axis = axis
        self.device_number = device_number
        self.dll_path = dll_path
        self.move_timeout = move_timeout
        self._controller: _Controller | None = None

    def _connect(self) -> None:
        self._controller = _Controller.acquire(self.dll_path, self.device_number)
        if not self._get("ECC_getStatusConnected"):
            self.log.warning("no actor is electrically connected to axis %d", self.axis)

    def _disconnect(self) -> None:
        if self._controller is not None:
            self._controller.release()
            self._controller = None

    # --------------------------------------------------------------- DLL calls
    def _call(self, function: str, *args: Any) -> None:
        controller = self._controller
        if controller is None:
            raise DeviceError(f"{self.name} is not connected")
        with controller.lock:
            _check(getattr(controller.dll, function)(controller.handle, self.axis, *args))

    def _get(self, function: str) -> int:
        value = c_int32()
        self._call(function, byref(value))
        return value.value

    def _control_get(self, function: str) -> int:
        value = c_int32()
        self._call(function, byref(value), 0)
        return value.value

    def _control_set(self, function: str, value: float) -> None:
        self._call(function, byref(c_int32(int(value))), 1)

    # ------------------------------------------------------------------ queries
    def actor_name(self) -> str:
        buffer = ctypes.create_string_buffer(20)
        self._call("ECC_getActorName", buffer)
        return buffer.value.decode("ascii", errors="replace").strip()

    def actor_type(self) -> str:
        return ACTOR_TYPES[self._get("ECC_getActorType")]

    def identify(self) -> str:
        return f"{self.model} axis {self.axis}: {self.actor_name()} ({self.actor_type()})"

    def target_reached(self) -> bool:
        return bool(self._get("ECC_getStatusTargetRange"))

    def reference_valid(self) -> bool:
        return bool(self._get("ECC_getStatusReference"))

    def amplitude(self) -> float:
        return self._control_get("ECC_controlAmplitude") * 1e-3  # mV -> V

    def set_amplitude(self, volts: float) -> None:
        self._control_set("ECC_controlAmplitude", round(volts * 1e3))

    def frequency(self) -> float:
        return self._control_get("ECC_controlFrequency") * 1e-3  # mHz -> Hz

    def set_frequency(self, hertz: float) -> None:
        self._control_set("ECC_controlFrequency", round(hertz * 1e3))

    def set_output_enabled(self, enabled: bool) -> None:
        self._control_set("ECC_controlOutput", int(enabled))

    # ------------------------------------------------------- Positioner interface
    def position(self) -> float:
        return self._get("ECC_getPosition") * 1e-6  # nm -> mm

    def move_to(self, position: float, wait: bool = True) -> None:
        with self.lock:
            self._control_set("ECC_controlTargetPosition", round(position * 1e6))
            # Closed-loop positioning must be (re)enabled for the move to start.
            self._control_set("ECC_controlMove", 1)
        if wait:
            self.wait_until_stopped()

    def is_moving(self) -> bool:
        return self._get("ECC_getStatusMoving") != 0

    def stop(self) -> None:
        with self.lock:
            self._control_set("ECC_controlMove", 0)
            self._control_set("ECC_controlContinousFwd", 0)
            self._control_set("ECC_controlContinousBkwd", 0)

    def single_step(self, forward: bool = True) -> None:
        self._call("ECC_setSingleStep", int(not forward))

    def jog(self, direction: int) -> None:
        if direction > 0:
            self._control_set("ECC_controlContinousFwd", 1)
        elif direction < 0:
            self._control_set("ECC_controlContinousBkwd", 1)
        else:
            self.stop()

    def reset_position(self) -> None:
        """Set the current position to zero and invalidate the reference."""
        self._call("ECC_setReset")

    def read_status(self) -> dict[str, Reading]:
        with self.lock:
            return {
                "Position": Reading(self.position(), self.unit),
                "Moving": Reading(self.is_moving()),
                "Target reached": Reading(self.target_reached()),
                "Reference valid": Reading(self.reference_valid()),
            }

    def settings(self) -> list[Setting]:
        return [
            Setting("amplitude", "Step amplitude", self.set_amplitude, unit="V",
                    minimum=0, maximum=60, decimals=1, initial=self.amplitude()),
            Setting("frequency", "Step frequency", self.set_frequency, unit="Hz",
                    minimum=1, maximum=5000, decimals=0, initial=self.frequency()),
        ]

    def actions(self) -> list[Action]:
        return [
            Action("Step +", lambda: self.single_step(True)),
            Action("Step -", lambda: self.single_step(False)),
            Action("Output on", lambda: self.set_output_enabled(True)),
            Action("Output off", lambda: self.set_output_enabled(False)),
            Action("Zero position", self.reset_position, confirm="Reset the position of this axis to zero?"),
        ]
