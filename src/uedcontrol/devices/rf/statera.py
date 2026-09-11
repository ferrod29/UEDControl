"""DrXWorks Statera 100 solid-state RF amplifier (drives the compression cavity).

8-byte USB interrupt packets: a three-letter parameter code, ``=`` (set) or
``?`` (get), and a little-endian float32 value. Replies carry a two-character
status in bytes 0-1 and the float value in bytes 4-7.
"""

from __future__ import annotations

import struct

from ..base import Action, Reading, Setting
from ..interfaces import RFAmplifier
from ..usb_hid import PacketLink, UsbInterruptDevice

VENDOR_ID = 0x1AB2
PRODUCT_ID = 0x3CD4
SET = 0x3D  # "="
GET = 0x3F  # "?"


class DrXWorksStatera100(RFAmplifier):
    model = "DrXWorks Statera 100"

    def __init__(self, *, name: str | None = None, index: int = 0, link: PacketLink | None = None) -> None:
        super().__init__(name)
        self.index = index
        self._link = link

    def _connect(self) -> None:
        if self._link is None:
            self._link = UsbInterruptDevice(VENDOR_ID, PRODUCT_ID, index=self.index, packet_size=8)
        self._link.open()

    def _disconnect(self) -> None:
        self._link.close()

    def _transaction(self, code: str, operation: int, value: float = 0.0) -> tuple[str, float]:
        packet = code.encode("ascii") + bytes([operation]) + struct.pack("<f", round(float(value), 2))
        with self.lock:
            reply = self._link.exchange(packet, response_size=8)
        status = reply[:2].decode("ascii", errors="replace")
        result = struct.unpack("<f", reply[4:8])[0] if operation == GET and len(reply) >= 8 else 0.0
        return status, result

    def _get(self, code: str) -> float:
        return self._transaction(code, GET)[1]

    def _set(self, code: str, value: float) -> str:
        return self._transaction(code, SET, value)[0]

    # ------------------------------------------------------------ RFAmplifier
    def rf_enabled(self) -> bool:
        return bool(round(self._get("Swi")))

    def set_rf_enabled(self, enabled: bool) -> None:
        self._set("Swi", int(enabled))

    # ------------------------------------------------------------- parameters
    def is_remote(self) -> bool:
        return bool(round(self._get("Rem")))

    def set_remote(self, remote: bool) -> None:
        self._set("Rem", int(remote))

    def temperature(self) -> float:
        return self._get("Tem")

    def power_setpoint(self) -> float:
        return self._get("Pow")

    def set_power(self, value: float) -> None:
        self._set("Pow", value)

    def phase(self) -> float:
        return self._get("Pha")

    def set_phase(self, degrees: float) -> None:
        self._set("Pha", degrees)

    def transmitted_power(self) -> float:
        return self._get("Tra")

    def reflected_power(self) -> float:
        return self._get("Ref")

    def set_phase_feedback(self, enabled: bool) -> None:
        self._set("Pfe", int(enabled))

    def set_amplitude_feedback(self, enabled: bool) -> None:
        self._set("Afe", int(enabled))

    def read_status(self) -> dict[str, Reading]:
        with self.lock:
            return {
                "RF output": Reading(self.rf_enabled()),
                "Remote": Reading(self.is_remote()),
                "Cavity temperature": Reading(self.temperature(), "°C"),
                "Transmitted": Reading(self.transmitted_power()),
                "Reflected": Reading(self.reflected_power()),
            }

    def settings(self) -> list[Setting]:
        with self.lock:
            return [
                Setting("power", "Power set point", self.set_power, minimum=0, maximum=100,
                        decimals=2, initial=self.power_setpoint()),
                Setting("phase", "Phase", self.set_phase, unit="°", minimum=-360, maximum=360,
                        decimals=2, initial=self.phase()),
            ]

    def actions(self) -> list[Action]:
        return [
            Action("RF on", lambda: self.set_rf_enabled(True), confirm="Switch the RF amplifier on?"),
            Action("RF off", lambda: self.set_rf_enabled(False)),
            Action("Remote", lambda: self.set_remote(True)),
            Action("Local", lambda: self.set_remote(False)),
            Action("Phase feedback on", lambda: self.set_phase_feedback(True)),
            Action("Phase feedback off", lambda: self.set_phase_feedback(False)),
            Action("Amplitude feedback on", lambda: self.set_amplitude_feedback(True)),
            Action("Amplitude feedback off", lambda: self.set_amplitude_feedback(False)),
        ]
