"""Mini-Circuits USB RF instruments: PWR-4GHS power sensor and RCDAT-6G-120H attenuator.

Both use 64-byte interrupt packets. A reply echoes the command code in byte 0
followed by a NUL-terminated ASCII string.
"""

from __future__ import annotations

from ..base import Reading, Setting
from ..interfaces import RFAttenuator, RFPowerSensor
from ..transport import parse_float
from ..usb_hid import PacketLink, UsbInterruptDevice

VENDOR_ID = 0x20CE


def reply_text(reply: bytes) -> str:
    end = reply.find(b"\0", 1)
    return reply[1 : end if end > 0 else len(reply)].decode("ascii", errors="replace").strip()


class _MiniCircuitsMixin:
    """Connection handling shared by the Mini-Circuits devices."""

    product_id: int
    model_command: int
    serial_command: int

    def _init_link(self, index: int, serial_number: str | None, link: PacketLink | None) -> None:
        self.index = index
        self.serial_number = None if serial_number is None else str(serial_number)
        self._link = link

    def _connect(self) -> None:
        if self._link is None:
            match = None
            if self.serial_number is not None:
                wanted = self.serial_number

                def match(device: UsbInterruptDevice) -> bool:
                    return reply_text(device.exchange(bytes([self.serial_command]))) == wanted

            self._link = UsbInterruptDevice(VENDOR_ID, self.product_id, index=self.index, match=match)
        self._link.open()

    def _disconnect(self) -> None:
        self._link.close()

    def _ask(self, *payload: int) -> str:
        with self.lock:
            return reply_text(self._link.exchange(bytes(payload)))

    def serial(self) -> str:
        return self._ask(self.serial_command)

    def model_name(self) -> str:
        return self._ask(self.model_command)

    def identify(self) -> str:
        return f"{self.model_name()} S/N {self.serial()}"


class MiniCircuitsPowerSensor(_MiniCircuitsMixin, RFPowerSensor):
    """PWR-4GHS power sensor. With ``coupling_db`` set, the line power is also reported."""

    model = "Mini-Circuits PWR-4GHS"
    product_id = 0x0011
    model_command = 104
    serial_command = 105
    GET_FIRMWARE = 99
    SET_MEASUREMENT_MODE = 15
    GET_TEMPERATURE = 103
    GET_POWER = 102

    def __init__(
        self,
        *,
        name: str | None = None,
        index: int = 0,
        serial_number: str | None = None,
        frequency_mhz: float = 3000.0,
        coupling_db: float = 0.0,
        link: PacketLink | None = None,
    ) -> None:
        super().__init__(name)
        self._init_link(index, serial_number, link)
        self.frequency_mhz = frequency_mhz
        self.coupling_db = coupling_db

    def power_dbm(self, frequency_mhz: float | None = None) -> float:
        frequency = int(round(self.frequency_mhz if frequency_mhz is None else frequency_mhz))
        return parse_float(self._ask(self.GET_POWER, frequency // 256, frequency % 256, ord("M")))

    def temperature(self) -> float:
        return parse_float(self._ask(self.GET_TEMPERATURE))

    def firmware(self) -> str:
        return self._ask(self.GET_FIRMWARE)

    def set_fast_mode(self, fast: bool) -> None:
        """Fast sampling instead of the default low-noise averaging."""
        self._ask(self.SET_MEASUREMENT_MODE, 1 if fast else 0)

    def set_frequency(self, frequency_mhz: float) -> None:
        self.frequency_mhz = float(frequency_mhz)

    def set_coupling(self, coupling_db: float) -> None:
        self.coupling_db = float(coupling_db)

    def read_status(self) -> dict[str, Reading]:
        with self.lock:
            status = super().read_status()
            status["Temperature"] = Reading(self.temperature(), "°C")
        return status

    def settings(self) -> list[Setting]:
        return [
            Setting("frequency", "Frequency", self.set_frequency, unit="MHz",
                    minimum=1, maximum=6000, decimals=1, initial=self.frequency_mhz),
            Setting("coupling", "Coupling", self.set_coupling, unit="dB",
                    minimum=0, maximum=60, decimals=2, initial=self.coupling_db),
        ]


class MiniCircuitsAttenuator(_MiniCircuitsMixin, RFAttenuator):
    """RCDAT-6G-120H programmable attenuator (0-120 dB in 0.25 dB steps)."""

    model = "Mini-Circuits RCDAT-6G-120H"
    product_id = 0x0023
    model_command = 40
    serial_command = 41
    SEND_SCPI = 1

    def __init__(
        self,
        *,
        name: str | None = None,
        index: int = 0,
        serial_number: str | None = None,
        link: PacketLink | None = None,
    ) -> None:
        super().__init__(name)
        self._init_link(index, serial_number, link)

    def scpi(self, command: str) -> str:
        with self.lock:
            return reply_text(self._link.exchange(bytes([self.SEND_SCPI]) + command.encode("ascii")))

    def attenuation(self) -> float:
        return parse_float(self.scpi(":ATT?"))

    def set_attenuation(self, attenuation_db: float) -> None:
        if not 0 <= attenuation_db <= 120:
            raise ValueError("attenuation must be 0-120 dB")
        self.scpi(f":SETATT={attenuation_db:g}")

    def read_status(self) -> dict[str, Reading]:
        return {"Attenuation": Reading(self.attenuation(), "dB")}

    def settings(self) -> list[Setting]:
        return [
            Setting("attenuation", "Attenuation", self.set_attenuation, unit="dB",
                    minimum=0, maximum=120, decimals=2, initial=self.attenuation()),
        ]
