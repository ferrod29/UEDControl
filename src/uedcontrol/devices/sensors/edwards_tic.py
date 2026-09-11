"""Edwards TIC turbo and instrument controller (TIC3 / TIC6): gauges and turbo pump.

ASCII protocol, CR terminated. ``?V<object>`` returns
``=V<object> field;field;...``; commands ``!C<object> <value>`` return
``*C<object> <error code>``. Pressures come in pascal and are reported here
in mbar.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..base import Action, DeviceError, Reading
from ..transport import SerialInstrument, Transport

GAUGE_OBJECTS = {1: 913, 2: 914, 3: 915, 4: 934, 5: 935, 6: 936}
TIC_STATUS = 902
TURBO_PUMP = 904
TURBO_SPEED = 905
TURBO_POWER = 906

UNITS = {"59": "Pa", "66": "V", "81": "%"}

PUMP_STATES = {
    "0": "Stopped",
    "1": "Starting delay",
    "2": "Stopping short delay",
    "3": "Stopping normal delay",
    "4": "Running",
    "5": "Accelerating",
    "6": "Fault braking",
    "7": "Braking",
}

GAUGE_STATES = {
    "0": "Not connected",
    "1": "Connected",
    "2": "New ID",
    "3": "Change",
    "4": "In alert",
    "5": "Off",
    "6": "Striking",
    "7": "Initialising",
    "8": "Calibrating",
    "9": "Zeroing",
    "10": "Degassing",
    "11": "On",
    "12": "Inhibited",
}

GAUGE_TYPES = {
    "0": "Unknown", "1": "No device", "2": "EXP_CM", "3": "EXP_STD", "4": "CMAN_S",
    "5": "CMAN_D", "6": "TURBO", "7": "APGM", "8": "APGL", "9": "APGXM", "10": "APGXH",
    "11": "APGXL", "12": "ATCA", "13": "ATCD", "14": "ATCM", "15": "WRG", "16": "AIMC",
    "17": "AIMN", "18": "AIMS", "19": "AIMX", "20": "AIGC_I2R", "21": "AIGC_2FIL",
    "22": "ION_EB", "23": "AIGXS", "24": "USER", "25": "ASG",
}

GAUGE_COMMANDS = {"off": 0, "on": 1, "new_id": 2, "zero": 3, "calibrate": 4, "degas": 5}

PRIORITIES = {"0": "OK", "1": "Warning", "2": "Alarm", "3": "Alarm"}

ERRORS = {
    "1": "invalid command for object ID",
    "2": "invalid query/command",
    "3": "missing parameter",
    "4": "parameter out of range",
    "5": "invalid command in current state",
    "6": "data checksum error",
    "7": "EEPROM read or write error",
    "8": "operation took too long",
    "9": "invalid config ID",
}

ALERTS = {
    "0": "No alert", "1": "ADC fault", "2": "ADC not ready", "3": "Over range", "4": "Under range",
    "5": "ADC invalid", "6": "No gauge", "7": "Unknown", "8": "Not supported", "9": "New ID",
    "10": "Over range", "11": "Under range", "12": "Over range", "13": "Ion emission timeout",
    "14": "Not struck", "15": "Filament fail", "16": "Mag fail", "17": "Striker fail",
    "18": "Not struck", "19": "Filament fail", "20": "Calibration error", "21": "Initialising",
    "22": "Emission error", "23": "Over pressure", "24": "ASG cannot zero", "25": "Ramp-up timeout",
    "26": "Droop timeout", "27": "Run hours high", "28": "SC interlock", "29": "ID volts error",
    "30": "Serial ID fail", "31": "Upload active", "32": "DX fault", "33": "Temperature alert",
    "34": "SYSI inhibit", "35": "External inhibit", "36": "Temperature inhibit", "37": "No reading",
    "38": "Message", "39": "NOV failure", "40": "Upload timeout", "41": "Download failed",
    "42": "No tube", "43": "Use gauges 4-6", "44": "Degas inhibited", "45": "IGC inhibited",
    "46": "Brownout/short", "47": "Service due",
}


@dataclass(frozen=True)
class GaugeReading:
    value: float
    unit: str
    state: str
    alert: str
    priority: str

    @property
    def pressure_mbar(self) -> float | None:
        return self.value * 0.01 if self.unit == "Pa" else None


class EdwardsTIC(SerialInstrument):
    model = "Edwards TIC"
    serial_defaults = {"baudrate": 9600, "timeout": 0.5}
    write_termination = "\r"
    read_termination = b"\r"

    def __init__(
        self,
        port: str | None = None,
        *,
        name: str | None = None,
        transport: Transport | None = None,
        gauges: Sequence[int] = (1, 2, 3),
        has_turbo: bool = True,
        **serial_options: Any,
    ) -> None:
        super().__init__(port, name=name, transport=transport, **serial_options)
        unknown = set(gauges) - set(GAUGE_OBJECTS)
        if unknown:
            raise ValueError(f"gauges must be among 1..6, got {sorted(unknown)}")
        self.gauges = tuple(gauges)
        self.has_turbo = has_turbo

    def _transact(self, message: str) -> str:
        reply = self.query(message)
        kind = reply[:1]
        if kind == "*":
            code = reply.partition(" ")[2].strip() or "0"
            if code != "0":
                raise DeviceError(f"{self.name}: {message!r} failed: {ERRORS.get(code, code)}")
            return ""
        if kind != "=":
            raise DeviceError(f"{self.name}: unexpected reply {reply!r}")
        return reply[len(message):].lstrip(" ;")

    def _fields(self, message: str) -> list[str]:
        return self._transact(message).split(";")

    # ------------------------------------------------------------------ gauges
    def gauge(self, number: int) -> GaugeReading:
        fields = self._fields(f"?V{GAUGE_OBJECTS[number]}")
        return GaugeReading(
            value=float(fields[0]),
            unit=UNITS.get(fields[1], fields[1]),
            state=GAUGE_STATES.get(fields[2], fields[2]),
            alert=ALERTS.get(fields[3], fields[3]),
            priority=PRIORITIES.get(fields[4], fields[4]),
        )

    def pressure(self, number: int) -> float:
        reading = self.gauge(number)
        if reading.pressure_mbar is None:
            raise DeviceError(f"gauge {number} reports {reading.unit}, not a pressure")
        return reading.pressure_mbar

    def gauge_command(self, number: int, command: str) -> None:
        self._transact(f"!C{GAUGE_OBJECTS[number]} {GAUGE_COMMANDS[command]}")

    def gauge_type(self, number: int) -> str:
        code = self._fields(f"?S{GAUGE_OBJECTS[number]} 5")[0]
        return GAUGE_TYPES.get(code, code)

    def gauge_name(self, number: int) -> str:
        return self._fields(f"?S{GAUGE_OBJECTS[number]} 68")[0]

    # ------------------------------------------------------------- turbo pump
    def turbo_state(self) -> str:
        return PUMP_STATES.get(self._fields(f"?V{TURBO_PUMP}")[0], "Unknown")

    def set_turbo(self, running: bool) -> None:
        self._transact(f"!C{TURBO_PUMP} {int(running)}")

    def turbo_speed(self) -> float:
        return float(self._fields(f"?V{TURBO_SPEED}")[0])

    def turbo_power(self) -> float:
        return float(self._fields(f"?V{TURBO_POWER}")[0])

    # ------------------------------------------------------------ GUI hooks
    def read_status(self) -> dict[str, Reading]:
        status: dict[str, Reading] = {}
        with self.lock:
            for number in self.gauges:
                reading = self.gauge(number)
                if reading.pressure_mbar is not None:
                    status[f"Gauge {number}"] = Reading(reading.pressure_mbar, "mbar")
                else:
                    status[f"Gauge {number}"] = Reading(reading.value, reading.unit)
                status[f"Gauge {number} state"] = Reading(reading.state)
            if self.has_turbo:
                status["Turbo"] = Reading(self.turbo_state())
                status["Turbo speed"] = Reading(self.turbo_speed(), "%")
                status["Turbo power"] = Reading(self.turbo_power(), "W")
        return status

    def actions(self) -> list[Action]:
        actions = []
        if self.has_turbo:
            actions += [
                Action("Start turbo", lambda: self.set_turbo(True), confirm="Start the turbo pump?"),
                Action("Stop turbo", lambda: self.set_turbo(False), confirm="Stop the turbo pump?"),
            ]
        for number in self.gauges:
            actions += [
                Action(f"Gauge {number} on", lambda n=number: self.gauge_command(n, "on")),
                Action(f"Gauge {number} off", lambda n=number: self.gauge_command(n, "off")),
            ]
        return actions
