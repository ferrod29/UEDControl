"""Delay generator built by the MPSD electronics workshop (trigger/synchronisation box).

.. warning::
   **Unverified.** The original driver (``Devices/delaygenerator.py``) could not
   have worked: it wrote Python lists to the port and overwrote every value
   with a constant. The frame layout here is what that code was aiming at and
   matches the other MPSD box (:mod:`uedcontrol.devices.power.heinzinger`):
   a code byte, its bit complement and a 32-bit little-endian value. Times are
   converted with ``clock_hz`` (42 MHz in the original notes). The channel
   table is copied verbatim; note that ``E`` has the same code as ``A`` there.
   Check everything against the box documentation before use.
"""

from __future__ import annotations

from typing import Any

from ..base import Action, Reading, Setting
from ..transport import SerialInstrument, Transport

CHANNELS = {"A": 0x0, "B": 0x1, "C": 0x2, "D": 0x3, "E": 0x0, "F": 0x4}
CHANNEL_PARAMETERS = {
    "delay_mode": 0x0,
    "delay": 0x1,
    "width": 0x2,
    "duty_cycle": 0x3,
    "offset": 0x4,
    "output_mode": 0x5,
    "accept_changes": 0x6,
}
SIGNAL_COMMANDS = {
    "test": 0xF0,
    "trigger_mode": 0xF1,
    "pulses_to_count": 0xF2,
    "arm": 0xF3,
    "disarm": 0xF4,
    "force_trigger": 0xF5,
    "save_eeprom": 0xFA,
}
TRIGGER_MODES = {"off": 0, "on": 1, "gate": 2, "single": 3, "count": 4}
CHANNEL_MODES = {"off": 0, "normal": 1, "single": 2, "duty_cycle": 3}
OUTPUT_MODES = {"active_high": 0, "active_low": 1, "always_low": 2, "always_high": 3}
_NEEDS_ACCEPT = {"delay_mode", "delay", "width"}


def frame(code: int, value: int = 0) -> bytes:
    return bytes([code & 0xFF, ~code & 0xFF]) + (int(value) & 0xFFFFFFFF).to_bytes(4, "little")


class MPSDDelayGenerator(SerialInstrument):
    model = "MPSD delay generator"
    serial_defaults = {"baudrate": 38400, "timeout": 0.2}

    def __init__(
        self,
        port: str | None = None,
        *,
        name: str | None = None,
        transport: Transport | None = None,
        clock_hz: float = 42e6,
        **serial_options: Any,
    ) -> None:
        super().__init__(port, name=name, transport=transport, **serial_options)
        self.clock_hz = clock_hz
        self._trigger_mode = "off"

    def _exchange(self, code: int, value: int = 0) -> str:
        with self.lock:
            self.transport.reset_input_buffer()
            self.transport.write(frame(code, value))
            return self.read_line()

    def seconds_to_counts(self, seconds: float) -> int:
        return int(round(seconds * self.clock_hz))

    def set_channel_parameter(self, channel: str, parameter: str, value: int) -> None:
        channel_code = CHANNELS[channel.upper()] << 4
        with self.lock:
            self._exchange(channel_code | CHANNEL_PARAMETERS[parameter], value)
            if parameter in _NEEDS_ACCEPT:
                self._exchange(channel_code | CHANNEL_PARAMETERS["accept_changes"], 0)

    def set_delay(self, channel: str, seconds: float) -> None:
        self.set_channel_parameter(channel, "delay", self.seconds_to_counts(seconds))

    def set_width(self, channel: str, seconds: float) -> None:
        self.set_channel_parameter(channel, "width", self.seconds_to_counts(seconds))

    def set_channel_mode(self, channel: str, mode: str) -> None:
        self.set_channel_parameter(channel, "delay_mode", CHANNEL_MODES[mode])

    def set_output_mode(self, channel: str, mode: str) -> None:
        self.set_channel_parameter(channel, "output_mode", OUTPUT_MODES[mode])

    def set_trigger_mode(self, mode: str) -> None:
        self._exchange(SIGNAL_COMMANDS["trigger_mode"], TRIGGER_MODES[mode])
        self._trigger_mode = mode

    def set_pulses_to_count(self, pulses: int) -> None:
        self._exchange(SIGNAL_COMMANDS["pulses_to_count"], pulses)

    def arm(self) -> None:
        self._exchange(SIGNAL_COMMANDS["arm"], TRIGGER_MODES[self._trigger_mode])

    def disarm(self) -> None:
        self._exchange(SIGNAL_COMMANDS["disarm"], TRIGGER_MODES[self._trigger_mode])

    def force_trigger(self) -> None:
        self._exchange(SIGNAL_COMMANDS["force_trigger"])

    def save_eeprom(self) -> None:
        self._exchange(SIGNAL_COMMANDS["save_eeprom"])

    def read_status(self) -> dict[str, Reading]:
        return {"Trigger mode": Reading(self._trigger_mode)}

    def settings(self) -> list[Setting]:
        settings = [
            Setting("trigger_mode", "Trigger mode", self.set_trigger_mode, kind="choice",
                    choices=tuple(TRIGGER_MODES), initial=self._trigger_mode),
        ]
        for channel in "ABCD":
            settings += [
                Setting(f"delay_{channel}", f"Delay {channel}",
                        lambda us, ch=channel: self.set_delay(ch, us * 1e-6),
                        unit="µs", minimum=0, maximum=1e6, decimals=3),
                Setting(f"width_{channel}", f"Width {channel}",
                        lambda us, ch=channel: self.set_width(ch, us * 1e-6),
                        unit="µs", minimum=0, maximum=1e6, decimals=3),
            ]
        return settings

    def actions(self) -> list[Action]:
        return [
            Action("Arm", self.arm),
            Action("Disarm", self.disarm),
            Action("Force trigger", self.force_trigger),
            Action("Save to EEPROM", self.save_eeprom, confirm="Store the current settings in EEPROM?"),
        ]
