"""Byte transports and the base class for serial instruments.

Drivers never use pyserial directly; they talk to a :class:`Transport`. This
keeps the protocol code testable without hardware (see :class:`FakeTransport`)
and keeps pyserial an optional dependency that is only imported on connect.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from .base import Device, DeviceError, NotConnectedError


@runtime_checkable
class Transport(Protocol):
    """Minimal byte-stream interface used by the drivers."""

    @property
    def is_open(self) -> bool: ...

    def open(self) -> None: ...

    def close(self) -> None: ...

    def write(self, data: bytes) -> None: ...

    def read(self, size: int) -> bytes:
        """Read up to ``size`` bytes (fewer if the timeout expires)."""
        ...

    def read_until(self, terminator: bytes = b"\n") -> bytes:
        """Read until ``terminator`` (included) or until the timeout expires."""
        ...

    def reset_input_buffer(self) -> None: ...


class SerialTransport:
    """A :class:`Transport` backed by pyserial."""

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        *,
        bytesize: int = 8,
        parity: str = "N",
        stopbits: float = 1,
        timeout: float | None = 0.5,
        write_timeout: float | None = None,
        xonxoff: bool = False,
        rtscts: bool = False,
        rs485: bool = False,
    ) -> None:
        self.port = port
        self._options: dict[str, Any] = {
            "baudrate": baudrate,
            "bytesize": bytesize,
            "parity": parity,
            "stopbits": stopbits,
            "timeout": timeout,
            "write_timeout": write_timeout,
            "xonxoff": xonxoff,
            "rtscts": rtscts,
        }
        self._rs485 = rs485
        self._serial: Any = None

    @property
    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def open(self) -> None:
        if self.is_open:
            return
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise DeviceError("pyserial is required for serial instruments (pip install pyserial)") from exc
        try:
            self._serial = serial.Serial(self.port, **self._options)
        except serial.SerialException as exc:
            raise DeviceError(f"cannot open {self.port}: {exc}") from exc
        if self._rs485:
            import serial.rs485

            self._serial.rs485_mode = serial.rs485.RS485Settings()

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _port(self) -> Any:
        if not self.is_open:
            raise NotConnectedError(f"serial port {self.port} is not open")
        return self._serial

    def write(self, data: bytes) -> None:
        port = self._port()
        port.write(data)
        port.flush()

    def read(self, size: int) -> bytes:
        return bytes(self._port().read(size))

    def read_until(self, terminator: bytes = b"\n") -> bytes:
        return bytes(self._port().read_until(terminator))

    def reset_input_buffer(self) -> None:
        self._port().reset_input_buffer()


Responder = Callable[[bytes], "bytes | None"]


class FakeTransport:
    """In-memory transport used by the tests and for offline development.

    ``responder`` is called with every chunk written and returns the bytes the
    simulated instrument answers with (``None`` for no answer). A mapping from
    written bytes to reply bytes is accepted as a shortcut.
    """

    def __init__(self, responder: Responder | Mapping[bytes, bytes] | None = None) -> None:
        if isinstance(responder, Mapping):
            table = dict(responder)
            self._responder: Responder = table.get
        else:
            self._responder = responder or (lambda data: None)
        self._rx = bytearray()
        self._open = False
        self.written: list[bytes] = []

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def write(self, data: bytes) -> None:
        if not self._open:
            raise NotConnectedError("fake transport is closed")
        data = bytes(data)
        self.written.append(data)
        reply = self._responder(data)
        if reply:
            self._rx.extend(reply)

    def read(self, size: int) -> bytes:
        chunk = bytes(self._rx[:size])
        del self._rx[:size]
        return chunk

    def read_until(self, terminator: bytes = b"\n") -> bytes:
        index = self._rx.find(terminator)
        end = len(self._rx) if index < 0 else index + len(terminator)
        chunk = bytes(self._rx[:end])
        del self._rx[:end]
        return chunk

    def reset_input_buffer(self) -> None:
        self._rx.clear()

    def feed(self, data: bytes) -> None:
        """Queue unsolicited bytes, as if the instrument had sent them."""
        self._rx.extend(data)


class SerialInstrument(Device):
    """Base class for instruments reachable through a serial (COM) port.

    Subclasses set ``serial_defaults`` (baud rate, handshake...) and the line
    terminators, then implement their commands on top of :meth:`write`,
    :meth:`query` or, for binary protocols, ``self.transport`` directly.
    Keyword arguments given at construction time (typically from the YAML
    configuration) override ``serial_defaults``.
    """

    serial_defaults: dict[str, Any] = {"baudrate": 9600, "timeout": 0.5}
    write_termination: str = "\r\n"
    read_termination: bytes = b"\n"
    encoding: str = "ascii"

    def __init__(
        self,
        port: str | None = None,
        *,
        name: str | None = None,
        transport: Transport | None = None,
        **serial_options: Any,
    ) -> None:
        super().__init__(name)
        if transport is None:
            if port is None:
                raise ValueError(f"{type(self).__name__} needs a serial 'port' (e.g. 'COM3')")
            transport = SerialTransport(port, **{**self.serial_defaults, **serial_options})
        elif serial_options:
            raise TypeError("serial options cannot be combined with an explicit transport")
        self.port = port
        self.transport = transport

    # ----------------------------------------------------------------- lifecycle
    def _connect(self) -> None:
        self.transport.open()
        try:
            self._initialize()
        except Exception:
            self.transport.close()
            raise

    def _disconnect(self) -> None:
        try:
            self._finalize()
        finally:
            self.transport.close()

    def _initialize(self) -> None:
        """Hook called right after the port opens (identification, remote mode...)."""

    def _finalize(self) -> None:
        """Hook called right before the port closes (back to local mode...)."""

    # --------------------------------------------------------------- text I/O
    def write(self, command: str) -> None:
        with self.lock:
            self.transport.reset_input_buffer()
            self.transport.write(f"{command}{self.write_termination}".encode(self.encoding))

    def read_line(self) -> str:
        raw = self.transport.read_until(self.read_termination)
        return raw.decode(self.encoding, errors="replace").strip()

    def query(self, command: str) -> str:
        with self.lock:
            self.write(command)
            reply = self.read_line()
        if not reply:
            raise DeviceError(f"{self.name}: no reply to {command!r}")
        self.log.debug("%s -> %s", command, reply)
        return reply

    def query_float(self, command: str) -> float:
        return parse_float(self.query(command))


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_float(text: str) -> float:
    """Return the last number in an instrument reply, e.g. ``"VM=12.30"`` -> ``12.3``."""
    matches = _NUMBER.findall(text)
    if not matches:
        raise DeviceError(f"no number in reply {text!r}")
    return float(matches[-1])
