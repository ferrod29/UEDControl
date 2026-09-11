"""Minimal libusb1 wrapper for instruments that exchange fixed-size interrupt packets.

Used by the Mini-Circuits power sensors and attenuator and by the DrXWorks
Statera amplifier. libusb1 is imported only when a device is opened.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from .base import DeviceError, NotConnectedError


class PacketLink(Protocol):
    """What the USB drivers need; tests substitute a fake."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def exchange(self, payload: bytes, response_size: int | None = None) -> bytes: ...


class UsbInterruptDevice:
    """Open the ``index``-th device with the given VID/PID (or the first one for
    which ``match`` returns True) and talk to it through interrupt endpoint 1."""

    def __init__(
        self,
        vendor_id: int,
        product_id: int,
        *,
        index: int = 0,
        match: Callable[[UsbInterruptDevice], bool] | None = None,
        endpoint: int = 1,
        packet_size: int = 64,
        timeout_ms: int = 2000,
    ) -> None:
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.index = index
        self.match = match
        self.endpoint = endpoint
        self.packet_size = packet_size
        self.timeout_ms = timeout_ms
        self._context: Any = None
        self._handle: Any = None

    def open(self) -> None:
        try:
            import usb1
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise DeviceError("libusb1 is required for USB instruments (pip install libusb1)") from exc
        self._context = usb1.USBContext()
        candidates = [
            device
            for device in self._context.getDeviceIterator(skip_on_error=True)
            if device.getVendorID() == self.vendor_id and device.getProductID() == self.product_id
        ]
        label = f"{self.vendor_id:04X}:{self.product_id:04X}"
        try:
            if not candidates:
                raise DeviceError(f"no USB device {label} found")
            if self.match is None:
                if self.index >= len(candidates):
                    raise DeviceError(f"USB device {label} #{self.index} not found ({len(candidates)} present)")
                self._handle = self._claim(candidates[self.index])
                return
            for device in candidates:
                try:
                    self._handle = self._claim(device)
                except Exception:  # in use by another program
                    continue
                if self.match(self):
                    return
                self._release()
            raise DeviceError(f"no USB device {label} matches the requested serial number")
        except Exception:
            self.close()
            raise

    @staticmethod
    def _claim(device: Any) -> Any:
        handle = device.open()
        try:
            if handle.kernelDriverActive(0):
                handle.detachKernelDriver(0)
        except Exception:  # not supported on Windows
            pass
        handle.claimInterface(0)
        return handle

    def _release(self) -> None:
        if self._handle is not None:
            try:
                self._handle.releaseInterface(0)
            finally:
                self._handle.close()
                self._handle = None

    def close(self) -> None:
        self._release()
        if self._context is not None:
            self._context.close()
            self._context = None

    def exchange(self, payload: bytes, response_size: int | None = None) -> bytes:
        if self._handle is None:
            raise NotConnectedError("USB device is not open")
        packet = bytes(payload).ljust(self.packet_size, b"\0")
        self._handle.interruptWrite(self.endpoint, packet, self.timeout_ms)
        return bytes(self._handle.interruptRead(self.endpoint, response_size or self.packet_size, self.timeout_ms))
