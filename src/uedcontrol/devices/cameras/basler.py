"""Basler cameras through pylon (pypylon)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..base import DeviceError
from ..interfaces import Camera


class BaslerCamera(Camera):
    model = "Basler camera"

    def __init__(
        self,
        *,
        name: str | None = None,
        serial_number: str | None = None,
        pixel_size_um: float = 3.45,
        pixel_format: str | None = "Mono12",
        grab_timeout_ms: int = 5000,
        emulation: bool = False,
        transpose: bool = False,
        flip_lr: bool = False,
        flip_ud: bool = False,
    ) -> None:
        super().__init__(name)
        self.serial_number = None if serial_number is None else str(serial_number)
        self.pixel_size_um = pixel_size_um
        self.pixel_format = pixel_format
        self.grab_timeout_ms = grab_timeout_ms
        self.emulation = emulation
        self.transpose, self.flip_lr, self.flip_ud = transpose, flip_lr, flip_ud
        self._pylon: Any = None
        self._camera: Any = None
        self._model_name = ""

    def _connect(self) -> None:
        if self.emulation:
            os.environ.setdefault("PYLON_CAMEMU", "1")
        try:
            from pypylon import pylon
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise DeviceError("pypylon is required for Basler cameras (pip install pypylon)") from exc
        factory = pylon.TlFactory.GetInstance()
        devices = list(factory.EnumerateDevices())
        if self.serial_number is not None:
            devices = [info for info in devices if info.GetSerialNumber() == self.serial_number]
        if not devices:
            suffix = f" with S/N {self.serial_number}" if self.serial_number else ""
            raise DeviceError(f"no Basler camera found{suffix}")
        camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
        camera.Open()
        if self.pixel_format:
            try:
                camera.PixelFormat.SetValue(self.pixel_format)
            except Exception as exc:
                self.log.warning("cannot set pixel format %s: %s", self.pixel_format, exc)
        self._pylon, self._camera = pylon, camera
        self._model_name = devices[0].GetModelName()

    def _disconnect(self) -> None:
        self.stop_acquisition()
        self._camera.Close()
        self._camera = None

    def identify(self) -> str:
        return f"Basler {self._model_name}"

    def _get_float(self, nodes: Sequence[str]) -> float:
        for node in nodes:
            try:
                return float(getattr(self._camera, node).GetValue())
            except Exception:
                continue
        raise DeviceError(f"camera exposes none of {', '.join(nodes)}")

    def _set_float(self, nodes: Sequence[str], value: float) -> None:
        for node in nodes:
            try:
                getattr(self._camera, node).SetValue(value)
                return
            except Exception:
                continue
        raise DeviceError(f"camera exposes none of {', '.join(nodes)}")

    def set_exposure(self, exposure_ms: float) -> None:
        with self.lock:
            self._set_float(("ExposureTime", "ExposureTimeAbs"), exposure_ms * 1e3)

    def exposure(self) -> float:
        with self.lock:
            return self._get_float(("ExposureTime", "ExposureTimeAbs")) / 1e3

    def frame_rate(self) -> float:
        with self.lock:
            try:
                return self._get_float(("ResultingFrameRate", "ResultingFrameRateAbs"))
            except DeviceError:
                return super().frame_rate()

    def start_acquisition(self) -> None:
        with self.lock:
            if not self._camera.IsGrabbing():
                self._camera.StartGrabbing(self._pylon.GrabStrategy_LatestImageOnly)

    def stop_acquisition(self) -> None:
        with self.lock:
            if self._camera is not None and self._camera.IsGrabbing():
                self._camera.StopGrabbing()

    def grab(self) -> np.ndarray:
        with self.lock:
            if not self._camera.IsGrabbing():
                self.start_acquisition()
            result = self._camera.RetrieveResult(self.grab_timeout_ms, self._pylon.TimeoutHandling_ThrowException)
            try:
                if not result.GrabSucceeded():
                    raise DeviceError(f"grab failed: {result.GetErrorDescription()}")
                image = np.array(result.Array, copy=True)
            finally:
                result.Release()
        return self.orient(image)
