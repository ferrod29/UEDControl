"""Any camera supported by Micro-Manager, through ``pymmcore``.

Either load a Micro-Manager configuration file, or give the device adapter
``library`` and ``device`` names. Camera-specific properties (e.g. the Andor
Zyla shutter mode or the Basler pixel type used in the original code) go in
``properties``; see ``config/example_instrument.yaml``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..base import DeviceError
from ..interfaces import Camera


class MicroManagerCamera(Camera):
    model = "Micro-Manager camera"

    def __init__(
        self,
        *,
        name: str | None = None,
        mm_dir: str = r"C:\Program Files\Micro-Manager-2.0",
        config_file: str | None = None,
        device_label: str = "Camera",
        library: str | None = None,
        device: str | None = None,
        properties: Mapping[str, Any] | None = None,
        pixel_size_um: float = 1.0,
        grab_timeout_s: float = 5.0,
        transpose: bool = False,
        flip_lr: bool = False,
        flip_ud: bool = False,
    ) -> None:
        super().__init__(name)
        if config_file is None and not (library and device):
            raise ValueError("give either 'config_file' or both 'library' and 'device'")
        self.mm_dir = mm_dir
        self.config_file = config_file
        self.device_label = device_label
        self.library = library
        self.device = device
        self.properties = dict(properties or {})
        self.pixel_size_um = pixel_size_um
        self.grab_timeout_s = grab_timeout_s
        self.transpose, self.flip_lr, self.flip_ud = transpose, flip_lr, flip_ud
        self._core: Any = None

    def _connect(self) -> None:
        try:
            import pymmcore
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise DeviceError("pymmcore is required for Micro-Manager cameras (pip install pymmcore)") from exc
        core = pymmcore.CMMCore()
        core.setDeviceAdapterSearchPaths([self.mm_dir])
        previous_dir = os.getcwd()
        try:
            os.chdir(self.mm_dir)  # adapters load their own DLLs relative to this folder
            if self.config_file:
                core.loadSystemConfiguration(str(Path(self.mm_dir, self.config_file)))
            else:
                core.loadDevice(self.device_label, self.library, self.device)
                core.initializeDevice(self.device_label)
            core.setCameraDevice(self.device_label)
            for key, value in self.properties.items():
                core.setProperty(self.device_label, key, str(value))
        except Exception as exc:
            core.unloadAllDevices()
            raise DeviceError(f"Micro-Manager could not load {self.device_label}: {exc}") from exc
        finally:
            os.chdir(previous_dir)
        self._core = core

    def _disconnect(self) -> None:
        self.stop_acquisition()
        self._core.unloadAllDevices()
        self._core = None

    def set_exposure(self, exposure_ms: float) -> None:
        with self.lock:
            self._core.setExposure(float(exposure_ms))

    def exposure(self) -> float:
        with self.lock:
            return float(self._core.getExposure())

    def start_acquisition(self) -> None:
        with self.lock:
            if not self._core.isSequenceRunning():
                self._core.startContinuousSequenceAcquisition(0)

    def stop_acquisition(self) -> None:
        with self.lock:
            if self._core is not None and self._core.isSequenceRunning():
                self._core.stopSequenceAcquisition()

    def grab(self) -> np.ndarray:
        with self.lock:
            core = self._core
            if core.isSequenceRunning():
                deadline = time.monotonic() + self.grab_timeout_s
                while core.getRemainingImageCount() == 0:
                    if time.monotonic() > deadline:
                        raise DeviceError(f"{self.name}: no image within {self.grab_timeout_s} s")
                    time.sleep(0.001)
                image = core.popNextImage()
            else:
                core.snapImage()
                image = core.getImage()
        return self.orient(np.asarray(image))
