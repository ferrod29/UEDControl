"""PCO cameras through the ``pco`` Python package (ring-buffer recording)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..base import DeviceError
from ..interfaces import Camera

LATEST_IMAGE = 0xFFFFFFFF  # PCO_RECORDER_LATEST_IMAGE
_TIMEBASE_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0}


class PCOCamera(Camera):
    model = "PCO camera"

    def __init__(
        self,
        *,
        name: str | None = None,
        pixel_size_um: float = 4.65,
        exposure_ms: float = 100.0,
        buffer_images: int = 10,
        transpose: bool = False,
        flip_lr: bool = False,
        flip_ud: bool = False,
    ) -> None:
        super().__init__(name)
        self.pixel_size_um = pixel_size_um
        self.buffer_images = buffer_images
        self.transpose, self.flip_lr, self.flip_ud = transpose, flip_lr, flip_ud
        self._initial_exposure_ms = exposure_ms
        self._cam: Any = None

    def _connect(self) -> None:
        try:
            import pco
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise DeviceError("the 'pco' package is required for PCO cameras (pip install pco)") from exc
        self._cam = pco.Camera()
        self.set_exposure(self._initial_exposure_ms)

    def _disconnect(self) -> None:
        self.stop_acquisition()
        self._cam.close()
        self._cam = None

    def _recording(self) -> bool:
        state = self._cam.is_recording
        return bool(state() if callable(state) else state)

    def set_exposure(self, exposure_ms: float) -> None:
        with self.lock:
            self._cam.sdk.set_delay_exposure_time(0, "ms", int(round(exposure_ms * 1e3)), "us")

    def exposure(self) -> float:
        with self.lock:
            info = self._cam.sdk.get_delay_exposure_time()
        return info["exposure"] * _TIMEBASE_TO_MS[info["exposure timebase"]]

    def frame_rate(self) -> float:
        with self.lock:
            try:
                return self._cam.sdk.get_frame_rate()["frame rate mHz"] / 1000.0
            except Exception:
                return super().frame_rate()

    def start_acquisition(self) -> None:
        with self.lock:
            if not self._recording():
                self._cam.record(number_of_images=self.buffer_images, mode="ring buffer")
                self._cam.wait_for_first_image()

    def stop_acquisition(self) -> None:
        with self.lock:
            if self._cam is not None and self._recording():
                self._cam.stop()

    def grab(self) -> np.ndarray:
        with self.lock:
            if not self._recording():
                self.start_acquisition()
            if hasattr(self._cam, "wait_for_new_image"):
                self._cam.wait_for_new_image()
            else:
                time.sleep(self.exposure() * 1e-3)
            try:
                image, _meta = self._cam.image(image_index=LATEST_IMAGE)
            except Exception:
                image, _meta = self._cam.image(0)
        return self.orient(np.asarray(image))
