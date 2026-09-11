"""DECTRIS QUADRO (EIGER family) hybrid pixel detector, through the SIMPLON REST API.

A frame is acquired with ``arm`` / ``trigger`` / ``disarm`` and fetched as a
TIFF from the monitor interface. By default frames are flipped left-right and
transposed, which reproduces the orientation the original software displayed.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np

from ..base import Action, DeviceError, Setting
from ..interfaces import Camera

COUNTING_MODES = ("normal", "retrigger")
TRIGGER_MODES = ("ints", "exts", "exte", "inte")


def decode_darray(value: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(value["data"])
    return np.frombuffer(raw, dtype=np.dtype(value["type"])).reshape(value["shape"]).copy()


def encode_darray(array: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(array)
    return {
        "__darray__": (1, 0, 0),
        "type": array.dtype.str,
        "shape": list(array.shape),
        "filters": ["base64"],
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


class DectrisCamera(Camera):
    model = "DECTRIS QUADRO"

    def __init__(
        self,
        host: str = "10.6.101.55",
        port: int = 80,
        *,
        name: str | None = None,
        api_version: str = "1.8.0",
        incident_energy_ev: float = 1e5,
        timeout_s: float = 10.0,
        transpose: bool = True,
        flip_lr: bool = True,
        flip_ud: bool = False,
        session: Any = None,
    ) -> None:
        super().__init__(name)
        self.host = host
        self.port = port
        self.api_version = api_version
        self.incident_energy_ev = incident_energy_ev
        self.timeout_s = timeout_s
        self.transpose, self.flip_lr, self.flip_ud = transpose, flip_lr, flip_ud
        self._session = session
        self._exposure_ms: float | None = None
        self._saved_mask: np.ndarray | None = None

    # ------------------------------------------------------------------ REST
    def _url(self, interface: str, kind: str, parameter: str) -> str:
        return f"http://{self.host}:{self.port}/{interface}/api/{self.api_version}/{kind}/{parameter}"

    def _check(self, response: Any) -> Any:
        if not 200 <= response.status_code < 300:
            raise DeviceError(f"{self.name}: HTTP {response.status_code} {response.reason}")
        return response

    def get_config(self, parameter: str, interface: str = "detector") -> Any:
        response = self._session.get(self._url(interface, "config", parameter), timeout=self.timeout_s)
        return self._check(response).json()["value"]

    def set_config(self, parameter: str, value: Any, interface: str = "detector") -> None:
        response = self._session.put(
            self._url(interface, "config", parameter),
            data=json.dumps({"value": value}),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_s,
        )
        self._check(response)

    def get_status(self, parameter: str, interface: str = "detector") -> Any:
        response = self._session.get(self._url(interface, "status", parameter), timeout=self.timeout_s)
        return self._check(response).json()["value"]

    def command(self, command: str, interface: str = "detector") -> None:
        response = self._session.put(self._url(interface, "command", command), timeout=self.timeout_s)
        self._check(response)

    # ------------------------------------------------------------- lifecycle
    def _connect(self) -> None:
        if self._session is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise DeviceError("requests is required for the DECTRIS detector (pip install requests)") from exc
            self._session = requests.Session()
        try:
            self.set_config("mode", "enabled", interface="monitor")
            self.set_config("incident_energy", self.incident_energy_ev)
            self.pixel_size_um = float(self.get_config("x_pixel_size")) * 1e6
            self._exposure_ms = float(self.get_config("count_time")) * 1e3
        except DeviceError:
            raise
        except Exception as exc:
            raise DeviceError(f"cannot reach the DECTRIS detector at {self.host}: {exc}") from exc

    def _disconnect(self) -> None:
        try:
            self.command("abort")
        except Exception as exc:  # the detector may already be gone
            self.log.warning("abort on disconnect failed: %s", exc)

    def identify(self) -> str:
        return f"{self.get_config('description')} S/N {self.get_config('detector_number')}"

    # ------------------------------------------------------------ Camera API
    def set_exposure(self, exposure_ms: float) -> None:
        seconds = exposure_ms * 1e-3
        with self.lock:
            self.set_config("count_time", seconds)
            self.set_config("frame_time", seconds)
            self._exposure_ms = float(exposure_ms)

    def exposure(self) -> float:
        if self._exposure_ms is None:
            self._exposure_ms = float(self.get_config("count_time")) * 1e3
        return self._exposure_ms

    def frame_rate(self) -> float:
        return 1.0 / float(self.get_config("frame_time"))

    def set_frames_per_trigger(self, frames: int) -> None:
        self.set_config("nimages", int(frames))

    def image_shape(self) -> tuple[int, int]:
        return int(self.get_config("y_pixels_in_detector")), int(self.get_config("x_pixels_in_detector"))

    def grab(self) -> np.ndarray:
        import tifffile

        with self.lock:
            self.command("arm")
            self.command("trigger")
            self.command("disarm")
            response = self._session.get(
                self._url("monitor", "images", "monitor"),
                headers={"Accept": "application/tiff"},
                timeout=self.timeout_s,
            )
            self._check(response)
        return self.orient(np.asarray(tifffile.imread(io.BytesIO(response.content))))

    # ----------------------------------------------------------- pixel mask
    def pixel_mask(self) -> np.ndarray:
        return decode_darray(self.get_config("pixel_mask"))

    def set_pixel_mask(self, mask: np.ndarray, mask_type: str = "pixel_mask") -> None:
        self.set_config(mask_type, encode_darray(mask))

    def apply_pixel_mask(self, enabled: bool) -> None:
        self.set_config("pixel_mask_applied", bool(enabled))

    def _to_detector_frame(self, array: np.ndarray) -> np.ndarray:
        """Undo :meth:`orient` so display coordinates map onto detector pixels."""
        if self.transpose:
            array = array.T
        if self.flip_ud:
            array = array[::-1, :]
        if self.flip_lr:
            array = array[:, ::-1]
        return array

    def mask_disk(self, cx: float, cy: float, radius: float, enabled: bool = True) -> None:
        """Mask a disk (display coordinates) in the detector, e.g. to hide the direct beam.

        Disabling restores the mask that was active before.
        """
        with self.lock:
            if not enabled:
                if self._saved_mask is not None:
                    self.set_pixel_mask(self._saved_mask)
                    self._saved_mask = None
                return
            mask = self.pixel_mask()
            if self._saved_mask is None:
                self._saved_mask = mask.copy()
            else:
                mask = self._saved_mask.copy()
            display_shape = mask.T.shape if self.transpose else mask.shape
            rows, cols = np.ogrid[: display_shape[0], : display_shape[1]]
            disk = (cols - cx) ** 2 + (rows - cy) ** 2 <= radius**2
            mask[self._to_detector_frame(disk)] |= 1
            self.set_pixel_mask(mask)
            self.apply_pixel_mask(True)

    def set_roi_mode(self, enabled: bool, height: int = 128) -> None:
        with self.lock:
            self.set_config("roi_mode", "lines" if enabled else "disabled")
            if enabled:
                self.set_config("roi_bit_depth", 16)
                self.set_config("roi_y_size", int(height))

    # ------------------------------------------------------------- GUI hooks
    def settings(self) -> list[Setting]:
        with self.lock:
            return [
                Setting("counting_mode", "Counting mode", lambda v: self.set_config("counting_mode", v),
                        kind="choice", choices=COUNTING_MODES, initial=self.get_config("counting_mode")),
                Setting("trigger_mode", "Trigger mode", lambda v: self.set_config("trigger_mode", v),
                        kind="choice", choices=TRIGGER_MODES, initial=self.get_config("trigger_mode")),
                Setting("incident_energy", "Incident energy", lambda v: self.set_config("incident_energy", v),
                        unit="eV", minimum=1e3, maximum=3e5, decimals=0, initial=self.incident_energy_ev),
            ]

    def actions(self) -> list[Action]:
        return [
            Action("Initialize detector", lambda: self.command("initialize"),
                   confirm="Initialize the detector? This takes a while."),
            Action("Abort", lambda: self.command("abort")),
        ]
