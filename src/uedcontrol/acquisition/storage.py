"""Saving images and scans.

Every scan gets its own folder ``<data_dir>/<name>/`` (``_001``, ``_002``...
are appended instead of overwriting) containing:

* ``<name>.h5``: metadata, delays, background, every averaged image and the
  averaged results;
* the plain-text files the original software wrote (``tsteps.txt``,
  ``pp_avgi_data.txt``, ``pp_rmsc_data.txt``, ``pump_probe_image.txt``), so
  existing analysis scripts keep working;
* optionally one TIFF and/or PNG per image, named like before
  (``001_delay_-1.5ps.tiff``).

All file operations run on a single background thread, in order, so the
acquisition never waits for the disk.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import tifffile

if TYPE_CHECKING:
    from ..analysis.pumpprobe import PumpProbeAccumulator
    from .scan import ScanPoint, ScanSettings

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".npy")


def to_uint16(image: np.ndarray) -> np.ndarray:
    """Stretch an image to the full 16-bit range (for PNG previews)."""
    data = np.nan_to_num(np.asarray(image, dtype=np.float64))
    low, high = float(data.min()), float(data.max())
    if high <= low:
        return np.zeros(data.shape, dtype=np.uint16)
    return ((data - low) / (high - low) * 65535).astype(np.uint16)


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Save an image; TIFF keeps the values (float32), PNG is a scaled 16-bit preview."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        tifffile.imwrite(path, np.asarray(image, dtype=np.float32))
    elif suffix == ".png":
        import imageio.v3 as iio

        iio.imwrite(path, to_uint16(image))
    elif suffix == ".npy":
        np.save(path, np.asarray(image))
    else:
        raise ValueError(f"unsupported image format {suffix!r}; use one of {', '.join(IMAGE_SUFFIXES)}")
    return path


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as a 2-D float array (colour images are averaged to grey)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        image = tifffile.imread(path)
    elif suffix == ".npy":
        image = np.load(path)
    else:
        import imageio.v3 as iio

        image = iio.imread(path)
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=-1)
    if image.ndim != 2:
        raise ValueError(f"{path} is not a 2-D image")
    return image


def unique_path(path: Path) -> Path:
    """``path`` if it does not exist, otherwise ``path_001``, ``path_002``..."""
    if not path.exists():
        return path
    for number in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{number:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"too many files named like {path}")


def _attribute(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value)
    return json.dumps(value, default=str)


def _replace_dataset(group: h5py.Group, name: str, data: Any) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


class ScanWriter:
    def __init__(
        self,
        directory: str | Path,
        name: str,
        metadata: Mapping[str, Any] | None = None,
        *,
        hdf5: bool = True,
        tiff: bool = False,
        png: bool = False,
        text: bool = True,
        arrays: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.name = name
        self.scan_dir = unique_path(Path(directory) / name)
        self.metadata = dict(metadata or {})
        self.arrays = {key: np.asarray(value) for key, value in (arrays or {}).items()}
        self.hdf5, self.tiff, self.png, self.text = hdf5, tiff, png, text
        self.path: Path | None = None
        self._file: h5py.File | None = None
        self._points: list[tuple[int, int, int, float, float, float]] = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan-writer")
        self._futures: list[Future[None]] = []

    # --------------------------------------------------------------- plumbing
    def _submit(self, function: Callable[..., None], *args: Any) -> None:
        self._raise_errors()
        self._futures.append(self._executor.submit(function, *args))

    def _raise_errors(self) -> None:
        pending = []
        for future in self._futures:
            if not future.done():
                pending.append(future)
            elif future.exception() is not None:
                raise future.exception()  # type: ignore[misc]
        self._futures = pending

    # ------------------------------------------------------------- public API
    def open(self, settings: ScanSettings) -> None:
        info = {
            "frames_per_point": settings.frames_per_point,
            "runs": settings.runs,
            "bidirectional": settings.bidirectional,
            "settle_time_s": settings.settle_time_s,
            "roi": None if settings.roi is None else [settings.roi.cx, settings.roi.cy,
                                                       settings.roi.radius, settings.roi.inner_radius],
        }
        self._submit(self._open, settings.delays_ps.copy(), settings.background, info)

    def write_point(self, point: ScanPoint) -> None:
        self._submit(self._write_point, point)

    def write_summary(self, accumulator: PumpProbeAccumulator) -> None:
        profile = accumulator.mean_profile
        self._submit(
            self._write_summary,
            accumulator.delays.copy(),
            accumulator.mean_total.copy(),
            accumulator.mean_rms.copy(),
            None if profile is None else profile.copy(),
            accumulator.counts.copy(),
        )

    def close(self) -> None:
        self._submit(self._close)
        self._executor.shutdown(wait=True)
        self._raise_errors()

    # ------------------------------------------------------ background work
    def _open(self, delays: np.ndarray, background: np.ndarray | None, info: dict[str, Any]) -> None:
        self.scan_dir.mkdir(parents=True, exist_ok=True)
        if self.hdf5:
            self.path = self.scan_dir / f"{self.name}.h5"
            handle = h5py.File(self.path, "w")
            for key, value in {**self.metadata, **info}.items():
                handle.attrs[key] = _attribute(value)
            handle.attrs["start_time"] = dt.datetime.now().isoformat(timespec="seconds")
            handle.create_dataset("delays_ps", data=delays)
            if background is not None:
                handle.create_dataset("background", data=np.asarray(background, dtype=np.float32),
                                      compression="gzip", compression_opts=4)
            for key, value in self.arrays.items():
                handle.create_dataset(key, data=value, compression="gzip", compression_opts=4)
            handle.create_group("images")
            self._file = handle
        if self.text:
            table = np.column_stack([np.arange(delays.size), delays])
            np.savetxt(self.scan_dir / "tsteps.txt", table, fmt=["%d", "%.6g"])
        log.info("scan data goes to %s", self.scan_dir)

    def _write_point(self, point: ScanPoint) -> None:
        image = np.asarray(point.image, dtype=np.float32)
        if self._file is not None:
            dataset = self._file["images"].create_dataset(
                f"{point.number:05d}", data=image, compression="gzip", compression_opts=4
            )
            dataset.attrs.update(
                run=point.run,
                index=point.index,
                delay_ps=point.delay_ps,
                timestamp=point.timestamp,
                roi_total=point.stats.counts.total,
                roi_rms=point.stats.rms_contrast,
            )
        stem = f"{point.number:03d}_delay_{point.delay_ps:g}ps"
        if self.tiff:
            save_image(self.scan_dir / f"{stem}.tiff", image)
        if self.png:
            save_image(self.scan_dir / f"{stem}.png", image)
        self._points.append(
            (point.number, point.run, point.index, point.delay_ps, point.stats.counts.total, point.stats.rms_contrast)
        )

    def _write_summary(
        self,
        delays: np.ndarray,
        total: np.ndarray,
        rms: np.ndarray,
        profile: np.ndarray | None,
        counts: np.ndarray,
    ) -> None:
        if self._file is not None:
            results = self._file.require_group("results")
            _replace_dataset(results, "delays_ps", delays)
            _replace_dataset(results, "roi_total", total)
            _replace_dataset(results, "roi_rms", rms)
            _replace_dataset(results, "points_per_delay", counts)
            if profile is not None:
                _replace_dataset(results, "roi_profile", profile)
            if self._points:
                table = np.array(
                    self._points,
                    dtype=[("number", "i4"), ("run", "i4"), ("index", "i4"),
                           ("delay_ps", "f8"), ("roi_total", "f8"), ("roi_rms", "f8")],
                )
                _replace_dataset(results, "points", table)
        if self.text:
            index = np.arange(delays.size)
            np.savetxt(self.scan_dir / "pp_avgi_data.txt", np.column_stack([index, delays, total]),
                       fmt=["%d", "%.6g", "%.8g"])
            np.savetxt(self.scan_dir / "pp_rmsc_data.txt", np.column_stack([index, delays, rms]),
                       fmt=["%d", "%.6g", "%.8g"])
            if profile is not None:
                positions = np.arange(profile.shape[1]) - (profile.shape[1] - 1) / 2
                np.savetxt(self.scan_dir / "pump_probe_image.txt", np.column_stack([positions, profile.T]),
                           fmt="%.8g")

    def _close(self) -> None:
        if self._file is not None:
            self._file.attrs["end_time"] = dt.datetime.now().isoformat(timespec="seconds")
            self._file.close()
            self._file = None
