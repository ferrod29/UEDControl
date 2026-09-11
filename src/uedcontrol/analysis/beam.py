"""Beam and region-of-interest (ROI) statistics.

Images are indexed ``[row, column]``; ``x`` is the column and ``y`` the row
coordinate, both in pixels. The width and count definitions follow the
original acquisition software (threshold counting at 1/2, 1/e² and 1/√2 of
the peak) but are vectorised; the old code looped over every pixel in Python
for every frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import correlate

from ..constants import ELEMENTARY_CHARGE_FC

HALF_MAXIMUM = 0.5
ONE_OVER_E2 = math.exp(-2.0)  # 0.1353
RMS_LEVEL = 1.0 / math.sqrt(2.0)  # 0.7071
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))  # 2.3548


@dataclass(frozen=True)
class CircularROI:
    """A circle, or an annulus when ``inner_radius > 0``, in pixel coordinates."""

    cx: float
    cy: float
    radius: float
    inner_radius: float = 0.0

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("ROI radius must be positive")
        if not 0 <= self.inner_radius < self.radius:
            raise ValueError("inner radius must satisfy 0 <= inner < radius")

    def bounds(self, shape: tuple[int, ...]) -> tuple[slice, slice]:
        """Row and column slices of the bounding box, clipped to the image."""
        n_rows, n_cols = shape[:2]
        r0 = max(math.floor(self.cy - self.radius), 0)
        r1 = min(math.ceil(self.cy + self.radius) + 1, n_rows)
        c0 = max(math.floor(self.cx - self.radius), 0)
        c1 = min(math.ceil(self.cx + self.radius) + 1, n_cols)
        if r0 >= r1 or c0 >= c1:
            raise ValueError("ROI lies outside the image")
        return slice(r0, r1), slice(c0, c1)

    def mask(self, shape: tuple[int, ...]) -> np.ndarray:
        """Boolean mask of the ROI over a full image of ``shape``."""
        rows, cols = np.ogrid[: shape[0], : shape[1]]
        d2 = (cols - self.cx) ** 2 + (rows - self.cy) ** 2
        inside = d2 <= self.radius**2
        if self.inner_radius > 0:
            inside &= d2 >= self.inner_radius**2
        return inside

    def moved_to(self, cx: float, cy: float) -> CircularROI:
        return replace(self, cx=float(cx), cy=float(cy))


@dataclass(frozen=True)
class RoiData:
    """Bounding box of an ROI with the pixels outside the ROI set to zero."""

    values: np.ndarray
    inside: np.ndarray
    row0: int
    col0: int


def extract_roi(image: np.ndarray, roi: CircularROI | None) -> RoiData:
    data = np.asarray(image, dtype=np.float64)
    if roi is None:
        return RoiData(data, np.ones(data.shape, dtype=bool), 0, 0)
    rows, cols = roi.bounds(data.shape)
    rr, cc = np.ogrid[rows, cols]
    d2 = (cc - roi.cx) ** 2 + (rr - roi.cy) ** 2
    inside = d2 <= roi.radius**2
    if roi.inner_radius > 0:
        inside &= d2 >= roi.inner_radius**2
    return RoiData(np.where(inside, data[rows, cols], 0.0), inside, rows.start, cols.start)


@dataclass(frozen=True)
class ProfileWidths:
    """Widths of a 1-D profile in pixels."""

    fwhm: float  # samples at or above half of the maximum
    e2: float  # samples at or above 1/e² of the maximum
    rms: float  # sigma of a Gaussian with the same FWHM

    def scaled(self, factor: float) -> ProfileWidths:
        return ProfileWidths(self.fwhm * factor, self.e2 * factor, self.rms * factor)


def profile_widths(profile: np.ndarray) -> ProfileWidths:
    values = np.abs(np.asarray(profile, dtype=np.float64))
    peak = float(values.max()) if values.size else 0.0
    if peak <= 0:
        return ProfileWidths(0.0, 0.0, 0.0)
    fwhm = float(np.count_nonzero(values >= HALF_MAXIMUM * peak))
    e2 = float(np.count_nonzero(values >= ONE_OVER_E2 * peak))
    return ProfileWidths(fwhm, e2, fwhm / FWHM_PER_SIGMA)


@dataclass(frozen=True)
class RoiCounts:
    """Detector counts inside the ROI."""

    total: float
    peak: float
    per_pixel: float
    rms: float  # counts in pixels at or above 1/√2 of the peak
    fwhm: float  # counts in pixels at or above half of the peak
    e2: float  # counts in pixels at or above 1/e² of the peak


def roi_counts(roi: RoiData) -> RoiCounts:
    pixels = roi.values[roi.inside]
    if pixels.size == 0:
        return RoiCounts(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    peak = float(pixels.max())
    total = float(pixels.sum())

    def above(level: float) -> float:
        return float(pixels[pixels >= level * peak].sum()) if peak > 0 else 0.0

    return RoiCounts(
        total=total,
        peak=peak,
        per_pixel=total / pixels.size,
        rms=above(RMS_LEVEL),
        fwhm=above(HALF_MAXIMUM),
        e2=above(ONE_OVER_E2),
    )


@dataclass(frozen=True)
class BeamCharge:
    """Charge figures derived from the counts of an electron-counting detector."""

    electrons_per_pulse: float  # NaN for a continuous beam
    bunch_charge_fc: float  # NaN for a continuous beam
    current_pa: float
    electrons_rms: float  # per pulse (per exposure for a continuous beam)
    electrons_fwhm: float
    electrons_e2: float


def beam_charge(
    counts: RoiCounts,
    exposure_ms: float,
    rep_rate_hz: float,
    pulsed: bool = True,
    counts_per_electron: float = 1.0,
) -> BeamCharge:
    """Convert ROI counts to electrons per pulse, bunch charge and average current."""
    seconds = exposure_ms * 1e-3
    if seconds <= 0:
        raise ValueError("exposure must be positive")
    if pulsed and rep_rate_hz <= 0:
        raise ValueError("repetition rate must be positive for a pulsed beam")
    electrons = counts.total / counts_per_electron
    current_pa = electrons * ELEMENTARY_CHARGE_FC / seconds * 1e-3  # fC/s = fA -> pA
    pulses = rep_rate_hz * seconds if pulsed else 1.0
    scale = 1.0 / (counts_per_electron * pulses)
    per_pulse = electrons / pulses if pulsed else math.nan
    return BeamCharge(
        electrons_per_pulse=per_pulse,
        bunch_charge_fc=per_pulse * ELEMENTARY_CHARGE_FC,
        current_pa=current_pa,
        electrons_rms=counts.rms * scale,
        electrons_fwhm=counts.fwhm * scale,
        electrons_e2=counts.e2 * scale,
    )


def profile_correlation(values: np.ndarray) -> np.ndarray:
    """Cross-correlation of the horizontal and vertical projections (pump-probe signal)."""
    x_profile = values.sum(axis=0)
    y_profile = values.sum(axis=1)
    return correlate(x_profile, y_profile, mode="same") / float(values.size) ** 2


@dataclass(frozen=True)
class RoiStatistics:
    peak_x: float
    peak_y: float
    centroid_x: float
    centroid_y: float
    diameter: int
    x_profile: np.ndarray
    y_profile: np.ndarray
    x_widths: ProfileWidths
    y_widths: ProfileWidths
    counts: RoiCounts
    rms_contrast: float  # sqrt(std² + mean²) of the ROI pixels
    correlation: np.ndarray
    row0: int  # image row of the first profile sample
    col0: int  # image column of the first profile sample


def analyze_roi(image: np.ndarray, roi: CircularROI | None = None) -> RoiStatistics:
    """All statistics shown in the acquisition panel, for one image."""
    data = extract_roi(image, roi)
    values = data.values
    x_profile = values.sum(axis=0)
    y_profile = values.sum(axis=1)
    peak_row, peak_col = np.unravel_index(int(np.argmax(values)), values.shape)

    weights = np.clip(values, 0.0, None)
    weight = float(weights.sum())
    if weight > 0:
        centroid_x = data.col0 + float((weights.sum(axis=0) * np.arange(values.shape[1])).sum()) / weight
        centroid_y = data.row0 + float((weights.sum(axis=1) * np.arange(values.shape[0])).sum()) / weight
    else:
        centroid_x = data.col0 + (values.shape[1] - 1) / 2
        centroid_y = data.row0 + (values.shape[0] - 1) / 2

    pixels = values[data.inside]
    rms = float(np.sqrt(np.mean(pixels**2))) if pixels.size else 0.0
    return RoiStatistics(
        peak_x=float(data.col0 + peak_col),
        peak_y=float(data.row0 + peak_row),
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        diameter=int(values.shape[1]),
        x_profile=x_profile,
        y_profile=y_profile,
        x_widths=profile_widths(x_profile),
        y_widths=profile_widths(y_profile),
        counts=roi_counts(data),
        rms_contrast=rms,
        correlation=profile_correlation(values),
        row0=data.row0,
        col0=data.col0,
    )


def block_region(image: np.ndarray, roi: CircularROI) -> np.ndarray:
    """Copy of ``image`` with the ROI pixels set to the image minimum (software beam block)."""
    blocked = np.array(image, dtype=np.float64, copy=True)
    blocked[roi.mask(blocked.shape)] = blocked.min()
    return blocked
