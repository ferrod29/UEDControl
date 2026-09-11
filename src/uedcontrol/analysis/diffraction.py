"""Electron diffraction helpers: wavelength, scattering vector, radial profiles, rings.

Note on the scattering vector: for a ring of radius ``r`` on a detector at
distance ``L`` the scattering angle is ``2θ = arctan(r/L)`` and
``q = 4π sin(θ)/λ``. The original software used ``sin(arctan(r/L))``, i.e.
``sin(2θ)``, which overestimated q by about a factor of two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import rotate, shift
from scipy.signal import find_peaks, peak_widths

from ..constants import ELECTRON_MASS, ELEMENTARY_CHARGE, PLANCK, SPEED_OF_LIGHT


def electron_wavelength_angstrom(voltage_v: float) -> float:
    """Relativistic de Broglie wavelength of electrons accelerated through ``voltage_v``."""
    if voltage_v <= 0:
        raise ValueError("accelerating voltage must be positive")
    energy = ELEMENTARY_CHARGE * voltage_v
    momentum = math.sqrt(2 * ELECTRON_MASS * energy * (1 + energy / (2 * ELECTRON_MASS * SPEED_OF_LIGHT**2)))
    return PLANCK / momentum * 1e10


def lorentz_factors(voltage_v: float) -> tuple[float, float]:
    """(beta, gamma) of electrons accelerated through ``voltage_v``."""
    gamma = 1 + ELEMENTARY_CHARGE * voltage_v / (ELECTRON_MASS * SPEED_OF_LIGHT**2)
    return math.sqrt(1 - 1 / gamma**2), gamma


def scattering_vector(
    radius_px: float | np.ndarray, pixel_size_um: float, camera_length_m: float, wavelength_angstrom: float
) -> float | np.ndarray:
    """Scattering vector q = 4π sin(θ)/λ in 1/Å for a radius on the detector."""
    two_theta = np.arctan(np.asarray(radius_px, dtype=float) * pixel_size_um * 1e-6 / camera_length_m)
    return 4 * np.pi * np.sin(two_theta / 2) / wavelength_angstrom


def radial_profile(
    image: np.ndarray, center: tuple[float, float], bin_width: float = 1.0, exclude: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal average around ``center = (x, y)``. Returns (radii, mean intensity)."""
    data = np.asarray(image, dtype=np.float64)
    rows, cols = np.indices(data.shape)
    radius = np.hypot(cols - center[0], rows - center[1])
    valid = np.ones(data.shape, dtype=bool) if exclude is None else ~exclude
    bins = (radius[valid] / bin_width).astype(np.int64)
    sums = np.bincount(bins, weights=data[valid])
    counts = np.bincount(bins)
    profile = sums / np.maximum(counts, 1)
    radii = (np.arange(profile.size) + 0.5) * bin_width
    return radii, profile


def symmetrize(image: np.ndarray, n_fold: int, center: tuple[float, float] | None = None) -> np.ndarray:
    """Average the image over ``n_fold`` rotations about ``center`` (x, y)."""
    if n_fold < 1:
        raise ValueError("n_fold must be >= 1")
    data = np.asarray(image, dtype=np.float64)
    middle = ((data.shape[1] - 1) / 2, (data.shape[0] - 1) / 2)
    offset = (0.0, 0.0) if center is None else (middle[1] - center[1], middle[0] - center[0])
    centered = shift(data, offset, order=1) if center is not None else data
    total = centered.copy()
    for k in range(1, n_fold):
        total += rotate(centered, 360.0 * k / n_fold, reshape=False, order=1)
    result = total / n_fold
    return shift(result, (-offset[0], -offset[1]), order=1) if center is not None else result


@dataclass(frozen=True)
class DiffractionPeak:
    radius_px: float
    intensity: float
    width_px: float
    q: float  # 1/Å
    q_width: float  # 1/Å


def find_rings(
    radii: np.ndarray,
    profile: np.ndarray,
    *,
    pixel_size_um: float,
    camera_length_m: float,
    wavelength_angstrom: float,
    prominence: float | None = None,
    min_distance_px: float = 5.0,
) -> list[DiffractionPeak]:
    """Locate rings as peaks of the radial profile and convert them to q."""
    values = np.nan_to_num(np.asarray(profile, dtype=float))
    step = float(radii[1] - radii[0]) if len(radii) > 1 else 1.0
    if prominence is None:
        prominence = 0.05 * float(np.ptp(values)) if values.size else 0.0
    indices, _ = find_peaks(values, prominence=prominence, distance=max(1, int(min_distance_px / step)))
    if indices.size == 0:
        return []
    widths = peak_widths(values, indices, rel_height=0.5)[0] * step
    def to_q(radius: float) -> float:
        return float(scattering_vector(radius, pixel_size_um, camera_length_m, wavelength_angstrom))

    peaks = []
    for index, width in zip(indices, widths, strict=True):
        radius = float(radii[index])
        q_width = to_q(radius + width / 2) - to_q(max(radius - width / 2, 0.0))
        peaks.append(DiffractionPeak(radius, float(values[index]), float(width), to_q(radius), q_width))
    return peaks


def transverse_coherence_length_nm(wavelength_angstrom: float, divergence_rad: float) -> float:
    """L_c = λ / (2π σ_θ) for an rms angular spread σ_θ."""
    if divergence_rad <= 0:
        raise ValueError("divergence must be positive")
    return wavelength_angstrom * 0.1 / (2 * math.pi * divergence_rad)


def normalized_emittance_nm(sigma_um: float, divergence_rad: float, voltage_v: float) -> float:
    """ε_n = βγ σ_x σ_θ (at a beam waist), in nm·rad."""
    beta, gamma = lorentz_factors(voltage_v)
    return beta * gamma * sigma_um * 1e-6 * divergence_rad * 1e9
