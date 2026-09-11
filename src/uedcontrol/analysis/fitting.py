"""Least-squares fits of beam profiles with Gaussian or Lorentzian line shapes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .beam import FWHM_PER_SIGMA, profile_widths


def gaussian(x: np.ndarray, amplitude: float, sigma: float, center: float, offset: float = 0.0) -> np.ndarray:
    return offset + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x: np.ndarray, amplitude: float, gamma: float, center: float, offset: float = 0.0) -> np.ndarray:
    return offset + amplitude / (1.0 + ((x - center) / gamma) ** 2)


MODELS = {"gaussian": gaussian, "lorentzian": lorentzian}


@dataclass(frozen=True)
class ProfileFit:
    model: str
    amplitude: float
    width: float  # sigma (Gaussian) or half width at half maximum (Lorentzian)
    center: float
    offset: float
    success: bool

    @property
    def fwhm(self) -> float:
        return FWHM_PER_SIGMA * self.width if self.model == "gaussian" else 2.0 * self.width

    @property
    def rms(self) -> float:
        """Sigma of a Gaussian with the same FWHM."""
        return self.fwhm / FWHM_PER_SIGMA

    @property
    def e2_width(self) -> float:
        """Full width at 1/e² of the peak."""
        if self.model == "gaussian":
            return 4.0 * self.width
        return 2.0 * self.width * math.sqrt(math.e**2 - 1.0)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return MODELS[self.model](np.asarray(x, dtype=float), self.amplitude, self.width, self.center, self.offset)


def fit_profile(profile: np.ndarray, model: str = "gaussian", x: np.ndarray | None = None) -> ProfileFit:
    """Robust (soft-L1) least-squares fit of a 1-D profile with an offset."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}")
    y = np.asarray(profile, dtype=float)
    if y.size < 4:
        raise ValueError("profile too short to fit")
    xs = np.arange(y.size, dtype=float) if x is None else np.asarray(x, dtype=float)
    spacing = float(xs[1] - xs[0])

    offset0 = float(y.min())
    amplitude0 = float(y.max() - offset0) or 1.0
    center0 = float(xs[int(np.argmax(y))])
    fwhm0 = max(profile_widths(y - offset0).fwhm, 2.0) * abs(spacing)
    width0 = fwhm0 / FWHM_PER_SIGMA if model == "gaussian" else fwhm0 / 2.0

    function = MODELS[model]
    result = least_squares(
        lambda p: function(xs, *p) - y,
        x0=[amplitude0, width0, center0, offset0],
        loss="soft_l1",
        f_scale=max(0.05 * amplitude0, 1e-12),
        method="trf",
        x_scale="jac",
        max_nfev=10000,
    )
    amplitude, width, center, offset = (float(v) for v in result.x)
    return ProfileFit(model, amplitude, abs(width), center, offset, bool(result.success))
