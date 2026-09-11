import math

import numpy as np
import pytest

from uedcontrol.analysis.beam import (
    FWHM_PER_SIGMA,
    CircularROI,
    analyze_roi,
    beam_charge,
    block_region,
    extract_roi,
    profile_widths,
    roi_counts,
)
from uedcontrol.analysis.fitting import fit_profile


def gaussian_image(shape=(201, 201), cx=100, cy=90, sigma=10.0, amplitude=1000.0):
    rows, cols = np.indices(shape)
    return amplitude * np.exp(-((cols - cx) ** 2 + (rows - cy) ** 2) / (2 * sigma**2))


def test_profile_widths_of_a_gaussian():
    x = np.arange(201)
    widths = profile_widths(np.exp(-((x - 100) ** 2) / (2 * 10.0**2)))
    assert abs(widths.fwhm - FWHM_PER_SIGMA * 10) <= 1.5
    assert abs(widths.e2 - 40) <= 1.5
    assert widths.rms == pytest.approx(widths.fwhm / FWHM_PER_SIGMA)
    assert profile_widths(np.zeros(5)).fwhm == 0


def test_extract_roi_geometry():
    image = gaussian_image()
    data = extract_roi(image, CircularROI(cx=100, cy=90, radius=30))
    assert data.values.shape == (61, 61)
    assert (data.row0, data.col0) == (60, 70)
    assert data.inside[30, 30] and not data.inside[0, 0]
    assert data.values[0, 0] == 0.0


def test_roi_clipping_and_validation():
    image = np.ones((50, 50))
    data = extract_roi(image, CircularROI(cx=2, cy=2, radius=10))
    assert (data.row0, data.col0) == (0, 0)
    with pytest.raises(ValueError):
        extract_roi(image, CircularROI(cx=200, cy=200, radius=5))
    with pytest.raises(ValueError):
        CircularROI(0, 0, 5, inner_radius=6)


def test_annulus_excludes_the_centre():
    image = gaussian_image()
    disk = roi_counts(extract_roi(image, CircularROI(100, 90, 30)))
    ring = roi_counts(extract_roi(image, CircularROI(100, 90, 30, inner_radius=10)))
    assert ring.total < disk.total
    assert disk.total == pytest.approx(image[CircularROI(100, 90, 30).mask(image.shape)].sum())


def test_analyze_roi_centre_and_widths():
    stats = analyze_roi(gaussian_image(), CircularROI(cx=100, cy=90, radius=50))
    assert (stats.peak_x, stats.peak_y) == (100.0, 90.0)
    assert stats.centroid_x == pytest.approx(100, abs=0.01)
    assert stats.centroid_y == pytest.approx(90, abs=0.01)
    assert abs(stats.x_widths.fwhm - FWHM_PER_SIGMA * 10) <= 1.5
    assert stats.correlation.size == stats.x_profile.size
    assert stats.rms_contrast > 0


def test_beam_charge_pulsed_and_continuous():
    counts = roi_counts(extract_roi(np.full((10, 10), 1e4), None))  # 1e6 counts in total
    pulsed = beam_charge(counts, exposure_ms=100, rep_rate_hz=1000)
    assert pulsed.electrons_per_pulse == pytest.approx(1e4)  # 100 pulses per exposure
    assert pulsed.bunch_charge_fc == pytest.approx(1e4 * 1.602176634e-4)
    assert pulsed.current_pa == pytest.approx(1e6 * 1.602176634e-4 / 0.1 * 1e-3)
    continuous = beam_charge(counts, exposure_ms=100, rep_rate_hz=0, pulsed=False)
    assert math.isnan(continuous.electrons_per_pulse)
    assert continuous.current_pa == pytest.approx(pulsed.current_pa)
    with pytest.raises(ValueError):
        beam_charge(counts, exposure_ms=0, rep_rate_hz=1000)


def test_block_region():
    image = gaussian_image()
    blocked = block_region(image, CircularROI(100, 90, 5))
    assert blocked[90, 100] == image.min()
    assert blocked[0, 0] == image[0, 0]


@pytest.mark.parametrize("model, width", [("gaussian", 6.2), ("lorentzian", 4.0)])
def test_fit_profile_recovers_parameters(model, width):
    x = np.arange(100, dtype=float)
    if model == "gaussian":
        y = 5 + 100 * np.exp(-0.5 * ((x - 40.3) / width) ** 2)
    else:
        y = 5 + 100 / (1 + ((x - 40.3) / width) ** 2)
    y += np.random.default_rng(0).normal(0, 0.5, x.size)
    fit = fit_profile(y, model)
    assert fit.success
    assert fit.width == pytest.approx(width, rel=0.02)
    assert fit.center == pytest.approx(40.3, abs=0.05)
    assert fit.offset == pytest.approx(5, abs=0.5)
    assert fit.evaluate(np.array([40.3]))[0] == pytest.approx(105, rel=0.02)
