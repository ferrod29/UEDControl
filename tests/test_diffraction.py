import math

import numpy as np
import pytest

from uedcontrol.analysis.diffraction import (
    electron_wavelength_angstrom,
    find_rings,
    lorentz_factors,
    normalized_emittance_nm,
    radial_profile,
    scattering_vector,
    symmetrize,
    transverse_coherence_length_nm,
)
from uedcontrol.devices.simulated import SimulatedCamera


def test_electron_wavelength_at_100_kv():
    assert electron_wavelength_angstrom(100e3) == pytest.approx(0.037014, rel=1e-4)
    with pytest.raises(ValueError):
        electron_wavelength_angstrom(0)


def test_lorentz_factors():
    beta, gamma = lorentz_factors(100e3)
    assert gamma == pytest.approx(1.19569, rel=1e-4)
    assert beta == pytest.approx(0.54822, rel=1e-4)


def test_scattering_vector_small_angle_limit():
    wavelength = electron_wavelength_angstrom(100e3)
    q = scattering_vector(100, pixel_size_um=75, camera_length_m=1.0, wavelength_angstrom=wavelength)
    small_angle = 2 * math.pi * (100 * 75e-6) / wavelength  # 2π·r/(λL), r in m, L = 1 m
    assert q == pytest.approx(small_angle, rel=1e-4)


def test_radial_profile_finds_a_ring():
    rows, cols = np.indices((201, 201))
    radius = np.hypot(cols - 100, rows - 100)
    image = np.exp(-((radius - 50) ** 2) / 8)
    radii, profile = radial_profile(image, (100, 100))
    assert radii[np.argmax(profile)] == pytest.approx(50, abs=1)


def test_find_rings_on_simulated_pattern():
    camera = SimulatedCamera(pattern="diffraction_rings", realtime=False, noise=0.0)
    camera.connect()
    image = camera.grab()
    radii, profile = radial_profile(image, (256, 256))
    wavelength = electron_wavelength_angstrom(100e3)
    peaks = find_rings(radii, profile, pixel_size_um=75, camera_length_m=1.0, wavelength_angstrom=wavelength,
                       min_distance_px=10)
    found = [p.radius_px for p in peaks if p.radius_px > 30]
    for expected in (0.16 * 512, 0.26 * 512, 0.36 * 512):
        assert min(abs(r - expected) for r in found) < 2
    assert all(p.q > 0 and p.q_width > 0 for p in peaks)


def test_symmetrize_keeps_a_symmetric_image():
    rows, cols = np.indices((101, 101))
    image = np.exp(-(np.hypot(cols - 50, rows - 50) ** 2) / 200)
    assert np.allclose(symmetrize(image, 6)[30:70, 30:70], image[30:70, 30:70], atol=1e-2)


def test_coherence_and_emittance():
    assert transverse_coherence_length_nm(0.037, 1e-4) == pytest.approx(0.0037 / (2 * math.pi * 1e-4))
    beta, gamma = lorentz_factors(100e3)
    assert normalized_emittance_nm(100, 1e-4, 100e3) == pytest.approx(beta * gamma * 100e-6 * 1e-4 * 1e9)
