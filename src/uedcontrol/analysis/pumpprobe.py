"""Pump-probe bookkeeping: scan order and running averages per delay point."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np


def scan_order(n_points: int, n_runs: int | None, bidirectional: bool = True) -> Iterator[tuple[int, int]]:
    """Yield ``(run, index)`` pairs, runs numbered from 1.

    With ``bidirectional`` the odd runs go forward and the even runs backward,
    so the delay stage never jumps back to the start. ``n_runs`` of ``None``
    (or 0) repeats until the consumer stops iterating.
    """
    if n_points < 1:
        raise ValueError("a scan needs at least one point")
    run = 1
    while not n_runs or run <= n_runs:
        forward = not bidirectional or run % 2 == 1
        indices = range(n_points) if forward else range(n_points - 1, -1, -1)
        for index in indices:
            yield run, index
        run += 1


class PumpProbeAccumulator:
    """Averages the ROI signals of every delay point over all runs so far."""

    def __init__(self, delays_ps: Sequence[float] | np.ndarray) -> None:
        self.delays = np.asarray(delays_ps, dtype=float).ravel()
        n = self.delays.size
        self.counts = np.zeros(n, dtype=np.int64)
        self._total = np.zeros(n)
        self._rms = np.zeros(n)
        self._profile: np.ndarray | None = None

    def add(self, index: int, total: float, rms: float, profile: np.ndarray) -> None:
        profile = np.asarray(profile, dtype=float).ravel()
        if self._profile is None:
            self._profile = np.zeros((self.delays.size, profile.size))
        elif profile.size != self._profile.shape[1]:
            # The ROI size changed: resample onto the length used so far.
            length = self._profile.shape[1]
            profile = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, profile.size), profile)
        self.counts[index] += 1
        self._total[index] += total
        self._rms[index] += rms
        self._profile[index] += profile

    def _mean(self, sums: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            counts = self.counts.reshape((-1,) + (1,) * (sums.ndim - 1))
            return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    @property
    def mean_total(self) -> np.ndarray:
        return self._mean(self._total)

    @property
    def mean_rms(self) -> np.ndarray:
        return self._mean(self._rms)

    @property
    def mean_profile(self) -> np.ndarray | None:
        return None if self._profile is None else self._mean(self._profile)

    @property
    def completed_points(self) -> int:
        return int(self.counts.sum())
