"""Statistical helpers and common checks of the core test suite."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from nutel.toy import C_VACUUM, SimData

#: Tolerance of the statistical checks in standard deviations (the tests must pass for any seed and torch version).
N_SIGMA = 5.0

#: sqrt(n) D of the Kolmogorov-Smirnov test exceeds this value with probability below 0.1 % for a correct pdf.
KS_LIMIT = 1.95


def assert_binomial(k: float, n: float, p: float, n_sigma: float = N_SIGMA) -> None:
    """Assert that ``k`` successes in ``n`` independent trials are compatible with the probability ``p``."""
    sigma = math.sqrt(p * (1 - p) / n)
    assert abs(k / n - p) < n_sigma * sigma, f"fraction {k / n:.5f}, expected {p:.5f} (sigma {sigma:.2g})"


def ks_distance(samples: np.ndarray, cdf: Callable[[np.ndarray], np.ndarray]) -> float:
    """Kolmogorov-Smirnov distance ``sqrt(n) D`` between samples and a cumulative distribution function."""
    x = np.sort(np.asarray(samples, dtype=np.float64))
    n = len(x)
    f = cdf(x)
    d = max(np.max(np.arange(1, n + 1) / n - f), np.max(f - np.arange(n) / n))
    return math.sqrt(n) * float(d)


def direct_time(
    mx: np.ndarray,
    mz: np.ndarray,
    x0: np.ndarray,
    z0: np.ndarray,
    dx: np.ndarray,
    dz: np.ndarray,
    t0: np.ndarray,
    n: float,
    n_group: float | None = None,
) -> np.ndarray:
    """Arrival time of direct Cherenkov light of a track (anchor x0, z0 at t0, direction d) at the points (mx, mz).

    Emission at ``s_e = s_perp - rho / tan(theta_c)``, flight ``rho / sin(theta_c)`` at ``c / n_group``; for
    ``n_group = n`` this is ``t0 + (s_perp + rho sqrt(n^2 - 1)) / c``.
    """
    ng = n if n_group is None else n_group
    mx, mz, x0, z0, dx, dz, t0 = (np.asarray(v, dtype=np.float64) for v in (mx, mz, x0, z0, dx, dz, t0))
    s_perp = (mx - x0) * dx + (mz - z0) * dz
    rho = np.abs((mx - x0) * dz - (mz - z0) * dx)
    theta = math.acos(1.0 / n)
    return t0 + (s_perp - rho / math.tan(theta)) / C_VACUUM + rho / math.sin(theta) * ng / C_VACUUM


def check_conservation(data: SimData) -> None:
    """Exact photon bookkeeping of every event and consistency of hits, pulses and counts."""
    p = data.photons
    np.testing.assert_array_equal(p.n_emitted, p.n_emitted_track + p.n_emitted_burst + p.n_emitted_cascade)
    ended = p.n_detected + p.n_absorbed_module + p.n_absorbed + p.n_escaped + p.n_lost
    np.testing.assert_array_equal(p.n_emitted, ended)
    np.testing.assert_array_equal(p.n_detected, p.n_detected_direct + p.n_detected_scattered)
    np.testing.assert_array_equal(p.n_hits, p.n_detected - p.n_out_of_window + p.n_noise)
    pulses = data.pulses
    assert pulses.n_events == len(data)
    np.testing.assert_array_equal(pulses.counts(), p.n_pulses)
    charge = np.bincount(pulses.event_index(), weights=pulses.charge, minlength=len(data))
    np.testing.assert_array_equal(charge, p.n_hits)
    noise = np.bincount(pulses.event_index(), weights=pulses.n_noise, minlength=len(data))
    np.testing.assert_array_equal(noise, p.n_noise)
    assert np.all(pulses.charge >= 1)
    assert np.all(pulses.n_noise <= pulses.charge)
    _check_sorted(pulses.event_index(), pulses.t)
    if data.hits is not None:
        h = data.hits
        np.testing.assert_array_equal(h.counts(), p.n_hits)
        np.testing.assert_array_equal(np.bincount(h.event_index()[h.is_noise], minlength=len(data)), p.n_noise)
        np.testing.assert_array_equal(h.is_noise, h.source == 3)
        assert np.all(h.n_scatter[h.is_noise] == -1)
        assert np.all(np.isnan(h.path_length[h.is_noise]))
        assert np.all(np.isfinite(h.path_length[~h.is_noise]))
        _check_sorted(h.event_index(), h.t)
        signal = ~h.is_noise
        direct = np.bincount(h.event_index()[signal & (h.n_scatter == 0)], minlength=len(data))
        assert np.all(direct <= p.n_detected_direct)


def _check_sorted(event: np.ndarray, t: np.ndarray) -> None:
    assert np.all(np.diff(event) >= 0)
    same = np.diff(event) == 0
    assert np.all(np.diff(t.astype(np.float64))[same] >= 0)


def hits_on_own_geometry(data: SimData) -> bool:
    """True if every hit and pulse lies on a module of its event's geometry (string, module, x, z)."""
    for table in (data.hits, data.pulses):
        if table is None:
            continue
        ev = table.event_index()
        gi = data.truth.geometry_index[ev]
        for k, det in enumerate(data.detectors):
            sel = gi == k
            row = det.geometry.module_index(table.string[sel], table.module[sel])
            if np.any(row < 0):
                return False
            if not (
                np.array_equal(det.geometry.x[row], table.x[sel]) and np.array_equal(det.geometry.z[row], table.z[sel])
            ):
                return False
    return True
