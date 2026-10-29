"""Events: the prior per event type, the muon segments, the stochastic losses and the photon emission.

Event types (``truth.type``): 0 ``track`` (through-going muon), 1 ``starting`` (nu_mu CC: cascade at the vertex and a
muon starting there), 2 ``cascade`` (nu_e CC or NC: a point-like shower), 3 ``atmospheric`` (a down-going muon).

The prior (:func:`sample_prior`, all ranges in :class:`~nutel.toy.config.Prior`)::

    track        direction angle ~ U(0, 2 pi); anchor a = b n_perp (closest point of the line to the origin) with
                 b ~ U(-R_gen, R_gen), n_perp = (-dir_z, dir_x), R_gen = half-diagonal of the instrumented area +
                 impact_margin; E log-uniform; t0 (time at the anchor) uniform
    starting     vertex uniform in the instrumented area + vertex_margin; direction as track; E_nu log-uniform;
                 inelasticity p(y) ~ 1 + eps (1 - y)^2 on [0, 1] (or uniform): cascade E_cas = y E_nu at the vertex,
                 muon E_mu = (1 - y) E_nu from the vertex
    cascade      vertex and direction as starting; E log-uniform
    atmospheric  as track, but down-going: p(cos zenith) ~ cos z exp(K1(h) / cos z) (Becherini et al. 2006, depth h)
                 or uniform in cos(zenith); azimuth 0 or pi with equal probability

Muons emit light inside the generation box (instrumented area + track_margin): a through-going muon from its entry
to its exit, a starting muon from the vertex to its exit (:func:`muon_segments`). Along the way, stochastic losses
(:func:`sample_bursts`) occur at the rate ``b ln(1 / v_min)`` per metre; each burst takes the fraction
``v ~ 1/v`` on ``[v_min, 1]`` of the current muon energy and becomes a point-like cascade; in between, the muon loses
``a`` GeV per metre; it stops (and emits no more light) when its energy drops below ``muon_stop_energy``.

Light (:func:`emit_photons`, torch): ``Poisson(Y_t L)`` photons uniform along the muon segment at
``+-(theta_c + N(0, sigma_c^2))`` to the direction (the two rays of the two-dimensional Cherenkov "cone"), emitted
at ``t0 + s / c``; ``Poisson(Y_c E)`` photons per cascade (vertex cascade and bursts) at ``+-theta`` to the cascade
axis, with ``theta`` from the Raedel-Wiebusch angular profile (or ``theta_c + N(0, sigma_cas^2)``), all at the cascade
time. Source codes: 0 track, 1 burst, 2 cascade.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .config import C_VACUUM, EVENT_TYPES, Emission, Physics, Prior, SimConfig
from .geometry import Geometry

__all__ = [
    "ATMOSPHERIC",
    "CASCADE",
    "RADEL_WIEBUSCH",
    "STARTING",
    "TRACK",
    "Photons",
    "Sources",
    "angle_from_zenith_azimuth",
    "atmospheric_cos_zenith",
    "box_interval",
    "direction_from_angle",
    "emit_photons",
    "generation_box",
    "icecube_inelasticity",
    "muon_segments",
    "photon_numbers",
    "radel_wiebusch_angle",
    "sample_bursts",
    "sample_prior",
    "type_codes",
    "zenith_azimuth",
]

TRACK, STARTING, CASCADE, ATMOSPHERIC = 0, 1, 2, 3

#: Light-source codes of photons and hits.
SOURCE_TRACK, SOURCE_BURST, SOURCE_CASCADE, SOURCE_NOISE = 0, 1, 2, 3


# ----------------------------------------------------------------------------------------------------------------------
# Directions


def direction_from_angle(angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the unit direction ``(cos angle, sin angle)`` of the direction angle ``atan2(dir_z, dir_x)`` [rad]."""
    angle = np.asarray(angle, dtype=np.float64)
    return np.cos(angle), np.sin(angle)


def zenith_azimuth(dir_x: np.ndarray, dir_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zenith ``arccos(-dir_z)`` in [0, pi] and azimuth (0 if coming from +x, i.e. ``dir_x < 0``, else pi) [rad]."""
    dir_x = np.asarray(dir_x, dtype=np.float64)
    dir_z = np.asarray(dir_z, dtype=np.float64)
    zenith = np.arccos(np.clip(-dir_z, -1.0, 1.0))
    azimuth = np.where(dir_x < 0, 0.0, np.pi)
    return zenith, azimuth


def angle_from_zenith_azimuth(zenith: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """Direction angle ``atan2(dir_z, dir_x)`` [rad] of a zenith and an azimuth (0: from +x, pi: from -x) [rad]."""
    zenith = np.asarray(zenith, dtype=np.float64)
    from_plus_x = np.cos(np.asarray(azimuth, dtype=np.float64)) > 0
    dir_z = -np.cos(zenith)
    dir_x = np.where(from_plus_x, -np.sin(zenith), np.sin(zenith))
    return np.arctan2(dir_z, dir_x)


# ----------------------------------------------------------------------------------------------------------------------
# The prior


def _log_uniform(u: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return lo * (hi / lo) ** u


def icecube_inelasticity(u: np.ndarray, eps: float) -> np.ndarray:
    """Inelasticity ``y`` with ``p(y) ~ 1 + eps (1 - y)^2`` on [0, 1] from uniform numbers ``u`` (inverse CDF).

    With ``w = 1 - y`` the CDF is ``(w + eps w^3 / 3) / (1 + eps / 3)``; the cubic has one real root (Cardano).
    The mean is ``1 - (1/2 + eps/4) / (1 + eps/3)`` (0.4375 for ``eps = 1``).
    """
    u = np.asarray(u, dtype=np.float64)
    if eps == 0:
        return 1.0 - u
    c = u * (1.0 + eps / 3.0)
    p, q = 3.0 / eps, -3.0 * c / eps  # w^3 + p w + q = 0
    root = np.sqrt(q * q / 4.0 + p**3 / 27.0)
    w = np.cbrt(-q / 2.0 + root) + np.cbrt(-q / 2.0 - root)
    return np.clip(1.0 - w, 0.0, 1.0)


def atmospheric_cos_zenith(u: np.ndarray, depth_kmwe: float, bounds: tuple[float, float]) -> np.ndarray:
    """``cos(zenith)`` of atmospheric muons at depth ``h``: ``p(c) ~ c exp(K1 / c)``, ``K1 = -0.581 h + 0.034``.

    The parametrisation of the muon intensity under water of Becherini et al. (2006), ``K0 cos z exp(K1 sec z)``,
    sampled by a tabulated inverse CDF on ``bounds``.
    """
    lo, hi = bounds
    if hi <= lo:
        return np.full(np.shape(u), lo, dtype=np.float64)
    k1 = -0.581 * depth_kmwe + 0.034
    grid = np.linspace(lo, hi, 4097)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        pdf = np.where(grid > 0, grid * np.exp(k1 / grid), 0.0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    return np.interp(np.asarray(u, dtype=np.float64) * cdf[-1], cdf, grid)


#: Parameters (a [1/sr], b, c, d [1/sr]) of the angular profile of electromagnetic cascades (Raedel & Wiebusch 2013).
RADEL_WIEBUSCH = (4.27033, -6.02527, 0.29887, -0.00103)


@functools.lru_cache(maxsize=16)
def _radel_wiebusch_table(refractive_index: float, n_grid: int = 20001) -> tuple[np.ndarray, np.ndarray]:
    """Tabulated CDF of the polar angle of cascade light, ``(a exp(b |cos theta - 1/n|^c) + d) sin theta``."""
    a, b, c, d = RADEL_WIEBUSCH
    theta = np.linspace(0.0, np.pi, n_grid)
    pdf = np.clip(a * np.exp(b * np.abs(np.cos(theta) - 1.0 / refractive_index) ** c) + d, 0.0, None) * np.sin(theta)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(theta))])
    return theta, cdf / cdf[-1]


def radel_wiebusch_angle(u: np.ndarray, refractive_index: float) -> np.ndarray:
    """Polar emission angles [rad] of cascade photons from uniform numbers ``u`` (tabulated inverse CDF, NumPy)."""
    theta, cdf = _radel_wiebusch_table(float(refractive_index))
    return np.interp(np.asarray(u, dtype=np.float64), cdf, theta)


def sample_prior(
    n: int,
    config: SimConfig | None = None,
    geometry: Geometry | None = None,
    rng: np.random.Generator | int | None = None,
) -> dict[str, np.ndarray]:
    """Draw the event parameters of ``n`` events from the prior.

    Parameters
    ----------
    n
        Number of events.
    config
        Configuration whose ``prior`` is used (default: the ``"full"`` preset).
    geometry
        Detector geometry that sets the instrumented area (default: ``config.geometry()``).
    rng
        NumPy generator, an integer seed or ``None`` (fresh entropy).

    Returns
    -------
    dict
        ``type`` (int8), ``angle`` (rad), ``dir_x``, ``dir_z``, ``x0``, ``z0`` (m, anchor or vertex), ``t0`` (ns),
        ``energy`` (GeV), ``cascade_fraction`` (NaN except for starting tracks); float64 arrays of length ``n``.
    """
    config = config or SimConfig()
    geometry = geometry if geometry is not None else config.geometry()
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    p: Prior = config.prior
    # every random number is drawn for every event, so that a fixed value or another type mix leaves the others alone
    u_type = rng.random(n)
    angle_iso = 2 * np.pi * rng.random(n)
    u_b = rng.random(n)
    u_vx, u_vz = rng.random(n), rng.random(n)
    u_cos, u_side = rng.random(n), rng.random(n)
    u_energy = rng.random(n)
    u_y = rng.random(n)
    u_t0 = rng.random(n)

    cumulative = np.cumsum(p.type_probabilities)
    kind = np.minimum(np.searchsorted(cumulative / cumulative[-1], u_type, side="right"), 3).astype(np.int8)

    # directions
    if p.atmospheric_zenith == "depth":
        cos_zenith = atmospheric_cos_zenith(u_cos, p.depth_kmwe, p.atmospheric_cos_zenith_range)
    else:
        lo, hi = p.atmospheric_cos_zenith_range
        cos_zenith = lo + u_cos * (hi - lo)
    from_plus_x = u_side < 0.5
    atm_dx = np.where(from_plus_x, -1.0, 1.0) * np.sqrt(1.0 - cos_zenith**2)
    atm_angle = np.arctan2(-cos_zenith, atm_dx)
    angle = np.where(kind == ATMOSPHERIC, atm_angle, angle_iso)
    if p.fixed_angle is not None:
        angle = np.full(n, math.radians(p.fixed_angle))
    dir_x, dir_z = direction_from_angle(angle)
    angle = np.arctan2(dir_z, dir_x)

    # anchor (tracks) or vertex (starting, cascade)
    x_min, x_max, z_min, z_max = geometry.bounds
    r_gen = geometry.half_diagonal + p.impact_margin
    b = (2 * u_b - 1) * r_gen
    vx = x_min - p.vertex_margin + u_vx * (x_max - x_min + 2 * p.vertex_margin)
    vz = z_min - p.vertex_margin + u_vz * (z_max - z_min + 2 * p.vertex_margin)
    if p.fixed_vertex is not None:
        vx = np.full(n, p.fixed_vertex[0])
        vz = np.full(n, p.fixed_vertex[1])
        along = vx * dir_x + vz * dir_z  # project the given point on the line onto its anchor
        ax, az = vx - along * dir_x, vz - along * dir_z
    else:
        ax, az = -b * dir_z, b * dir_x
    is_line = (kind == TRACK) | (kind == ATMOSPHERIC)
    x0 = np.where(is_line, ax, vx)
    z0 = np.where(is_line, az, vz)

    # energies, inelasticity, time
    energy = np.select(
        [kind == TRACK, kind == STARTING, kind == CASCADE],
        [
            _log_uniform(u_energy, p.track_energy_range),
            _log_uniform(u_energy, p.starting_energy_range),
            _log_uniform(u_energy, p.cascade_energy_range),
        ],
        _log_uniform(u_energy, p.atmospheric_energy_range),
    )
    if p.fixed_energy is not None:
        energy = np.full(n, p.fixed_energy)
    if p.inelasticity == "icecube":
        y = icecube_inelasticity(u_y, p.inelasticity_eps)
    else:
        y = p.inelasticity_range[0] + u_y * (p.inelasticity_range[1] - p.inelasticity_range[0])
    if p.fixed_inelasticity is not None:
        y = np.full(n, p.fixed_inelasticity)
    cascade_fraction = np.where(kind == STARTING, y, np.nan)
    t0 = p.t0_range[0] + u_t0 * (p.t0_range[1] - p.t0_range[0])
    if p.fixed_t0 is not None:
        t0 = np.full(n, p.fixed_t0)
    return {
        "type": kind,
        "angle": angle,
        "dir_x": dir_x,
        "dir_z": dir_z,
        "x0": x0,
        "z0": z0,
        "t0": t0,
        "energy": energy,
        "cascade_fraction": cascade_fraction,
    }


# ----------------------------------------------------------------------------------------------------------------------
# Muon segments and stochastic losses


def box_interval(
    px: np.ndarray, pz: np.ndarray, dx: np.ndarray, dz: np.ndarray, box: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Parameters ``[s_in, s_out]`` where the lines ``p + s d`` are inside the box ``(x_min, x_max, z_min, z_max)``.

    Lines that miss the box get ``s_in > s_out`` (an empty interval).
    """
    x_min, x_max, z_min, z_max = box
    with np.errstate(divide="ignore", invalid="ignore"):
        lo, hi = [], []
        for p, d, a, b in ((px, dx, x_min, x_max), (pz, dz, z_min, z_max)):
            t1, t2 = (a - p) / d, (b - p) / d
            parallel = d == 0
            inside = (p >= a) & (p <= b)
            t_lo = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
            t_hi = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
            lo.append(t_lo)
            hi.append(t_hi)
    return np.maximum(lo[0], lo[1]), np.minimum(hi[0], hi[1])


def generation_box(geometry: Geometry, prior: Prior) -> tuple[float, float, float, float]:
    """Return the generation box: the instrumented area plus ``track_margin`` on every side."""
    x_min, x_max, z_min, z_max = geometry.bounds
    m = prior.track_margin
    return x_min - m, x_max + m, z_min - m, z_max + m


def muon_segments(
    kind: np.ndarray,
    x0: np.ndarray,
    z0: np.ndarray,
    dir_x: np.ndarray,
    dir_z: np.ndarray,
    box: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Start and end ``(s_start, s_end)`` of the light-emitting muon segment along ``x0 + s d`` (m).

    Through-going tracks: the part of the line inside the generation box. Starting tracks: from the vertex (s = 0) to
    the exit. Cascades (and lines that miss the box): an empty segment ``s_start = s_end = 0``.
    """
    s_in, s_out = box_interval(x0, z0, dir_x, dir_z, box)
    is_line = (kind == TRACK) | (kind == ATMOSPHERIC)
    start = np.where(is_line, s_in, 0.0)
    end = np.where(kind == CASCADE, 0.0, s_out)
    empty = ~(end > start) | ~np.isfinite(start) | ~np.isfinite(end)
    return np.where(empty, 0.0, start), np.where(empty, 0.0, end)


@dataclass
class BurstSample:
    """Stochastic losses of a set of events: the bursts (ordered by event and along the track) and the stop points."""

    event: np.ndarray  #: event of each burst (index into the input arrays, int64)
    s: np.ndarray  #: position of each burst along the line [m]
    energy: np.ndarray  #: energy of each burst [GeV]
    s_end: np.ndarray  #: end of the light-emitting segment of every event [m] (earlier if the muon stopped)
    energy_end: np.ndarray  #: muon energy at the end of the segment [GeV]
    stopped: np.ndarray  #: True if the muon stopped before the end of its segment (bool)


def sample_bursts(
    s_start: np.ndarray,
    s_end: np.ndarray,
    energy_track: np.ndarray,
    emission: Emission,
    physics: Physics,
    rng: np.random.Generator,
) -> BurstSample:
    """Stochastic losses along the muon segments ``[s_start, s_end]`` with initial energies ``energy_track``.

    Bursts occur at the rate ``b ln(1/v_min)`` per metre (distances ``~ Exp``); a burst takes the fraction
    ``v = v_min^(1 - u)`` (``p(v) ~ 1/v`` on ``[v_min, 1]``) of the current energy. Between bursts the muon loses
    ``energy_loss_a`` GeV per metre. It stops (and its segment ends) when its energy falls below
    ``muon_stop_energy``. With ``physics.stochastic = False`` there are no bursts.
    """
    rate = emission.burst_rate if physics.stochastic else 0.0
    a = emission.energy_loss_a
    e_stop = emission.muon_stop_energy
    log_v_min = math.log(emission.stochastic_v_min)
    s = np.asarray(s_start, dtype=np.float64).copy()
    end = np.asarray(s_end, dtype=np.float64).copy()
    end_box = end.copy()
    energy = np.asarray(energy_track, dtype=np.float64).copy()
    active = (end > s) & (energy >= e_stop) & ((rate > 0) | (a > 0))
    # a muon that starts below the stop energy emits nothing
    end = np.where((end > s) & (energy < e_stop), s, end)
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    while np.any(active):
        i = np.flatnonzero(active)
        step = rng.exponential(1.0 / rate, len(i)) if rate > 0 else np.full(len(i), np.inf)
        to_stop = (energy[i] - e_stop) / a if a > 0 else np.full(len(i), np.inf)
        to_end = end[i] - s[i]
        burst = step < np.minimum(to_stop, to_end)
        # no burst before the end of the segment or the stop by continuous losses
        quiet = i[~burst]
        stops = quiet[to_stop[~burst] < to_end[~burst]]
        energy[quiet] -= a * np.minimum(to_stop[~burst], to_end[~burst]) if a > 0 else 0.0
        end[stops] = s[stops] + (to_stop[~burst][to_stop[~burst] < to_end[~burst]])
        active[quiet] = False
        # bursts
        j = i[burst]
        if len(j) == 0:
            continue
        s[j] += step[burst]
        energy[j] -= a * step[burst]
        v = np.exp(log_v_min * (1.0 - rng.random(len(j))))
        e_burst = v * energy[j]
        energy[j] -= e_burst
        parts.append((j, s[j].copy(), e_burst))
        stopped = j[energy[j] < e_stop]
        end[stopped] = s[stopped]
        active[stopped] = False
    if parts:
        event = np.concatenate([p[0] for p in parts])
        pos = np.concatenate([p[1] for p in parts])
        e = np.concatenate([p[2] for p in parts])
        order = np.lexsort((pos, event))
        event, pos, e = event[order], pos[order], e[order]
    else:
        event, pos, e = np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
    stopped = (end_box > np.asarray(s_start)) & (end < end_box)
    return BurstSample(
        event=event.astype(np.int64), s=pos, energy=e, s_end=end, energy_end=np.maximum(energy, 0.0), stopped=stopped
    )


def photon_numbers(mean: np.ndarray, statistics: str, rng: np.random.Generator) -> np.ndarray:
    """Draw the photon numbers of sources with the given mean: ``Poisson(mean)`` or ``round(mean)`` (int64)."""
    mean = np.asarray(mean, dtype=np.float64)
    if statistics == "fixed":
        return np.rint(mean).astype(np.int64)
    return rng.poisson(mean).astype(np.int64)


# ----------------------------------------------------------------------------------------------------------------------
# Photon emission (torch)


@dataclass
class Sources:
    """Light sources of a chunk of events (NumPy arrays; events are indices into the chunk).

    Muon segments start at ``(seg_x, seg_z)`` at the time ``seg_t`` and run along ``(seg_dx, seg_dz)`` for
    ``seg_length``; point sources (vertex cascades and bursts) emit around the axis ``(pt_dx, pt_dz)``.
    """

    seg_event: np.ndarray
    seg_x: np.ndarray
    seg_z: np.ndarray
    seg_dx: np.ndarray
    seg_dz: np.ndarray
    seg_length: np.ndarray
    seg_t: np.ndarray
    seg_n: np.ndarray
    pt_event: np.ndarray
    pt_x: np.ndarray
    pt_z: np.ndarray
    pt_dx: np.ndarray
    pt_dz: np.ndarray
    pt_t: np.ndarray
    pt_n: np.ndarray
    pt_source: np.ndarray

    @property
    def n_photons(self) -> int:
        """Total number of photons."""
        return int(self.seg_n.sum() + self.pt_n.sum())


@dataclass
class Photons:
    """Emitted photons (torch tensors of length n_photons, ordered by event)."""

    event: torch.Tensor  #: event of the photon, index into the chunk (int64)
    source: torch.Tensor  #: source code: 0 track, 1 burst, 2 cascade (int8)
    x: torch.Tensor  #: emission point [m] (float64)
    z: torch.Tensor
    ux: torch.Tensor  #: unit direction (float64)
    uz: torch.Tensor
    t: torch.Tensor  #: emission time [ns] (float64)

    def __len__(self) -> int:
        return int(self.event.numel())


def _tensor(values: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(values), device=device).to(dtype)


def _rays(
    base_x: torch.Tensor,
    base_z: torch.Tensor,
    theta_c: float,
    spread: float,
    generator: torch.Generator,
    table: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directions at ``+-theta`` to the unit vectors ``(base_x, base_z)``.

    ``theta = theta_c + N(0, spread^2)`` [rad], or drawn from a tabulated CDF ``table = (theta, cdf)``.
    """
    n = base_x.numel()
    device = base_x.device
    side = torch.where(torch.rand(n, generator=generator, device=device, dtype=torch.float64) < 0.5, -1.0, 1.0)
    if table is None:
        delta = theta_c + spread * torch.randn(n, generator=generator, device=device, dtype=torch.float64)
    else:
        grid = torch.as_tensor(table[0], device=device, dtype=torch.float64)
        cdf = torch.as_tensor(table[1], device=device, dtype=torch.float64)
        u = torch.rand(n, generator=generator, device=device, dtype=torch.float64)
        k = torch.clamp(torch.searchsorted(cdf, u), 1, len(cdf) - 1)
        c0, c1 = cdf[k - 1], cdf[k]
        w = torch.where(c1 > c0, (u - c0) / (c1 - c0), torch.zeros_like(u))
        delta = grid[k - 1] + w * (grid[k] - grid[k - 1])
    phi = side * delta
    cos, sin = torch.cos(phi), torch.sin(phi)
    return base_x * cos - base_z * sin, base_x * sin + base_z * cos


def emit_photons(
    sources: Sources,
    emission: Emission,
    refractive_index: float,
    generator: torch.Generator,
) -> Photons:
    """Create the photons of the sources of a chunk on the generator's device.

    Track photons: position uniform along the segment, time ``seg_t + s / c``, direction at
    ``+-(theta_c + N(0, cherenkov_spread^2))``. Point sources: at their position and time, direction at ``+-theta``
    to the axis, ``theta`` from the cascade profile of ``emission``. The photons are returned ordered by event (track
    photons of an event first, then those of its point sources in their order).
    """
    device = generator.device
    theta_c = math.acos(1.0 / refractive_index)
    # muon segments
    n_seg = _tensor(sources.seg_n, device, torch.int64)
    seg = torch.repeat_interleave(torch.arange(len(sources.seg_n), device=device), n_seg)
    s = torch.rand(seg.numel(), generator=generator, device=device, dtype=torch.float64)
    s = s * _tensor(sources.seg_length, device)[seg]
    dx, dz = _tensor(sources.seg_dx, device)[seg], _tensor(sources.seg_dz, device)[seg]
    tx = _tensor(sources.seg_x, device)[seg] + s * dx
    tz = _tensor(sources.seg_z, device)[seg] + s * dz
    tt = _tensor(sources.seg_t, device)[seg] + s / C_VACUUM
    tux, tuz = _rays(dx, dz, theta_c, math.radians(emission.cherenkov_spread), generator)
    t_event = _tensor(sources.seg_event, device, torch.int64)[seg]
    # point sources
    n_pt = _tensor(sources.pt_n, device, torch.int64)
    pt = torch.repeat_interleave(torch.arange(len(sources.pt_n), device=device), n_pt)
    pdx, pdz = _tensor(sources.pt_dx, device)[pt], _tensor(sources.pt_dz, device)[pt]
    table = _radel_wiebusch_table(float(refractive_index)) if emission.cascade_profile == "radel-wiebusch" else None
    pux, puz = _rays(pdx, pdz, theta_c, math.radians(emission.cascade_spread), generator, table)
    p_event = _tensor(sources.pt_event, device, torch.int64)[pt]
    event = torch.cat([t_event, p_event])
    order = torch.sort(event, stable=True).indices
    return Photons(
        event=event[order],
        source=torch.cat(
            [
                torch.full((seg.numel(),), SOURCE_TRACK, dtype=torch.int8, device=device),
                _tensor(sources.pt_source, device, torch.int8)[pt],
            ]
        )[order],
        x=torch.cat([tx, _tensor(sources.pt_x, device)[pt]])[order],
        z=torch.cat([tz, _tensor(sources.pt_z, device)[pt]])[order],
        ux=torch.cat([tux, pux])[order],
        uz=torch.cat([tuz, puz])[order],
        t=torch.cat([tt, _tensor(sources.pt_t, device)[pt]])[order],
    )


def type_codes(values: Any) -> np.ndarray:
    """Event type codes (int8) from codes or names (``"track"``, ``"starting"``, ``"cascade"``, ``"atmospheric"``)."""
    arr = np.atleast_1d(np.asarray(values))
    if arr.dtype.kind in "US":
        unknown = set(arr.tolist()) - set(EVENT_TYPES)
        if unknown:
            raise ValueError(f"unknown event types {sorted(unknown)}; choose from {', '.join(EVENT_TYPES)}")
        return np.array([EVENT_TYPES.index(v) for v in arr.tolist()], dtype=np.int8)
    codes = arr.astype(np.int64)
    if np.any((codes < 0) | (codes > 3)) or not np.array_equal(codes, arr):
        raise ValueError("event type codes must be integers 0 (track), 1 (starting), 2 (cascade), 3 (atmospheric)")
    return codes.astype(np.int8)
