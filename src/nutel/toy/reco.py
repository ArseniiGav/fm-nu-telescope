"""Naive reconstruction and the geometry of direct Cherenkov light, written against plain NumPy arrays.

Conventions (the same as the simulation): positions ``(x, z)`` in metres with ``z`` up, times in nanoseconds, the
direction of TRAVEL as a unit vector ``d = (dir_x, dir_z)`` or as the plain angle ``atan2(dir_z, dir_x)``. Ragged
per-event arrays (hits or pulses) are concatenated and delimited by ``offset`` (length ``n_events + 1``), as in the
data files.

Contents:

* :func:`linefit` / :func:`linefit_events`: the classic LineFit (charge-weighted least squares of position vs time);
* :func:`fit_charge_calibration` and :func:`energy_from_charge`: energy proportional to the total charge;
* :func:`expected_arrival_time`: the arrival time of direct Cherenkov light (track) or of light from a point
  (cascade) at a module, and :func:`time_residuals` / :func:`time_residuals_arrays` against the true event;
* :func:`angular_error`: the angle between two directions in degrees;
* :func:`causal_cleaning`: the causality filter for noise hits;
* geometry helpers: :func:`direction_angle`, :func:`unit_vector`, :func:`track_coordinates`,
  :func:`emission_coordinate`, :func:`direct_light_possible`, :func:`shadowed` (zenith and azimuth:
  :func:`nutel.toy.zenith_azimuth`).
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple, Protocol

import numpy as np
from numpy.typing import ArrayLike

from .config import C_VACUUM

__all__ = [
    "LineFit",
    "angular_error",
    "causal_cleaning",
    "direct_light_possible",
    "direction_angle",
    "emission_coordinate",
    "energy_from_charge",
    "event_index",
    "expected_arrival_time",
    "fit_charge_calibration",
    "linefit",
    "linefit_events",
    "shadowed",
    "time_residuals",
    "time_residuals_arrays",
    "track_coordinates",
    "unit_vector",
]

_TRACK, _STARTING, _CASCADE, _ATMOSPHERIC = 0, 1, 2, 3


class _Ragged(Protocol):
    """Anything with per-row positions and times delimited by event offsets (hits or pulses)."""

    @property
    def offset(self) -> Any: ...

    @property
    def x(self) -> Any: ...

    @property
    def z(self) -> Any: ...

    @property
    def t(self) -> Any: ...


# ----------------------------------------------------------------------------------------------------------------------
# Geometry helpers


def event_index(offset: ArrayLike) -> np.ndarray:
    """Event number of every row of a ragged array.

    Parameters
    ----------
    offset
        Event offsets, length ``n_events + 1``, non-decreasing, starting at 0.

    Returns
    -------
    numpy.ndarray
        Int64 array of length ``offset[-1]``.

    Examples
    --------
    >>> event_index([0, 2, 2, 3])
    array([0, 0, 2])
    """
    offset = _check_offset(offset)
    return np.repeat(np.arange(len(offset) - 1, dtype=np.int64), np.diff(offset))


def direction_angle(dir_x: ArrayLike, dir_z: ArrayLike) -> np.ndarray:
    """Plain direction angle ``atan2(dir_z, dir_x)`` in ``(-pi, pi]`` [rad]."""
    return np.arctan2(np.asarray(dir_z, dtype=np.float64), np.asarray(dir_x, dtype=np.float64))


def unit_vector(angle: ArrayLike) -> np.ndarray:
    """Return the unit vector ``(cos angle, sin angle)`` of a direction angle [rad], shape ``[..., 2]``."""
    a = np.asarray(angle, dtype=np.float64)
    return np.stack([np.cos(a), np.sin(a)], axis=-1)


def track_coordinates(points: ArrayLike, anchor: ArrayLike, direction: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of points relative to a straight line.

    Parameters
    ----------
    points
        Positions ``(x, z)``, shape ``[..., 2]`` [m].
    anchor
        A point of the line, shape ``[..., 2]`` (broadcast against ``points``) [m].
    direction
        Direction of the line, shape ``[..., 2]`` (normalised here).

    Returns
    -------
    s : numpy.ndarray
        Longitudinal coordinate ``(p - a) . d`` of the foot of the perpendicular [m].
    rho : numpy.ndarray
        Signed perpendicular distance ``(p - a) . n`` with ``n = (-d_z, d_x)`` (positive on the left of ``d``) [m].
    """
    p = np.asarray(points, dtype=np.float64)
    a = np.asarray(anchor, dtype=np.float64)
    d = _normalise(direction)
    rel = p - a
    s = rel[..., 0] * d[..., 0] + rel[..., 1] * d[..., 1]
    rho = -rel[..., 0] * d[..., 1] + rel[..., 1] * d[..., 0]
    return s, rho


def emission_coordinate(points: ArrayLike, anchor: ArrayLike, direction: ArrayLike, n: ArrayLike) -> np.ndarray:
    """Longitudinal coordinate on the track of the emission point of the direct Cherenkov light seen at ``points``.

    The light that reaches a point at perpendicular distance ``rho`` leaves the track ``rho cot(theta_c) =
    rho / sqrt(n^2 - 1)`` before the foot of the perpendicular: ``s_e = s_perp - rho / sqrt(n^2 - 1)``. A module with
    ``s_e < 0`` (relative to the vertex of a starting track) gets no direct track light.

    Parameters
    ----------
    points, anchor, direction
        As in :func:`track_coordinates`.
    n
        Refractive index (> 1).

    Returns
    -------
    numpy.ndarray
        ``s_e`` [m], measured from ``anchor`` along ``direction``.
    """
    n_arr = np.asarray(n, dtype=np.float64)
    if np.any(n_arr <= 1):
        raise ValueError("the refractive index must exceed 1 for Cherenkov light")
    s, rho = track_coordinates(points, anchor, direction)
    return s - np.abs(rho) / np.sqrt(n_arr**2 - 1)


def direct_light_possible(
    points: ArrayLike, anchor: ArrayLike, direction: ArrayLike, n: ArrayLike, s_start: ArrayLike, s_end: ArrayLike
) -> np.ndarray:
    """Whether the direct Cherenkov light of a muon can reach ``points``: its emission point lies on the segment.

    The muon emits light only on ``s_start <= s <= s_end`` (``data.truth.s_start``, ``data.truth.s_end``, measured
    from the anchor or vertex). A module whose emission coordinate ``s_e`` (:func:`emission_coordinate`) lies outside
    this segment gets no direct track light, however close it is to the line. Modules hidden behind another module
    (shadowing) are not considered here.

    Parameters
    ----------
    points, anchor, direction, n
        As in :func:`emission_coordinate`.
    s_start, s_end
        Ends of the light-emitting segment [m], broadcast against ``points[..., 0]``; NaN (no muon light) gives False.

    Returns
    -------
    numpy.ndarray
        Boolean array, True where ``s_start <= s_e <= s_end``.

    Examples
    --------
    A track along +x emitting on ``[-100, 100]`` m in water (n = 1.33, ``cot theta_c = 1.14``): the module 20 m
    beside the line at ``x = 0`` receives light from ``s_e = -22.8`` m, the one 200 m away from ``s_e = -228`` m.

    >>> direct_light_possible([[0.0, 20.0], [0.0, 200.0]], [0.0, 0.0], [1.0, 0.0], 1.33, -100.0, 100.0)
    array([ True, False])
    """
    s_e = emission_coordinate(points, anchor, direction, n)
    lo = np.asarray(s_start, dtype=np.float64)
    hi = np.asarray(s_end, dtype=np.float64)
    return np.asarray((s_e >= lo) & (s_e <= hi))


def shadowed(start: ArrayLike, end: ArrayLike, modules: ArrayLike, radius: float, *, margin: float = 2.0) -> np.ndarray:
    """Whether another module stands in the way of the straight light from ``start`` to the module at ``end``.

    Modules are opaque discs of radius ``radius``. The light is shadowed (fully or in part) where the centre of any
    other module comes closer than ``margin x radius`` to the segment between ``start`` (an emission point, e.g.
    ``anchor + s_e d`` of :func:`emission_coordinate`, or a cascade vertex) and ``end``. ``margin = 2`` flags every
    module whose disc overlaps the beam of width ``2 radius`` that reaches ``end`` (partial shadows included);
    ``margin = 1`` only the modules that cross the central ray.

    Parameters
    ----------
    start, end
        Positions, shape ``[..., 2]`` (broadcast against each other) [m]; ``end`` is a module position.
    modules
        All module positions of the detector, shape ``[M, 2]`` [m].
    radius
        Module radius [m].
    margin
        Distance limit in units of ``radius``.

    Returns
    -------
    numpy.ndarray
        Boolean array of the broadcast shape of ``start`` and ``end`` without the last axis.

    Examples
    --------
    >>> modules = [[10.0, 0.0], [20.0, 0.0], [20.0, 5.0]]
    >>> shadowed([0.0, 0.0], modules, modules, 0.5)
    array([False,  True, False])
    """
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    a, b = np.broadcast_arrays(a, b)
    q_all = np.asarray(modules, dtype=np.float64).reshape(-1, 2)
    seg = b - a
    length2 = np.sum(seg**2, axis=-1)
    limit = margin * float(radius)
    out = np.zeros(a.shape[:-1], dtype=bool)
    for q in q_all:
        rel = q - a
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.sum(rel * seg, axis=-1) / length2
        between = (t > 0) & (t < 1)
        foot = a + np.clip(np.nan_to_num(t), 0.0, 1.0)[..., None] * seg
        close = np.hypot(q[0] - foot[..., 0], q[1] - foot[..., 1]) < limit
        itself = np.hypot(q[0] - b[..., 0], q[1] - b[..., 1]) < 1e-6
        out |= between & close & ~itself
    return out


# ----------------------------------------------------------------------------------------------------------------------
# LineFit


class LineFit(NamedTuple):
    """Result of :func:`linefit` (scalars and shape ``[2]``) or :func:`linefit_events` (one row per event).

    Attributes
    ----------
    anchor
        Charge-weighted mean position, the point of the fitted line at ``time`` [m], shape ``[..., 2]``.
    direction
        Unit vector of the fitted velocity, shape ``[..., 2]`` (NaN if undefined).
    speed
        Magnitude of the fitted velocity [m/ns] (the speed of light in vacuum is 0.2998 m/ns).
    time
        Charge-weighted mean time [ns].
    angle
        Direction angle ``atan2(direction_z, direction_x)`` [rad].
    """

    anchor: np.ndarray
    direction: np.ndarray
    speed: np.ndarray
    time: np.ndarray
    angle: np.ndarray


def linefit_events(
    offset: ArrayLike, x: ArrayLike, z: ArrayLike, t: ArrayLike, weights: ArrayLike | None = None
) -> LineFit:
    """LineFit of every event of a ragged array (vectorised over events).

    For each event the hit positions are fitted by a straight line moving with constant velocity,
    ``r(t) = r_bar + v (t - t_bar)``, by weighted least squares. The solution per coordinate is::

        v_x = cov_w(x, t) / var_w(t),    v_z = cov_w(z, t) / var_w(t)

    with the weighted means ``r_bar``, ``t_bar`` and weighted (co)variances. The direction is ``v / |v|`` and the
    speed ``|v|``. The fit ignores the Cherenkov geometry, so the speed is not the speed of the particle; the
    direction is the time-ordered axis of the hits.

    Parameters
    ----------
    offset
        Event offsets, length ``n_events + 1``.
    x, z
        Hit (or pulse) positions [m], length ``offset[-1]``.
    t
        Hit times [ns].
    weights
        Non-negative weights, e.g. the pulse charges; ``None`` weighs every row equally.

    Returns
    -------
    LineFit
        Arrays with one row per event. Events with fewer than two rows, zero total weight or no spread in time
        get NaN.
    """
    offset = _check_offset(offset)
    n_events = len(offset) - 1
    x_arr, z_arr, t_arr = (np.asarray(v, dtype=np.float64) for v in (x, z, t))
    w = np.ones_like(t_arr) if weights is None else np.asarray(weights, dtype=np.float64)
    for name, values in (("x", x_arr), ("z", z_arr), ("t", t_arr), ("weights", w)):
        if values.shape != (offset[-1],):
            raise ValueError(f"{name} must have length offset[-1] = {offset[-1]}, got shape {values.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    e = event_index(offset)
    total = np.bincount(e, w, minlength=n_events)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_bar = np.bincount(e, w * t_arr, minlength=n_events) / total
        x_bar = np.bincount(e, w * x_arr, minlength=n_events) / total
        z_bar = np.bincount(e, w * z_arr, minlength=n_events) / total
        dt = t_arr - t_bar[e]
        var_t = np.bincount(e, w * dt * dt, minlength=n_events) / total
        cov_x = np.bincount(e, w * dt * (x_arr - x_bar[e]), minlength=n_events) / total
        cov_z = np.bincount(e, w * dt * (z_arr - z_bar[e]), minlength=n_events) / total
        scale = np.maximum(np.bincount(e, w * t_arr * t_arr, minlength=n_events) / total, 1.0)
        ok = (np.diff(offset) >= 2) & (total > 0) & (var_t > 1e-12 * scale)
        vx = np.where(ok, cov_x / np.where(ok, var_t, 1.0), np.nan)
        vz = np.where(ok, cov_z / np.where(ok, var_t, 1.0), np.nan)
        speed = np.hypot(vx, vz)
        direction = np.stack([vx / speed, vz / speed], axis=-1)
    direction[~(speed > 0)] = np.nan
    anchor = np.stack([x_bar, z_bar], axis=-1)
    return LineFit(
        anchor=anchor,
        direction=direction,
        speed=speed,
        time=t_bar,
        angle=np.arctan2(direction[:, 1], direction[:, 0]),
    )


def linefit(x: ArrayLike, z: ArrayLike, t: ArrayLike, weights: ArrayLike | None = None) -> LineFit:
    """LineFit of one event: the least-squares straight line of the hit positions vs time.

    See :func:`linefit_events` for the formulae.

    Parameters
    ----------
    x, z
        Hit positions [m].
    t
        Hit times [ns].
    weights
        Non-negative weights (e.g. charges); ``None`` = equal weights.

    Returns
    -------
    LineFit
        ``anchor`` and ``direction`` of shape ``[2]``; ``speed``, ``time`` and ``angle`` as 0-d arrays.

    Examples
    --------
    Points on a line moving with the speed of light along +x:

    >>> t = np.array([0.0, 10.0, 20.0])
    >>> fit = linefit(0.3 * t, np.zeros(3), t)
    >>> fit.direction, round(float(fit.speed), 4)
    (array([1., 0.]), 0.3)
    """
    t_arr = np.asarray(t, dtype=np.float64).ravel()
    fit = linefit_events(
        [0, len(t_arr)],
        np.asarray(x, dtype=np.float64).ravel(),
        np.asarray(z, dtype=np.float64).ravel(),
        t_arr,
        None if weights is None else np.asarray(weights, dtype=np.float64).ravel(),
    )
    return LineFit(*(np.asarray(v[0]) for v in fit))


# ----------------------------------------------------------------------------------------------------------------------
# Energy from charge


def energy_from_charge(charge: ArrayLike, calibration: float) -> np.ndarray:
    """Naive energy estimate proportional to the total charge, ``E = k Q``.

    Parameters
    ----------
    charge
        Total charge per event [p.e.] (e.g. the ``total_charge`` column of
        :func:`~nutel.toy.features.event_features`).
    calibration
        The constant ``k`` [GeV per p.e.] from :func:`fit_charge_calibration`.

    Returns
    -------
    numpy.ndarray
        Estimated energy [GeV] (0 for events without charge).
    """
    if not calibration > 0:
        raise ValueError(f"the calibration constant must be positive, got {calibration}")
    return calibration * np.asarray(charge, dtype=np.float64)


def fit_charge_calibration(
    charge: ArrayLike, energy: ArrayLike, method: Literal["log_median", "log_mean", "least_squares"] = "log_median"
) -> float:
    """Fit the one constant ``k`` of :func:`energy_from_charge` on a data set with known energies.

    Parameters
    ----------
    charge
        Total charge per event [p.e.]; events with zero charge are ignored.
    energy
        True energy per event [GeV].
    method
        ``"log_median"`` (default): ``k = exp(median(log(E / Q)))``, robust to outliers and appropriate for energies
        spread over decades; ``"log_mean"``: the same with the mean (least squares in ``log E``);
        ``"least_squares"``: ``k = sum(E Q) / sum(Q^2)`` (dominated by the largest events).

    Returns
    -------
    float
        ``k`` [GeV per p.e.].
    """
    q = np.asarray(charge, dtype=np.float64)
    e = np.asarray(energy, dtype=np.float64)
    if q.shape != e.shape:
        raise ValueError(f"charge and energy must have the same shape, got {q.shape} and {e.shape}")
    use = (q > 0) & (e > 0) & np.isfinite(q) & np.isfinite(e)
    if not use.any():
        raise ValueError("no event with positive charge and energy")
    q, e = q[use], e[use]
    if method == "log_median":
        return float(np.exp(np.median(np.log(e / q))))
    if method == "log_mean":
        return float(np.exp(np.mean(np.log(e / q))))
    if method == "least_squares":
        return float(np.sum(e * q) / np.sum(q * q))
    raise ValueError(f"unknown method {method!r}; choose 'log_median', 'log_mean' or 'least_squares'")


# ----------------------------------------------------------------------------------------------------------------------
# Direct-light arrival times


def expected_arrival_time(
    module_xz: ArrayLike,
    anchor: ArrayLike,
    direction: ArrayLike,
    t0: ArrayLike,
    n: ArrayLike,
    *,
    n_group: ArrayLike | None = None,
    event_type: ArrayLike | None = None,
    radius: ArrayLike = 0.0,
    c: float = C_VACUUM,
) -> np.ndarray:
    """Arrival time of direct (unscattered) light at a module.

    Track (the default, ``event_type`` None, 0 or 3). The muon moves along ``d`` with the speed of light ``c`` and is
    at the anchor ``a`` at time ``t0``. For a module at ``p`` write ``s_perp = (p - a) . d`` (the position of the foot
    of the perpendicular along the track) and ``rho = |(p - a) - s_perp d|`` (the distance of the module from the
    track). Cherenkov light leaves the track at the angle ``theta_c`` with ``cos(theta_c) = 1/n``; the ray that
    reaches the module starts at ``s_e = s_perp - rho cot(theta_c)`` at time ``t0 + s_e / c`` and travels the
    distance ``rho / sin(theta_c)`` at the speed ``c / n``::

        t = t0 + [s_perp - rho cos(theta_c) / sin(theta_c) + n rho / sin(theta_c)] / c
          = t0 + [s_perp + rho (n - 1/n) / sin(theta_c)] / c
          = t0 + [s_perp + rho sqrt(n^2 - 1)] / c,                using sin(theta_c) = sqrt(n^2 - 1) / n.

    In two dimensions only the two rays at ``+theta_c`` and ``-theta_c`` exist; each side of the track sees one. The
    formula is the same in three dimensions (the cone reaches every module). By Fermat's principle this is the
    earliest time any track light can arrive. If light travels at ``c / n_group`` with a group index different from
    the (phase) index ``n`` of the Cherenkov angle, the path term becomes ``n_group rho / sin(theta_c)`` and::

        t = t0 + [s_perp + rho (n_group n - 1) / sqrt(n^2 - 1)] / c.

    Cascade (``event_type`` 2): light from the point ``a`` (the vertex) at ``t0``: ``t = t0 + n_group |p - a| / c``.

    Starting track (``event_type`` 1): the muon starts at the vertex ``a``; the track formula applies where the
    emission point lies on the muon (``s_e >= 0``), the cascade formula elsewhere.

    Parameters
    ----------
    module_xz
        Module positions, shape ``[..., 2]`` [m].
    anchor
        Anchor point of the track (or vertex), shape ``[..., 2]`` broadcast against ``module_xz`` [m].
    direction
        Direction of travel, shape ``[..., 2]`` (normalised here; ignored for cascades).
    t0
        Time at the anchor [ns].
    n
        (Phase) refractive index, ``> 1``: the Cherenkov angle.
    n_group
        Group refractive index for the speed of light ``c / n_group``; ``None`` = ``n``.
    event_type
        Type code per element (0 track, 1 starting, 2 cascade, 3 atmospheric) or ``None`` for the track formula.
    radius
        Module radius [m]: with ``radius > 0`` the time at which the light first reaches the disc of the module, i.e.
        ``n_group radius / c`` earlier (the plane front of a track and the circular front of a cascade both touch the
        disc one radius before its centre).
    c
        Speed of light in vacuum [m/ns].

    Returns
    -------
    numpy.ndarray
        Arrival times [ns].

    Examples
    --------
    A module 10 m beside a track along +x in water (n = 1.33): ``(0 + 10 sqrt(1.33^2 - 1)) / c``.

    >>> t = expected_arrival_time([0.0, 10.0], [0.0, 0.0], [1.0, 0.0], 0.0, 1.33)
    >>> round(float(t), 3)
    29.249
    """
    p = np.asarray(module_xz, dtype=np.float64)
    a = np.asarray(anchor, dtype=np.float64)
    n_arr = np.asarray(n, dtype=np.float64)
    if np.any(n_arr <= 1):
        raise ValueError("the refractive index must exceed 1 for Cherenkov light")
    n_g = n_arr if n_group is None else np.asarray(n_group, dtype=np.float64)
    if np.any(n_g < 1):
        raise ValueError("the group refractive index must be >= 1")
    t0_arr = np.asarray(t0, dtype=np.float64)
    r = np.asarray(radius, dtype=np.float64)
    s, rho = track_coordinates(p, a, direction)
    root = np.sqrt(n_arr**2 - 1)
    t_track = t0_arr + (s + np.abs(rho) * (n_g * n_arr - 1) / root) / c
    if event_type is None:
        return t_track - n_g * r / c
    kind = np.asarray(event_type)
    distance = np.hypot(p[..., 0] - a[..., 0], p[..., 1] - a[..., 1])
    t_point = t0_arr + n_g * distance / c
    s_e = s - np.abs(rho) / root
    on_muon = (kind == _TRACK) | (kind == _ATMOSPHERIC) | ((kind == _STARTING) & (s_e >= 0))
    return np.where(on_muon, t_track, t_point) - n_g * r / c


def time_residuals_arrays(
    offset: ArrayLike,
    x: ArrayLike,
    z: ArrayLike,
    t: ArrayLike,
    *,
    event_type: ArrayLike,
    x0: ArrayLike,
    z0: ArrayLike,
    dir_x: ArrayLike,
    dir_z: ArrayLike,
    t0: ArrayLike,
    n: ArrayLike,
    n_group: ArrayLike | None = None,
    radius: ArrayLike = 0.0,
    c: float = C_VACUUM,
) -> np.ndarray:
    """Time residuals ``t - t_direct`` of every row of a ragged hit or pulse array against the true events.

    Parameters
    ----------
    offset
        Event offsets, length ``n_events + 1``.
    x, z, t
        Positions [m] and times [ns] of the rows.
    event_type, x0, z0, dir_x, dir_z, t0
        Truth per event (type code, anchor or vertex, direction of travel, time at the anchor).
    n, n_group, radius
        Refractive index, group index (``None`` = ``n``) and module radius per event (or scalars); see
        :func:`expected_arrival_time`.
    c
        Speed of light in vacuum [m/ns].

    Returns
    -------
    numpy.ndarray
        Residual per row [ns]: about 0 for direct light (spread by the time jitter, the Cherenkov angle spread and
        the module size), positive for scattered light, anything for noise.
    """
    offset = _check_offset(offset)
    e = event_index(offset)
    n_events = len(offset) - 1

    def per_event(values: ArrayLike) -> np.ndarray:
        arr = np.broadcast_to(np.asarray(values, dtype=np.float64), (n_events,))
        return arr[e]

    module_xz = np.stack([np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)], axis=-1)
    if module_xz.shape[0] != offset[-1]:
        raise ValueError(f"x and z must have length offset[-1] = {offset[-1]}")
    anchor = np.stack([per_event(x0), per_event(z0)], axis=-1)
    direction = np.stack([per_event(dir_x), per_event(dir_z)], axis=-1)
    kind = np.broadcast_to(np.asarray(event_type), (n_events,))[e]
    expected = expected_arrival_time(
        module_xz,
        anchor,
        direction,
        per_event(t0),
        per_event(n),
        n_group=None if n_group is None else per_event(n_group),
        event_type=kind,
        radius=per_event(radius),
        c=c,
    )
    return np.asarray(t, dtype=np.float64) - expected


def time_residuals(
    data: Any, level: Literal["pulses", "hits"] = "pulses", *, to_surface: bool = True, c: float = C_VACUUM
) -> np.ndarray:
    """Time residuals of every pulse (or hit) of a data set against the direct-light time of its true event.

    Parameters
    ----------
    data
        A :class:`~nutel.toy.data.SimData`.
    level
        ``"pulses"`` (residual of the pulse time, see ``readout.pulse_time``) or ``"hits"`` (every hit; needs the
        hit level).
    to_surface
        Measure the expected time to the surface of the module (``radius > 0`` in :func:`expected_arrival_time`),
        so that the earliest direct light has a residual of about 0; ``False`` uses the module centre.
    c
        Speed of light in vacuum [m/ns].

    Returns
    -------
    numpy.ndarray
        One residual per row of ``data.pulses`` (or ``data.hits``) [ns].
    """
    rows = getattr(data, level)
    if rows is None:
        raise ValueError(f"the data set has no {level} level")
    truth = data.truth
    gi = _geometry_index(data)
    detectors = list(data.detectors)
    n_of_geometry = np.array([float(det.medium.refractive_index) for det in detectors])
    n_group_of_geometry = np.array([float(det.medium.group_index) for det in detectors])
    radius_of_geometry = np.array([float(det.geometry.radius) for det in detectors])
    return time_residuals_arrays(
        rows.offset,
        rows.x,
        rows.z,
        rows.t,
        event_type=truth.type,
        x0=truth.x0,
        z0=truth.z0,
        dir_x=truth.dir_x,
        dir_z=truth.dir_z,
        t0=truth.t0,
        n=n_of_geometry[gi],
        n_group=n_group_of_geometry[gi],
        radius=radius_of_geometry[gi] if to_surface else 0.0,
        c=c,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Angles


def angular_error(angle_reco: ArrayLike, angle_true: ArrayLike, *, degrees: bool = False) -> np.ndarray:
    """Angle between two directions given by their direction angles, in degrees in ``[0, 180]``.

    The difference is wrapped, so that 359 deg and 1 deg differ by 2 deg. NaN inputs give NaN.

    Parameters
    ----------
    angle_reco, angle_true
        Direction angles (``atan2(dir_z, dir_x)``), in radians (or degrees with ``degrees=True``).
    degrees
        The inputs are in degrees.

    Returns
    -------
    numpy.ndarray
        Angular error [deg].

    Examples
    --------
    >>> angular_error(np.radians([359.0, 90.0]), np.radians([1.0, -90.0]))
    array([  2., 180.])
    """
    a = np.asarray(angle_reco, dtype=np.float64)
    b = np.asarray(angle_true, dtype=np.float64)
    if degrees:
        a, b = np.radians(a), np.radians(b)
    delta = np.mod(a - b + np.pi, 2 * np.pi) - np.pi
    return np.degrees(np.abs(delta))


# ----------------------------------------------------------------------------------------------------------------------
# Causal cleaning


def causal_cleaning(
    hits: _Ragged,
    slack: float = 20.0,
    *,
    n: float = 1.0,
    max_distance: float = 75.0,
    min_neighbours: int = 2,
    c: float = C_VACUUM,
) -> np.ndarray:
    """Keep the hits (or pulses) that are causally connected to at least ``min_neighbours`` other hits of the event.

    Two hits ``i, j`` of the same event are connected if::

        |t_i - t_j| <= n |r_i - r_j| / c + slack     and     |r_i - r_j| <= max_distance

    Light signals of one event are close in space and time; noise hits are uniform over the readout window, so a
    noise hit rarely has a causal partner. With ``n = 1`` (the default) the condition uses the vacuum speed of light,
    the limit for the muon; ``n`` = the refractive index allows for light that travels between the two modules
    (looser). ``slack`` absorbs the time jitter, the module size and scattering delays; ``max_distance`` adds the
    locality condition. Causality alone removes little: two modules 300 m apart may differ by 1 us, so in a window
    with tens of hits almost every noise hit has a distant partner. The defaults (75 m, two partners) suit the
    default line layout (50 m x 20 m spacing): most noise hits go, almost all signal hits stay. Scale
    ``max_distance`` with the spacing of other layouts; ``max_distance=math.inf, min_neighbours=1`` is the pure
    causality condition.

    Parameters
    ----------
    hits
        Hits or pulses: an object with ``offset``, ``x``, ``z``, ``t`` (e.g. ``data.hits`` or ``data.pulses``).
    slack
        Time slack [ns].
    n
        Refractive index used in the light-travel term (1 = vacuum).
    max_distance
        Largest distance of a partner hit [m] (``math.inf``: no locality condition).
    min_neighbours
        Number of partners a hit needs to be kept.
    c
        Speed of light in vacuum [m/ns].

    Returns
    -------
    numpy.ndarray
        Boolean mask over the rows of ``hits``: True = keep.
    """
    if slack < 0:
        raise ValueError(f"slack must be >= 0, got {slack}")
    if min_neighbours < 1:
        raise ValueError(f"min_neighbours must be >= 1, got {min_neighbours}")
    offset = _check_offset(hits.offset)
    x, z, t = (np.asarray(v, dtype=np.float64) for v in (hits.x, hits.z, hits.t))
    keep = np.zeros(offset[-1], dtype=bool)
    counts = np.diff(offset)
    for i in np.flatnonzero(counts > min_neighbours):
        a, b = offset[i], offset[i + 1]
        dist = np.hypot(x[a:b, None] - x[None, a:b], z[a:b, None] - z[None, a:b])
        dt = np.abs(t[a:b, None] - t[None, a:b])
        connected = (dt <= n * dist / c + slack) & (dist <= max_distance)
        np.fill_diagonal(connected, False)
        keep[a:b] = connected.sum(axis=1) >= min_neighbours
    return keep


# ----------------------------------------------------------------------------------------------------------------------
# Helpers


def _normalise(direction: ArrayLike) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    if d.shape[-1:] != (2,):
        raise ValueError(f"a direction must have shape [..., 2], got {d.shape}")
    norm = np.hypot(d[..., 0], d[..., 1])[..., None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return d / norm


def _check_offset(offset: ArrayLike) -> np.ndarray:
    out = np.asarray(offset, dtype=np.int64)
    if out.ndim != 1 or len(out) < 1 or out[0] != 0 or np.any(np.diff(out) < 0):
        raise ValueError("offset must be a non-decreasing 1-d array starting at 0")
    return out


def _geometry_index(data: Any) -> np.ndarray:
    gi = getattr(data.truth, "geometry_index", None)
    if gi is None:
        return np.zeros(len(data.truth.event_id), dtype=np.int64)
    return np.asarray(gi, dtype=np.int64)
