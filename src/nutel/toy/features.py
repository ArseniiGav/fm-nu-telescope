"""Hand-made event features and the module-grid image, written against plain NumPy arrays.

* :func:`event_features` (data set) / :func:`features_from_arrays` (ragged arrays): one row per event with the
  numbers of hits, pulses, modules and strings, the total charge, the time span, the charge-weighted centre and its
  extent, and the LineFit direction and speed (the time-ordered axis of the event).
* :func:`image` (data set) / :func:`image_from_arrays`: per-module quantities on the ``[string, module]`` grid, the
  first input of a convolutional network; ragged strings and several geometries are padded, with an optional mask
  channel that marks the modules that exist.

Both return NumPy arrays; ``pandas.DataFrame(event_features(data))`` gives a table.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from .reco import _check_offset, _geometry_index, event_index, linefit_events

__all__ = [
    "FEATURE_NAMES",
    "IMAGE_CHANNELS",
    "event_features",
    "features_from_arrays",
    "image",
    "image_from_arrays",
    "image_grid",
]

#: Columns of :func:`event_features` (after ``event_id`` and ``geometry_index``), with their units.
FEATURE_NAMES: dict[str, str] = {
    "n_hits": "",
    "n_pulses": "",
    "total_charge": "p.e.",
    "n_modules": "",
    "n_strings": "",
    "t_first": "ns",
    "t_last": "ns",
    "duration": "ns",
    "x_cog": "m",
    "z_cog": "m",
    "x_rms": "m",
    "z_rms": "m",
    "linefit_dir_x": "",
    "linefit_dir_z": "",
    "linefit_angle": "rad",
    "linefit_speed": "m/ns",
}

#: Channels :func:`image` can build: first pulse time, total charge, number of pulses, charge-weighted mean time and
#: last pulse time of each module.
IMAGE_CHANNELS: tuple[str, ...] = ("t_first", "charge", "n_pulses", "t_mean", "t_last")
_TIME_CHANNELS = {"t_first", "t_mean", "t_last"}


# ----------------------------------------------------------------------------------------------------------------------
# Event features


def features_from_arrays(
    offset: ArrayLike,
    x: ArrayLike,
    z: ArrayLike,
    t: ArrayLike,
    charge: ArrayLike | None = None,
    *,
    string: ArrayLike | None = None,
    n_hits: ArrayLike | None = None,
) -> dict[str, np.ndarray]:
    """Per-event features of a ragged pulse (or hit) array.

    Parameters
    ----------
    offset
        Event offsets, length ``n_events + 1``.
    x, z
        Module positions of the pulses [m].
    t
        Pulse times [ns].
    charge
        Pulse charges [p.e.]; ``None`` = 1 per row (hits).
    string
        String id of each row (to count the hit strings); ``None`` counts distinct ``x`` positions instead.
    n_hits
        Number of hits per event; ``None`` = the total charge (a pulse's charge is the number of hits it merges).

    Returns
    -------
    dict of str to numpy.ndarray
        The columns of :data:`FEATURE_NAMES`, one value per event. Times and positions are NaN for events without
        pulses; the LineFit columns are NaN for events with fewer than two pulses or no spread in time.
        ``x_rms`` and ``z_rms`` are the charge-weighted standard deviations of the positions around the centre.
    """
    offset = _check_offset(offset)
    n_events = len(offset) - 1
    x_arr, z_arr, t_arr = (np.asarray(v, dtype=np.float64) for v in (x, z, t))
    q = np.ones_like(t_arr) if charge is None else np.asarray(charge, dtype=np.float64)
    for name, values in (("x", x_arr), ("z", z_arr), ("t", t_arr), ("charge", q)):
        if values.shape != (offset[-1],):
            raise ValueError(f"{name} must have length offset[-1] = {offset[-1]}, got shape {values.shape}")
    e = event_index(offset)
    counts = np.diff(offset)
    total = np.bincount(e, q, minlength=n_events)
    out: dict[str, np.ndarray] = {}
    out["n_hits"] = np.round(total).astype(np.int64) if n_hits is None else np.asarray(n_hits, dtype=np.int64)
    out["n_pulses"] = counts.astype(np.int64)
    out["total_charge"] = total
    out["n_modules"] = _count_distinct(e, x_arr, z_arr, n_events)
    out["n_strings"] = _count_distinct(e, np.asarray(string if string is not None else x_arr), None, n_events)
    t_first = np.full(n_events, np.inf)
    t_last = np.full(n_events, -np.inf)
    np.minimum.at(t_first, e, t_arr)
    np.maximum.at(t_last, e, t_arr)
    empty = counts == 0
    t_first[empty] = np.nan
    t_last[empty] = np.nan
    out["t_first"], out["t_last"], out["duration"] = t_first, t_last, t_last - t_first
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, pos in (("x", x_arr), ("z", z_arr)):
            mean = np.bincount(e, q * pos, minlength=n_events) / total
            var = np.bincount(e, q * (pos - mean[e]) ** 2, minlength=n_events) / total
            out[f"{name}_cog"] = mean
            out[f"{name}_rms"] = np.sqrt(np.maximum(var, 0.0))
    fit = linefit_events(offset, x_arr, z_arr, t_arr, q)
    out["linefit_dir_x"] = fit.direction[:, 0]
    out["linefit_dir_z"] = fit.direction[:, 1]
    out["linefit_angle"] = fit.angle
    out["linefit_speed"] = fit.speed
    return out


def event_features(data: Any, level: Literal["pulses", "hits"] = "pulses") -> dict[str, np.ndarray]:
    """Per-event features of a data set (see :func:`features_from_arrays` for the definitions).

    Parameters
    ----------
    data
        A :class:`~nutel.toy.data.SimData`.
    level
        ``"pulses"`` (default; weights = pulse charges) or ``"hits"`` (every hit with weight 1; needs the hit level).

    Returns
    -------
    dict of str to numpy.ndarray
        ``event_id``, ``geometry_index`` and the columns of :data:`FEATURE_NAMES`, one value per event.
        ``n_hits`` is the number of stored hits of the event (signal and noise inside the readout window).

    Examples
    --------
    >>> import pandas as pd  # doctest: +SKIP
    >>> table = pd.DataFrame(event_features(data))  # doctest: +SKIP
    """
    rows = getattr(data, level, None)
    if rows is None:
        raise ValueError(f"the data set has no {level} level")
    n_hits = _n_hits(data)
    charge = getattr(rows, "charge", None) if level == "pulses" else None
    table = features_from_arrays(rows.offset, rows.x, rows.z, rows.t, charge, string=rows.string, n_hits=n_hits)
    head = {
        "event_id": np.asarray(data.truth.event_id, dtype=np.int64),
        "geometry_index": _geometry_index(data),
    }
    return head | table


# ----------------------------------------------------------------------------------------------------------------------
# Image on the module grid


def image_from_arrays(
    offset: ArrayLike,
    string_index: ArrayLike,
    module_index: ArrayLike,
    t: ArrayLike,
    charge: ArrayLike | None,
    shape: tuple[int, int],
    *,
    channels: Sequence[str] = ("t_first", "charge", "n_pulses"),
    module_mask: ArrayLike | None = None,
    relative_time: bool = False,
    empty_time: float = np.nan,
) -> np.ndarray:
    """Per-module quantities of every event on a ``[string, module]`` grid.

    Parameters
    ----------
    offset
        Event offsets, length ``n_events + 1``.
    string_index, module_index
        Grid position of each pulse: the rank of its string (0 ... ``shape[0] - 1``) and its module number
        (0 = deepest, 0 ... ``shape[1] - 1``).
    t
        Pulse times [ns].
    charge
        Pulse charges [p.e.] (``None`` = 1 per row).
    shape
        ``(n_strings, n_modules)`` of the grid.
    channels
        Names from :data:`IMAGE_CHANNELS`, in the order of the output channels.
    module_mask
        Boolean ``[n_events, n_strings, n_modules]`` (or broadcastable): True where a module exists. Appended as a
        last channel (1.0 / 0.0) if given; padded cells of missing modules then differ from empty modules.
    relative_time
        Subtract the first pulse time of the event from the time channels.
    empty_time
        Value of the time channels in modules without pulses (and in padding).

    Returns
    -------
    numpy.ndarray
        Float32 array ``[n_events, n_channels (+1), n_strings, n_modules]``; charge and pulse counts are 0 in empty
        modules.
    """
    offset = _check_offset(offset)
    n_events = len(offset) - 1
    n_str, n_mod = (int(v) for v in shape)
    s_idx = np.asarray(string_index, dtype=np.int64)
    m_idx = np.asarray(module_index, dtype=np.int64)
    t_arr = np.asarray(t, dtype=np.float64)
    q = np.ones_like(t_arr) if charge is None else np.asarray(charge, dtype=np.float64)
    unknown = [c for c in channels if c not in IMAGE_CHANNELS]
    if unknown:
        raise ValueError(f"unknown image channels {unknown}; choose from {', '.join(IMAGE_CHANNELS)}")
    if len(s_idx) and (s_idx.min() < 0 or s_idx.max() >= n_str or m_idx.min() < 0 or m_idx.max() >= n_mod):
        raise ValueError(f"string or module index outside the grid of shape {shape}")
    e = event_index(offset)
    cells = n_str * n_mod
    flat = e * cells + s_idx * n_mod + m_idx
    size = n_events * cells
    charge_sum = np.bincount(flat, q, minlength=size)
    n_pulses = np.bincount(flat, minlength=size)
    hit = n_pulses > 0
    reference = np.zeros(n_events)
    if relative_time and len(t_arr):
        first = np.full(n_events, np.inf)
        np.minimum.at(first, e, t_arr)
        reference = np.where(np.isfinite(first), first, 0.0)
    out = np.empty((n_events, len(channels) + (module_mask is not None), n_str, n_mod), dtype=np.float32)
    for k, name in enumerate(channels):
        values: np.ndarray
        if name == "charge":
            values = charge_sum
        elif name == "n_pulses":
            values = n_pulses.astype(np.float64)
        else:
            values = np.full(size, np.inf if name == "t_first" else -np.inf)
            if name == "t_first":
                np.minimum.at(values, flat, t_arr)
            elif name == "t_last":
                np.maximum.at(values, flat, t_arr)
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    values = np.bincount(flat, q * t_arr, minlength=size) / charge_sum
            values = values - np.repeat(reference, cells)
            values[~hit] = empty_time
        out[:, k] = values.reshape(n_events, n_str, n_mod)
    if module_mask is not None:
        mask = np.broadcast_to(np.asarray(module_mask, dtype=bool), (n_events, n_str, n_mod))
        out[:, -1] = mask
    return out


def image_grid(geometries: Sequence[Any]) -> tuple[tuple[int, int], list[np.ndarray], np.ndarray]:
    """Build the common ``[string, module]`` grid of one or several geometries.

    Strings are ranked by their id within each geometry (rank 0 = smallest id); modules keep their number
    (0 = deepest). The grid is large enough for the largest geometry; smaller or ragged ones are padded.

    Parameters
    ----------
    geometries
        Objects with integer arrays ``string`` and ``module`` (e.g. :class:`~nutel.toy.geometry.Geometry`).

    Returns
    -------
    shape : tuple of int
        ``(n_strings, n_modules)`` of the grid.
    string_ids : list of numpy.ndarray
        Sorted string ids of each geometry (the rank of a string is its position here).
    mask : numpy.ndarray
        Boolean ``[n_geometries, n_strings, n_modules]``, True where the geometry has a module.
    """
    if not geometries:
        raise ValueError("no geometry given")
    string_ids = [np.unique(np.asarray(g.string, dtype=np.int64)) for g in geometries]
    modules = [np.asarray(g.module, dtype=np.int64) for g in geometries]
    if any(len(m) and m.min() < 0 for m in modules):
        raise ValueError("module numbers must be non-negative")
    shape = (max(len(s) for s in string_ids), max(int(m.max()) + 1 if len(m) else 0 for m in modules))
    mask = np.zeros((len(geometries), *shape), dtype=bool)
    for k, g in enumerate(geometries):
        rank = np.searchsorted(string_ids[k], np.asarray(g.string, dtype=np.int64))
        mask[k, rank, modules[k]] = True
    return shape, string_ids, mask


def image(
    data: Any,
    channels: Sequence[str] = ("t_first", "charge", "n_pulses"),
    *,
    mask: bool = False,
    relative_time: bool = False,
    empty_time: float = np.nan,
) -> np.ndarray:
    """Image of every event on the ``[string, module]`` grid of its detector.

    Parameters
    ----------
    data
        A :class:`~nutel.toy.data.SimData` (one or several geometries).
    channels
        Names from :data:`IMAGE_CHANNELS`: ``t_first`` (first pulse time, ns), ``charge`` (p.e.), ``n_pulses``,
        ``t_mean`` (charge-weighted mean time, ns), ``t_last`` (ns).
    mask
        Append a channel that is 1 where the event's detector has a module and 0 in the padding (ragged strings,
        smaller geometries of a multi-geometry file).
    relative_time
        Measure times from the first pulse of the event.
    empty_time
        Value of the time channels in modules without pulses.

    Returns
    -------
    numpy.ndarray
        Float32 ``[n_events, n_channels (+1), n_strings_max, n_modules_max]``, indexed ``[event, channel, string
        rank, module]`` (see :func:`image_grid`).
    """
    pulses = data.pulses
    geometries = list(data.geometries)
    shape, string_ids, grid_mask = image_grid(geometries)
    gi = _geometry_index(data)
    offset = _check_offset(pulses.offset)
    row_geometry = gi[event_index(offset)]
    row_string = np.asarray(pulses.string, dtype=np.int64)
    string_index = np.zeros(len(row_string), dtype=np.int64)
    for k, ids in enumerate(string_ids):
        rows = row_geometry == k
        rank = np.searchsorted(ids, row_string[rows])
        if np.any(rank >= len(ids)) or np.any(ids[np.minimum(rank, len(ids) - 1)] != row_string[rows]):
            raise ValueError(f"pulses of geometry {k} refer to strings that the geometry does not have")
        string_index[rows] = rank
    return image_from_arrays(
        offset,
        string_index,
        pulses.module,
        pulses.t,
        pulses.charge,
        shape,
        channels=channels,
        module_mask=grid_mask[gi] if mask else None,
        relative_time=relative_time,
        empty_time=empty_time,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Helpers


def _count_distinct(e: np.ndarray, a: np.ndarray, b: np.ndarray | None, n_events: int) -> np.ndarray:
    """Count the distinct values (or pairs of values) of every event."""
    if len(e) == 0:
        return np.zeros(n_events, dtype=np.int64)
    keys = [np.asarray(a)] if b is None else [np.asarray(b), np.asarray(a)]
    order = np.lexsort((*keys, e))  # the last key is the primary one
    new = np.ones(len(e), dtype=bool)
    new[1:] = e[order][1:] != e[order][:-1]
    for key in keys:
        sorted_key = key[order]
        new[1:] |= sorted_key[1:] != sorted_key[:-1]
    return np.bincount(e[order][new], minlength=n_events).astype(np.int64)


def _n_hits(data: Any) -> np.ndarray | None:
    photons = getattr(data, "photons", None)
    if photons is not None and getattr(photons, "n_hits", None) is not None:
        return np.asarray(photons.n_hits, dtype=np.int64)
    hits = getattr(data, "hits", None)
    if hits is not None:
        return np.diff(np.asarray(hits.offset, dtype=np.int64))
    return None
