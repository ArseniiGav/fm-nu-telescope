"""Readout of the optical modules: time jitter, readout window, noise hits and pulses.

Two data levels, as the ``merged_photons`` and ``pulses`` of NuBench::

    hits     every detected photon (time + Gaussian jitter, inside the readout window) and every noise hit
             (Poisson(noise_rate x window length) per module and event, times uniform in the window)
    pulses   per module, time-ordered greedy merging: a pulse starts at the earliest unassigned hit and takes every
             hit with t <= t_start + merge_window; its time is the first time, its charge the number of hits

Noise can also be added to stored data as an augmentation (:func:`add_noise`): train with noise, test without (or
the other way round), as in the NuBench protocol.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .data import Detector, Hits, PhotonCounts, Pulses, SimData

__all__ = ["add_noise", "apply_jitter_and_window", "merge_pulses", "pulses_from_hits", "sample_noise", "sort_hits"]


def apply_jitter_and_window(
    t: np.ndarray, time_jitter: float, window: tuple[float, float], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian jitter to the arrival times and flag the hits inside ``[start, end)`` of the window.

    Returns
    -------
    t : numpy.ndarray
        Hit times [ns] (float64).
    inside : numpy.ndarray
        True for hits inside the window (bool).
    """
    t = np.asarray(t, dtype=np.float64)
    if time_jitter > 0 and len(t):
        t = t + time_jitter * rng.standard_normal(len(t))
    inside = (t >= window[0]) & (t < window[1])
    return t, inside


def sample_noise(
    n_modules: ArrayLike,
    noise_rate: ArrayLike,
    window_start: ArrayLike,
    window_end: ArrayLike,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Noise hits of a set of events: ``Poisson(n_modules x rate x window length)`` per event, module uniform.

    This equals independent ``Poisson(rate x window length)`` counts per module. All arguments are per event.

    Returns
    -------
    event : numpy.ndarray
        Event of every noise hit (index into the inputs, int64, sorted).
    module : numpy.ndarray
        Row of the module in the event's geometry table (int64).
    t : numpy.ndarray
        Times uniform in the window [ns] (float64).
    """
    n_modules = np.asarray(n_modules, dtype=np.int64)
    start = np.asarray(window_start, dtype=np.float64)
    length = np.asarray(window_end, dtype=np.float64) - start
    mean = n_modules * np.asarray(noise_rate, dtype=np.float64) * length * 1e-9
    counts = rng.poisson(mean) if np.any(mean > 0) else np.zeros(len(mean), dtype=np.int64)
    event = np.repeat(np.arange(len(mean), dtype=np.int64), counts)
    module = np.floor(rng.random(len(event)) * n_modules[event]).astype(np.int64)
    t = start[event] + rng.random(len(event)) * length[event]
    return event, module, t


def sort_hits(event: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Order of the hits by (event, t) (stable)."""
    return np.lexsort((t, event))


def merge_pulses(
    event: ArrayLike, key: ArrayLike, t: ArrayLike, is_noise: ArrayLike, tau: ArrayLike
) -> dict[str, np.ndarray]:
    """Greedy time-ordered merging of the hits of every (event, module) into pulses.

    Per (event, module): a pulse starts at the earliest unassigned hit and takes every hit with
    ``t <= t_start + tau``; the next pulse starts at the next hit after that.

    Parameters
    ----------
    event, key
        Event and module key of every hit (integers; any order).
    t
        Hit times [ns].
    is_noise
        True for noise hits.
    tau
        Merge window [ns], a scalar or one value per hit.

    Returns
    -------
    dict
        ``first_hit`` (index of the first hit of every pulse into the inputs), ``event``, ``t`` (time of the first
        hit, float64), ``t_mean`` (mean time of the hits, float64), ``charge`` and ``n_noise`` (int64) of every pulse,
        sorted by (event, t); ``pulse_of_hit`` (pulse of every input hit, int64).
    """
    event = np.asarray(event, dtype=np.int64)
    key = np.asarray(key, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64)
    noise = np.asarray(is_noise, dtype=bool)
    tau_all = np.broadcast_to(np.asarray(tau, dtype=np.float64), t.shape)
    n = len(t)
    order = np.lexsort((t, key, event))
    e, k, ts, tau_s = event[order], key[order], t[order], tau_all[order]
    new_group = np.ones(n, dtype=bool)
    if n > 1:
        new_group[1:] = (e[1:] != e[:-1]) | (k[1:] != k[:-1])
    group = np.cumsum(new_group) - 1
    pulse_sorted = np.empty(n, dtype=np.int64)
    firsts: list[np.ndarray] = []
    n_pulses = 0
    remaining = np.arange(n)
    while remaining.size:
        g = group[remaining]
        first = np.ones(remaining.size, dtype=bool)
        first[1:] = g[1:] != g[:-1]
        start = np.maximum.accumulate(np.where(first, np.arange(remaining.size), 0))
        tr = ts[remaining]
        take = tr <= tr[start] + tau_s[remaining]
        local = np.cumsum(first) - 1
        pulse_sorted[remaining[take]] = n_pulses + local[take]
        firsts.append(remaining[first])
        n_pulses += int(first.sum())
        remaining = remaining[~take]
    first_sorted = np.concatenate(firsts) if firsts else np.zeros(0, dtype=np.int64)
    charge = np.bincount(pulse_sorted, minlength=n_pulses)
    t_sum = np.bincount(pulse_sorted, weights=ts, minlength=n_pulses)
    n_noise = np.bincount(pulse_sorted, weights=noise[order].astype(np.float64), minlength=n_pulses).astype(np.int64)
    p_event, p_t = e[first_sorted], ts[first_sorted]
    p_order = np.lexsort((p_t, p_event))
    rank = np.empty(n_pulses, dtype=np.int64)
    rank[p_order] = np.arange(n_pulses)
    pulse_of_hit = np.empty(n, dtype=np.int64)
    pulse_of_hit[order] = rank[pulse_sorted]
    return {
        "first_hit": order[first_sorted][p_order],
        "event": p_event[p_order],
        "t": p_t[p_order],
        "t_mean": (t_sum / np.maximum(charge, 1))[p_order],
        "charge": charge[p_order],
        "n_noise": n_noise[p_order],
        "pulse_of_hit": pulse_of_hit,
    }


def pulses_from_hits(hits: Hits, tau: ArrayLike, mean_time: ArrayLike = False) -> Pulses:
    """Merge the hits of a hit table into pulses.

    ``tau`` is the merge window, ``mean_time`` selects the mean hit time (else the first) as the pulse time; each a
    scalar or one value per event.
    """
    event = hits.event_index()
    tau_arr = np.asarray(tau, dtype=np.float64)
    tau_hit = tau_arr[event] if tau_arr.ndim else tau_arr
    key = hits.string.astype(np.int64) * (1 << 32) + hits.module
    merged = merge_pulses(event, key, hits.t, hits.is_noise, tau_hit)
    mean_arr = np.asarray(mean_time, dtype=bool)
    p_time = np.where(np.broadcast_to(mean_arr, (hits.n_events,))[merged["event"]], merged["t_mean"], merged["t"])
    order = np.lexsort((p_time, merged["event"]))  # (event, t) also for mean times
    first = merged["first_hit"][order]
    offset = np.zeros(hits.n_events + 1, dtype=np.int64)
    np.cumsum(np.bincount(merged["event"], minlength=hits.n_events), out=offset[1:])
    return Pulses.from_arrays(
        {
            "offset": offset,
            "string": hits.string[first],
            "module": hits.module[first],
            "x": hits.x[first],
            "z": hits.z[first],
            "t": p_time[order],
            "charge": merged["charge"][order],
            "n_noise": merged["n_noise"][order],
        }
    )


def _resolved_rate(detector: Detector) -> float:
    """Noise rate of a detector [Hz] (``None`` -> the background rate of its medium)."""
    rate = detector.readout.noise_rate
    return detector.medium.background_rate if rate is None else rate


def add_noise(data: SimData, rate: float, rng: np.random.Generator | int | None = None) -> SimData:
    """Add noise hits to a data set (augmentation) and rebuild its pulses.

    Every module of every event gets ``Poisson(rate x window length)`` more noise hits with times uniform in the
    readout window of its detector; the pulses are re-merged with the detector's merge window and pulse time.

    Parameters
    ----------
    data
        Data set with hits (level ``"hits"`` or ``"both"``).
    rate
        Noise rate per module [Hz].
    rng
        NumPy generator or integer seed.

    Returns
    -------
    SimData
        A new data set: hits, pulses and the counts ``n_noise``, ``n_hits``, ``n_pulses`` updated; every detector's
        ``readout.noise_rate`` increased by ``rate``; ``provenance["augmentation"]`` records the call.
    """
    if data.hits is None:
        raise ValueError("add_noise needs the hits: generate or load the data with level 'hits' or 'both'")
    if rate < 0:
        raise ValueError(f"rate must be >= 0, got {rate}")
    seed_info: Any = rng if isinstance(rng, int) else None
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    gi = data.truth.geometry_index.astype(np.int64)
    dets = data.detectors
    n_modules = np.array([d.geometry.n_modules for d in dets], dtype=np.int64)[gi]
    w0 = np.array([d.readout.window[0] for d in dets])[gi]
    w1 = np.array([d.readout.window[1] for d in dets])[gi]
    tau = np.array([d.readout.merge_window for d in dets])[gi]
    mean_time = np.array([d.readout.pulse_time == "mean" for d in dets])[gi]
    event, row, t = sample_noise(n_modules, np.full(len(gi), float(rate)), w0, w1, rng)
    h = data.hits
    string = np.empty(len(event), np.int32)
    module = np.empty(len(event), np.int32)
    x = np.empty(len(event), np.float32)
    z = np.empty(len(event), np.float32)
    for k, det in enumerate(dets):
        sel = gi[event] == k
        g = det.geometry
        string[sel], module[sel], x[sel], z[sel] = g.string[row[sel]], g.module[row[sel]], g.x[row[sel]], g.z[row[sel]]
    old_event = h.event_index()
    all_event = np.concatenate([old_event, event])
    all_t = np.concatenate([h.t.astype(np.float64), t])
    order = sort_hits(all_event, all_t)
    n_new = len(event)
    hits = Hits.from_arrays(
        {
            "offset": np.concatenate([[0], np.cumsum(np.bincount(all_event, minlength=len(data)))]),
            "string": np.concatenate([h.string, string])[order],
            "module": np.concatenate([h.module, module])[order],
            "x": np.concatenate([h.x, x])[order],
            "z": np.concatenate([h.z, z])[order],
            "t": all_t[order],
            "source": np.concatenate([h.source, np.full(n_new, 3, np.int8)])[order],
            "n_scatter": np.concatenate([h.n_scatter, np.full(n_new, -1, np.int32)])[order],
            "path_length": np.concatenate([h.path_length, np.full(n_new, np.nan, np.float32)])[order],
            "is_noise": np.concatenate([h.is_noise, np.ones(n_new, bool)])[order],
        }
    )
    pulses = pulses_from_hits(hits, tau, mean_time)
    counts = data.photons.arrays()
    counts["n_noise"] = counts["n_noise"] + np.bincount(event, minlength=len(data))
    counts["n_hits"] = hits.counts()
    counts["n_pulses"] = pulses.counts()
    detectors = [
        dataclasses.replace(d, readout=dataclasses.replace(d.readout, noise_rate=_resolved_rate(d) + rate))
        for d in dets
    ]
    provenance = dict(data.provenance)
    provenance["augmentation"] = [
        *provenance.get("augmentation", []),
        {"add_noise_rate_hz": float(rate), "seed": seed_info},
    ]
    return dataclasses.replace(
        data,
        hits=hits,
        pulses=pulses,
        photons=PhotonCounts.from_arrays(counts),
        detectors=detectors,
        provenance=provenance,
    )
