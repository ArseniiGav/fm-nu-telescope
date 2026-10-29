"""Event simulation: sample the prior, emit and transport the photons, read out the modules.

:func:`generate` draws the detectors (one, or K from a :class:`~nutel.toy.family.DetectorFamily`) and the event
parameters from the prior, then calls the same machinery as :func:`simulate`, which works for any given events::

    1. per event (NumPy)    muon segment in the generation box, stochastic losses, photon numbers per source,
                            noise hits
    2. per chunk (torch)    photon emission and transport for the events of one geometry, in chunks of about
                            photons_per_chunk photons (events are never split)
    3. per chunk (NumPy)    time jitter, readout window, noise, hits sorted by (event, t), pulses

Random numbers come from independent streams derived from the seed with :class:`numpy.random.SeedSequence`: the
prior; the per-event quantities (step 1), drawn for all events before the transport and hence the same for every
device and chunking; and one :class:`torch.Generator` and one NumPy generator per chunk, seeded from
``(seed, chunk index)``, for what is drawn per photon. All outputs are identical for the same seed, device type,
chunking, PyTorch build and hardware.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import math
import platform
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike

from .config import C_VACUUM, PRESETS, SimConfig
from .data import Bursts, Detector, Hits, Paths, PhotonCounts, PointType, Pulses, SimData, Truth
from .events import (
    ATMOSPHERIC,
    CASCADE,
    SOURCE_BURST,
    SOURCE_CASCADE,
    SOURCE_NOISE,
    STARTING,
    TRACK,
    Sources,
    box_interval,
    emit_photons,
    generation_box,
    muon_segments,
    photon_numbers,
    sample_bursts,
    sample_prior,
    type_codes,
    zenith_azimuth,
)
from .family import FAMILIES, DetectorFamily, sample_detectors
from .readout import apply_jitter_and_window, merge_pulses, sample_noise
from .transport import ModuleGrid, transport

__all__ = [
    "DEFAULT_PHOTONS_PER_CHUNK",
    "FAMILY_READOUT",
    "LEVELS",
    "family_conflicts",
    "generate",
    "resolve_device",
    "simulate",
    "world_box",
]

log = logging.getLogger(__name__)

#: Default number of photons transported at once, per device type (about 150 bytes per photon).
DEFAULT_PHOTONS_PER_CHUNK = {"cpu": 2**20, "cuda": 2**22}

#: Data levels: pulses are always produced; ``"both"`` also keeps the hits (``"hits"`` is an alias of ``"both"``).
LEVELS = ("hits", "pulses", "both")

_N_FATES = len(PointType)


# ----------------------------------------------------------------------------------------------------------------------
# Public API


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Turn ``"auto"`` (CUDA if available, else CPU), ``"cpu"``, ``"cuda"`` or ``"cuda:N"`` into a torch device."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str) and not re.fullmatch(r"(cpu|cuda(:\d+)?)", device.strip()):
        raise ValueError(f"unsupported device {device!r}: use 'auto', 'cpu', 'cuda' or 'cuda:N'")
    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda"):
        raise ValueError(f"unsupported device {device!r}: use 'auto', 'cpu', 'cuda' or 'cuda:N'")
    if dev.type == "cuda" and not torch.cuda.is_available():
        if torch.version.cuda is None:
            raise RuntimeError(
                f"device 'cuda' requested, but the installed PyTorch ({torch.__version__}) is a CPU-only build: "
                "install a CUDA build of PyTorch or use device='cpu'"
            )
        raise RuntimeError("device 'cuda' requested, but CUDA is not available (use device='cpu' or 'auto')")
    return dev


#: Readout parameters that a detector family samples per detector (the window, merge window and pulse time are kept).
FAMILY_READOUT = ("efficiency", "time_jitter", "noise_rate")


def family_conflicts(config: SimConfig) -> list[str]:
    """Return the settings of ``config`` that a detector family would replace (:func:`generate` refuses them).

    A family samples the whole layout and medium and the readout parameters :data:`FAMILY_READOUT`. A configuration
    may combine with a family if its layout and medium are the defaults (water) and its readout values are those of
    the ladder preset with the same physics switches.
    """
    default = SimConfig()
    out = ["custom_geometry"] if config.custom_geometry is not None else []
    for part in ("layout", "medium"):
        mine, base = getattr(config, part), getattr(default, part)
        out += [f.name for f in dataclasses.fields(mine) if getattr(mine, f.name) != getattr(base, f.name)]
    # compare with the preset of the same physics switches (else with every preset, the full one first)
    candidates = [c for c in PRESETS.values() if c.physics == config.physics] or list(reversed(PRESETS.values()))
    allowed = [tuple(getattr(c.readout, name) for name in FAMILY_READOUT) for c in candidates]
    values = tuple(getattr(config.readout, name) for name in FAMILY_READOUT)
    if values not in allowed:
        nearest = min(allowed, key=lambda a: sum(x != y for x, y in zip(a, values, strict=True)))
        out += [name for name, x, y in zip(FAMILY_READOUT, values, nearest, strict=True) if x != y]
    return out


def world_box(detector: Detector, config: SimConfig) -> tuple[float, float, float, float]:
    """Return the world box: the instrumented area plus ``min(factor x lambda_a, max)``, at least the track margin."""
    medium = config.effective_medium(detector.medium)
    ph = config.physics
    margin = min(ph.world_margin_factor * medium.absorption_length, ph.world_margin_max)
    margin = max(margin, config.prior.track_margin)
    x0, x1, z0, z1 = detector.geometry.bounds
    return x0 - margin, x1 + margin, z0 - margin, z1 + margin


def generate(
    n_events: int | None = None,
    *,
    config: SimConfig | None = None,
    seed: int = 0,
    device: str | torch.device | None = "auto",
    level: str = "both",
    save_paths: int = 0,
    family: DetectorFamily | str | None = None,
    n_geometries: int = 1,
    geometry_seed: int = 0,
    events_per_geometry: int | None = None,
    detectors: Sequence[Detector] | Detector | None = None,
    photons_per_chunk: int | None = None,
    first_event_id: int = 0,
    progress: bool = False,
) -> SimData:
    """Simulate events with parameters drawn from the prior, in one detector or in K detectors of a family.

    Parameters
    ----------
    n_events
        Number of events (may be omitted with ``events_per_geometry``).
    config
        Configuration (default: the ``"full"`` preset in water); see :func:`~nutel.toy.preset`.
    seed
        Seed of all random numbers of the events (non-negative integer).
    device
        ``"auto"`` (CUDA if available, else CPU), ``"cpu"`` or ``"cuda"``.
    level
        ``"both"`` (default): keep hits and pulses (``"hits"`` is an alias); ``"pulses"``: pulses only.
    save_paths
        Record the paths of all photons of the first ``save_paths`` events.
    family
        Detector family (or the name of a preset, ``"default"`` or ``"tridentlike"``): sample ``n_geometries``
        detectors (geometry, medium, readout) from it. ``None``: the single detector of ``config``. With a family,
        ``config`` keeps the default layout and medium and a preset readout (:func:`family_conflicts`); its physics,
        emission, prior, readout window, merge window and pulse time apply to every detector.
    n_geometries
        Number of detectors K sampled from the family.
    geometry_seed
        Seed of the detector sampling (the same seed gives the same detectors for any event seed).
    events_per_geometry
        Simulate exactly this many events per detector (default: ``n_events`` split as evenly as possible).
    detectors
        Explicit detectors (instead of ``config`` or a family).
    photons_per_chunk
        Approximate number of photons transported at once (memory control); default from
        :data:`DEFAULT_PHOTONS_PER_CHUNK`. The output depends on it through the order of the random numbers.
    first_event_id
        Event number of the first event (use different ranges for shards that will be merged).
    progress
        Report the progress on the standard error stream.

    Returns
    -------
    SimData
        Truth, photon bookkeeping, pulses and, depending on the options, hits, bursts and photon paths. The events
        are ordered by geometry (``truth.geometry_index`` is non-decreasing).

    Examples
    --------
    >>> from nutel.toy import generate, preset
    >>> data = generate(1000, config=preset("full", medium="ice"), seed=1)  # doctest: +SKIP
    >>> multi = generate(n_geometries=8, events_per_geometry=500, family="default", seed=2)  # doctest: +SKIP
    """
    config = config or SimConfig()
    seed = _check_int(seed, 0, "seed")
    geometry_seed = _check_int(geometry_seed, 0, "geometry_seed")
    n_geometries = _check_int(n_geometries, 1, "n_geometries")
    family_info: dict[str, Any] | None = None
    if detectors is not None:
        dets = [detectors] if isinstance(detectors, Detector) else list(detectors)
        if not dets or not all(isinstance(d, Detector) for d in dets):
            raise TypeError("detectors must be a Detector or a non-empty sequence of Detector")
    elif family is not None:
        if isinstance(family, str) and family not in FAMILIES:
            raise ValueError(f"unknown detector family {family!r}; choose from {', '.join(FAMILIES)}")
        fam = FAMILIES[family] if isinstance(family, str) else family
        conflicts = family_conflicts(config)
        if conflicts:
            raise ValueError(
                "a detector family samples the layout, the medium and the readout efficiency, time jitter and noise "
                f"of every detector; these settings of the configuration would be ignored: {', '.join(conflicts)}"
            )
        dets = sample_detectors(n_geometries, fam, geometry_seed, config)
        family_info = {"name": family if isinstance(family, str) else None, "ranges": fam.to_dict()}
    else:
        if n_geometries != 1:
            raise ValueError("n_geometries > 1 needs a detector family (family=...) or explicit detectors")
        dets = [config.detector()]
    k = len(dets)
    if events_per_geometry is not None:
        per = _check_int(events_per_geometry, 1, "events_per_geometry")
        if n_events is not None and int(n_events) != per * k:
            raise ValueError(f"n_events = {n_events} differs from events_per_geometry x n_geometries = {per * k}")
        counts = np.full(k, per, dtype=np.int64)
    else:
        if n_events is None:
            raise ValueError("give n_events or events_per_geometry")
        n = _check_int(n_events, 1, "n_events")
        counts = np.full(k, n // k, dtype=np.int64)
        counts[: n % k] += 1
    prior_rng = _streams(seed)[0]
    parts = [sample_prior(int(c), config, det.geometry, prior_rng) for c, det in zip(counts, dets, strict=True)]
    events = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    events["geometry_index"] = np.repeat(np.arange(k), counts)
    extra = {
        "family": family_info,
        "n_geometries": k,
        "geometry_seed": geometry_seed if family_info is not None else None,
        "events_per_geometry": counts.tolist(),
    }
    return _simulate(
        events, dets, config, seed, device, level, save_paths, photons_per_chunk, first_event_id, progress, extra
    )


def simulate(
    events: Mapping[str, ArrayLike],
    *,
    config: SimConfig | None = None,
    detectors: Sequence[Detector] | Detector | None = None,
    seed: int = 0,
    device: str | torch.device | None = "auto",
    level: str = "both",
    save_paths: int = 0,
    photons_per_chunk: int | None = None,
    first_event_id: int = 0,
    progress: bool = False,
) -> SimData:
    """Simulate given events (the same machinery as :func:`generate` without the prior).

    Parameters
    ----------
    events
        Arrays (or scalars) by name: ``type`` (codes 0-3 or names), the direction as ``angle`` [rad] or as
        ``dir_x`` and ``dir_z``, ``x0``, ``z0`` [m] (vertex, or any point of the line for tracks: the anchor is its
        projection), ``t0`` [ns] (time at the anchor or vertex), ``energy`` [GeV]; ``cascade_fraction`` (required
        for starting tracks); optional ``geometry_index`` (default 0).
    config
        Configuration (default: the ``"full"`` preset in water); its prior is not used.
    detectors
        Detectors indexed by ``geometry_index`` (default: the detector of ``config``).
    seed, device, level, save_paths, photons_per_chunk, first_event_id, progress
        As for :func:`generate`. With the seed of a :func:`generate` call and its truth, the result is the same.

    Returns
    -------
    SimData
    """
    config = config or SimConfig()
    seed = _check_int(seed, 0, "seed")
    if detectors is None:
        dets = [config.detector()]
    else:
        dets = [detectors] if isinstance(detectors, Detector) else list(detectors)
    ev = _normalise_events(events, len(dets))
    extra = {
        "family": None,
        "n_geometries": len(dets),
        "geometry_seed": None,
        "events_per_geometry": np.bincount(ev["geometry_index"], minlength=len(dets)).tolist(),
    }
    return _simulate(
        ev, dets, config, seed, device, level, save_paths, photons_per_chunk, first_event_id, progress, extra
    )


# ----------------------------------------------------------------------------------------------------------------------
# Event preparation


def _normalise_events(events: Mapping[str, ArrayLike], n_detectors: int) -> dict[str, np.ndarray]:
    """Broadcast and check the event arrays of :func:`simulate`."""
    names = set(events)
    if "angle" in names:
        direction: dict[str, Any] = {"angle": events["angle"]}
    elif {"dir_x", "dir_z"} <= names:
        direction = {"dir_x": events["dir_x"], "dir_z": events["dir_z"]}
    else:
        raise ValueError("give the direction as 'angle' or as 'dir_x' and 'dir_z'")
    required = {"type", "x0", "z0", "t0", "energy"}
    missing = required - names
    if missing:
        raise ValueError(f"missing event arrays: {sorted(missing)}")
    kind = type_codes(events["type"])
    fields = {name: events[name] for name in ("x0", "z0", "t0", "energy")} | direction
    fields["cascade_fraction"] = events.get("cascade_fraction", np.nan)
    fields["geometry_index"] = events.get("geometry_index", 0)
    arrays = dict(zip(fields, np.broadcast_arrays(*(np.asarray(v) for v in fields.values()), kind), strict=False))
    kind = np.broadcast_to(kind, arrays["x0"].shape).ravel().astype(np.int8)
    out = {name: np.array(v, dtype=np.float64).ravel() for name, v in arrays.items() if name != "geometry_index"}
    out["geometry_index"] = np.array(arrays["geometry_index"]).ravel().astype(np.int64)
    out["type"] = kind
    if "angle" in out:
        out["dir_x"], out["dir_z"] = np.cos(out["angle"]), np.sin(out["angle"])
    norm = np.hypot(out["dir_x"], out["dir_z"])
    if np.any(~(norm > 0)):
        raise ValueError("directions must be non-zero vectors")
    out["dir_x"], out["dir_z"] = out["dir_x"] / norm, out["dir_z"] / norm
    out["angle"] = np.arctan2(out["dir_z"], out["dir_x"])
    n = _check_int(len(kind), 1, "number of events")
    for name in ("x0", "z0", "t0", "energy"):
        if not np.all(np.isfinite(out[name])):
            raise ValueError(f"{name} must be finite")
    if np.any(out["energy"] <= 0):
        raise ValueError("energies must be positive")
    starting = kind == STARTING
    y = out["cascade_fraction"]
    if np.any(starting & ~((y >= 0) & (y <= 1))):
        raise ValueError("starting tracks need cascade_fraction in [0, 1]")
    out["cascade_fraction"] = np.where(starting, y, np.nan)
    gi = out["geometry_index"]
    if np.any((gi < 0) | (gi >= n_detectors)):
        raise ValueError(f"geometry_index must be in [0, {n_detectors - 1}]")
    # the anchor of a line is the point closest to the origin
    line = (kind == TRACK) | (kind == ATMOSPHERIC)
    along = out["x0"] * out["dir_x"] + out["z0"] * out["dir_z"]
    out["x0"] = np.where(line, out["x0"] - along * out["dir_x"], out["x0"])
    out["z0"] = np.where(line, out["z0"] - along * out["dir_z"], out["z0"])
    assert len(out["x0"]) == n
    return out


@functools.cache
def _seed_of(seed: int, stream: int, chunk: int) -> int:
    state = np.random.SeedSequence(entropy=seed, spawn_key=(stream, chunk)).generate_state(1, dtype=np.uint64)[0]
    return int(state >> np.uint64(1))


def _streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Independent NumPy streams of one seed: the prior and the per-event quantities."""
    prior, events = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(prior), np.random.default_rng(events)


# ----------------------------------------------------------------------------------------------------------------------
# The simulation


def _simulate(
    ev: dict[str, np.ndarray],
    dets: list[Detector],
    config: SimConfig,
    seed: int,
    device: str | torch.device | None,
    level: str,
    save_paths: int,
    photons_per_chunk: int | None,
    first_event_id: int,
    progress: bool,
    extra: dict[str, Any],
) -> SimData:
    start_time = time.perf_counter()
    if level not in LEVELS:
        raise ValueError(f"level must be one of {', '.join(LEVELS)}, got {level!r}")
    save_paths = _check_int(save_paths, 0, "save_paths")
    first_event_id = _check_int(first_event_id, 0, "first_event_id")
    dev = resolve_device(device)
    budget = (
        DEFAULT_PHOTONS_PER_CHUNK[dev.type]
        if photons_per_chunk is None
        else _check_int(photons_per_chunk, 1, "photons_per_chunk")
    )
    n = len(ev["type"])
    em, ph = config.emission, config.physics
    dets = [_resolved(det, config) for det in dets]
    kind, gi = ev["type"], ev["geometry_index"]
    dx, dz, x0, z0, t0, energy = ev["dir_x"], ev["dir_z"], ev["x0"], ev["z0"], ev["t0"], ev["energy"]
    y = np.nan_to_num(ev["cascade_fraction"], nan=0.0)
    energy_track = np.select(
        [(kind == TRACK) | (kind == ATMOSPHERIC), kind == STARTING], [energy, (1 - y) * energy], 0.0
    )
    energy_cascade = np.select([kind == STARTING, kind == CASCADE], [y * energy, energy], 0.0)

    # 1. per-event quantities, drawn for all events before the transport (geometry by geometry)
    event_rng = _streams(seed)[1]
    s_start, s_end = np.zeros(n), np.zeros(n)
    length_in_area = np.zeros(n)
    stopped = np.zeros(n, dtype=bool)
    burst_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    groups = [np.flatnonzero(gi == k) for k in range(len(dets))]
    for k, idx in enumerate(groups):
        if len(idx) == 0:
            continue
        geom = dets[k].geometry
        a, b = muon_segments(kind[idx], x0[idx], z0[idx], dx[idx], dz[idx], generation_box(geom, config.prior))
        bursts = sample_bursts(a, b, energy_track[idx], em, ph, event_rng)
        s_start[idx], s_end[idx] = a, bursts.s_end
        stopped[idx] = bursts.stopped
        lo, hi = box_interval(x0[idx], z0[idx], dx[idx], dz[idx], geom.bounds)
        length_in_area[idx] = np.clip(np.minimum(hi, bursts.s_end) - np.maximum(lo, a), 0.0, None)
        burst_parts.append((idx[bursts.event], bursts.s, bursts.energy))
    if burst_parts:
        b_event = np.concatenate([p[0] for p in burst_parts])
        b_s = np.concatenate([p[1] for p in burst_parts])
        b_energy = np.concatenate([p[2] for p in burst_parts])
    else:
        b_event, b_s, b_energy = np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
    b_order = np.lexsort((b_s, b_event))
    b_event, b_s, b_energy = b_event[b_order], b_s[b_order], b_energy[b_order]
    b_offset = np.concatenate([[0], np.cumsum(np.bincount(b_event, minlength=n))]).astype(np.int64)
    track_length = s_end - s_start
    n_track = photon_numbers(em.track_yield * track_length, em.photon_statistics, event_rng)
    n_burst_each = photon_numbers(em.cascade_yield * b_energy, em.photon_statistics, event_rng)
    light = np.where(kind == STARTING, em.hadronic_light_factor, 1.0)
    n_cascade = photon_numbers(em.cascade_yield * light * energy_cascade, em.photon_statistics, event_rng)
    n_burst = np.bincount(b_event, weights=n_burst_each, minlength=n).astype(np.int64)
    n_modules = np.array([d.geometry.n_modules for d in dets])[gi]
    noise_rate = np.array([d.readout.noise_rate for d in dets])[gi]
    w0 = np.array([d.readout.window[0] for d in dets])[gi]
    w1 = np.array([d.readout.window[1] for d in dets])[gi]
    noise_event, noise_row, noise_t = sample_noise(n_modules, noise_rate, w0, w1, event_rng)
    noise_offset = np.concatenate([[0], np.cumsum(np.bincount(noise_event, minlength=n))]).astype(np.int64)

    # 2. + 3. chunks of events of one geometry: emission, transport, readout
    n_photons = n_track + n_burst + n_cascade
    keep_hits = level in ("hits", "both")
    fate_counts = np.zeros((n, _N_FATES), dtype=np.int64)
    n_direct = np.zeros(n, dtype=np.int64)
    n_out = np.zeros(n, dtype=np.int64)
    hit_parts: list[dict[str, np.ndarray]] = []
    pulse_parts: list[dict[str, np.ndarray]] = []
    path_parts: list[dict[str, np.ndarray]] = []
    chunk_index = 0
    bar = _Progress(n, enabled=progress)
    done = 0
    log.info(
        "simulating %d events (%.3g photons) in %d geometr%s on %s",
        n,
        n_photons.sum(),
        len(dets),
        "y" if len(dets) == 1 else "ies",
        dev,
    )
    with torch.inference_mode():
        for k, idx in enumerate(groups):
            if len(idx) == 0:
                continue
            det = dets[k]
            medium = config.effective_medium(det.medium)
            geom = det.geometry
            modules = torch.as_tensor(geom.positions.astype(np.float64), device=dev)
            world = world_box(det, config)
            grid = ModuleGrid(modules, geom.radius, world)
            for c0, c1 in _plan_chunks(n_photons[idx], budget):
                ids = idx[c0:c1]
                generator = torch.Generator(device=dev)
                generator.manual_seed(_seed_of(seed, 1, chunk_index))
                readout_rng = np.random.default_rng(_seed_of(seed, 2, chunk_index))
                chunk_index += 1
                sources = _sources(
                    ids, x0, z0, dx, dz, t0, s_start, track_length, n_track, n_cascade, b_offset, b_s, n_burst_each
                )
                photons = emit_photons(sources, em, medium.refractive_index, generator)
                record = None
                if save_paths > 0 and ids[0] < save_paths:
                    record = torch.as_tensor(ids < save_paths, device=dev)[photons.event]
                result = transport(
                    photons,
                    modules,
                    geom.radius,
                    absorption_length=medium.absorption_length,
                    scattering_length=medium.scattering_length,
                    g=medium.g,
                    speed_of_light=medium.speed_of_light,
                    efficiency=det.readout.efficiency,
                    world=world,
                    max_steps=ph.max_steps,
                    generator=generator,
                    grid=grid,
                    record=record,
                )
                local = photons.event
                counts = torch.bincount(local * _N_FATES + result.fate.long(), minlength=len(ids) * _N_FATES)
                fate_counts[ids] = counts.view(len(ids), _N_FATES).cpu().numpy()
                det_i = (result.fate == int(PointType.DETECT)).nonzero().squeeze(1)
                d_event = local[det_i].cpu().numpy()
                d_row = result.module[det_i].cpu().numpy()
                d_time = (photons.t[det_i] + result.path_length[det_i] / medium.speed_of_light).cpu().numpy()
                d_source = photons.source[det_i].cpu().numpy()
                d_scatter = result.n_scatter[det_i].cpu().numpy()
                d_path = result.path_length[det_i].cpu().numpy()
                n_direct[ids] = np.bincount(d_event[d_scatter == 0], minlength=len(ids))
                hits, pulses, out_counts = _readout_chunk(
                    ids,
                    d_event,
                    d_row,
                    d_time,
                    d_source,
                    d_scatter,
                    d_path,
                    det,
                    readout_rng,
                    noise_offset,
                    noise_row,
                    noise_t,
                    keep_hits,
                )
                n_out[ids] = out_counts
                if keep_hits:
                    hit_parts.append(hits)
                pulse_parts.append(pulses)
                if result.paths is not None:
                    path_parts.append(_paths_chunk(result.paths, photons, ids, first_event_id))
                done += len(ids)
                bar.update(done)
    bar.close()

    # assemble
    event_id = first_event_id + np.arange(n, dtype=np.int64)
    zenith, azimuth = zenith_azimuth(dx, dz)
    energy_lost = np.bincount(b_event, weights=b_energy, minlength=n)
    truth = Truth.from_arrays(
        {
            "event_id": event_id,
            "geometry_index": gi,
            "type": kind,
            "dir_x": dx,
            "dir_z": dz,
            "zenith": zenith,
            "azimuth": azimuth,
            "angle": ev["angle"],
            "x0": x0,
            "z0": z0,
            "t0": t0,
            "energy": energy,
            "energy_track": energy_track,
            "energy_cascade": energy_cascade,
            "cascade_fraction": ev["cascade_fraction"],
            "track_length": track_length,
            "s_start": np.where(track_length > 0, s_start, np.nan),
            "s_end": np.where(track_length > 0, s_end, np.nan),
            "length_in_area": length_in_area,
            "n_bursts": np.diff(b_offset),
            "energy_lost": energy_lost,
            "stopped": stopped,
        }
    )
    n_noise = np.diff(noise_offset)
    n_detected = fate_counts[:, PointType.DETECT]
    pulses_t = _assemble(pulse_parts, n, Pulses)
    hits_t = _assemble(hit_parts, n, Hits) if keep_hits else None
    photons_t = PhotonCounts.from_arrays(
        {
            "n_emitted": n_photons,
            "n_emitted_track": n_track,
            "n_emitted_burst": n_burst,
            "n_emitted_cascade": n_cascade,
            "n_detected": n_detected,
            "n_detected_direct": n_direct,
            "n_detected_scattered": n_detected - n_direct,
            "n_absorbed_module": fate_counts[:, PointType.ABSORB_MODULE],
            "n_absorbed": fate_counts[:, PointType.ABSORB],
            "n_escaped": fate_counts[:, PointType.ESCAPE],
            "n_lost": fate_counts[:, PointType.LOST],
            "n_noise": n_noise,
            "n_out_of_window": n_out,
            "n_hits": n_detected - n_out + n_noise,
            "n_pulses": pulses_t.counts(),
        }
    )
    burst_t = Bursts.from_arrays(
        {
            "offset": b_offset,
            "x": x0[b_event] + b_s * dx[b_event],
            "z": z0[b_event] + b_s * dz[b_event],
            "t": t0[b_event] + b_s / C_VACUUM,
            "energy": b_energy,
        }
    )
    paths_t = _assemble_paths(path_parts) if save_paths > 0 else None
    data = SimData(
        truth=truth,
        photons=photons_t,
        pulses=pulses_t,
        detectors=list(dets),
        config=config,
        hits=hits_t,
        bursts=burst_t,
        paths=paths_t,
    )
    data.provenance = _provenance(
        seed=seed,
        device=dev,
        photons_per_chunk=budget,
        n_chunks=chunk_index,
        level=level,
        save_paths=save_paths,
        first_event_id=first_event_id,
        wall_time=time.perf_counter() - start_time,
        **extra,
    )
    wall = data.provenance["wall_time_s"]
    log.info("done in %.1f s (%.3g ms/event)", wall, 1e3 * wall / n)
    return data


def _resolved(det: Detector, config: SimConfig) -> Detector:
    """Detector with the noise rate resolved (``None`` -> the background rate of its medium)."""
    readout = config.effective_readout(det.readout, det.medium)
    return det if readout is det.readout else Detector(geometry=det.geometry, medium=det.medium, readout=readout)


def _plan_chunks(n_photons: np.ndarray, budget: int) -> list[tuple[int, int]]:
    """Consecutive event ranges ``(start, stop)`` with about ``budget`` photons each (events are never split).

    An event goes to chunk ``k`` if its first photon has a position in ``[k budget, (k + 1) budget)`` in the list of
    all photons, so a chunk holds at most ``budget`` photons plus those of one event.
    """
    if len(n_photons) == 0:
        return []
    first_photon = np.cumsum(n_photons) - n_photons
    chunk = first_photon // budget
    edges = np.concatenate([[0], np.flatnonzero(np.diff(chunk)) + 1, [len(n_photons)]])
    return [(int(a), int(b)) for a, b in itertools.pairwise(edges)]


def _sources(
    ids: np.ndarray,
    x0: np.ndarray,
    z0: np.ndarray,
    dx: np.ndarray,
    dz: np.ndarray,
    t0: np.ndarray,
    s_start: np.ndarray,
    track_length: np.ndarray,
    n_track: np.ndarray,
    n_cascade: np.ndarray,
    b_offset: np.ndarray,
    b_s: np.ndarray,
    n_burst_each: np.ndarray,
) -> Sources:
    """Light sources of the events ``ids`` (chunk-local event indices 0 .. len(ids) - 1)."""
    local = np.arange(len(ids))
    seg = n_track[ids] > 0
    e = ids[seg]
    # vertex cascades (at s = 0) and bursts (at s = b_s), ordered by event and along the track
    counts = b_offset[ids + 1] - b_offset[ids]
    b_rows = (
        np.concatenate([np.arange(b_offset[i], b_offset[i + 1]) for i in ids])
        if counts.sum()
        else np.zeros(0, np.int64)
    )
    b_local = np.repeat(local, counts)
    pt_local = np.concatenate([local, b_local])
    pt_s = np.concatenate([np.zeros(len(ids)), b_s[b_rows]])
    pt_n = np.concatenate([n_cascade[ids], n_burst_each[b_rows]])
    pt_source = np.concatenate([np.full(len(ids), SOURCE_CASCADE), np.full(len(b_rows), SOURCE_BURST)])
    order = np.lexsort((pt_s, pt_local))
    keep = order[pt_n[order] > 0]
    pt_local, pt_s, pt_n, pt_source = pt_local[keep], pt_s[keep], pt_n[keep], pt_source[keep]
    pe = ids[pt_local]
    return Sources(
        seg_event=local[seg],
        seg_x=x0[e] + s_start[e] * dx[e],
        seg_z=z0[e] + s_start[e] * dz[e],
        seg_dx=dx[e],
        seg_dz=dz[e],
        seg_length=track_length[e],
        seg_t=t0[e] + s_start[e] / C_VACUUM,
        seg_n=n_track[e],
        pt_event=pt_local,
        pt_x=x0[pe] + pt_s * dx[pe],
        pt_z=z0[pe] + pt_s * dz[pe],
        pt_dx=dx[pe],
        pt_dz=dz[pe],
        pt_t=t0[pe] + pt_s / C_VACUUM,
        pt_n=pt_n,
        pt_source=pt_source,
    )


def _readout_chunk(
    ids: np.ndarray,
    d_event: np.ndarray,
    d_row: np.ndarray,
    d_time: np.ndarray,
    d_source: np.ndarray,
    d_scatter: np.ndarray,
    d_path: np.ndarray,
    det: Detector,
    rng: np.random.Generator,
    noise_offset: np.ndarray,
    noise_row: np.ndarray,
    noise_t: np.ndarray,
    keep_hits: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Hits and pulses of one chunk (event indices global, sorted by (event, t)), and the out-of-window counts."""
    ro, geom = det.readout, det.geometry
    t, inside = apply_jitter_and_window(d_time, ro.time_jitter, ro.window, rng)
    out_counts = np.bincount(d_event[~inside], minlength=len(ids))
    # noise hits of these events (consecutive in the global noise arrays only per event: gather by offsets)
    n_rows = [np.arange(noise_offset[i], noise_offset[i + 1]) for i in ids]
    rows = np.concatenate(n_rows) if n_rows else np.zeros(0, np.int64)
    noise_local = np.repeat(np.arange(len(ids)), noise_offset[ids + 1] - noise_offset[ids])
    event = np.concatenate([d_event[inside], noise_local])
    row = np.concatenate([d_row[inside], noise_row[rows]]).astype(np.int64)
    time_ = np.concatenate([t[inside], noise_t[rows]])
    is_noise = np.concatenate([np.zeros(int(inside.sum()), bool), np.ones(len(rows), bool)])
    order = np.lexsort((time_, event))
    event, row, time_, is_noise = event[order], row[order], time_[order], is_noise[order]
    hits: dict[str, np.ndarray] = {}
    if keep_hits:
        n_sig = int(inside.sum())
        hits = {
            "event": ids[event],
            "string": geom.string[row],
            "module": geom.module[row],
            "x": geom.x[row],
            "z": geom.z[row],
            "t": time_,
            "source": np.concatenate([d_source[inside], np.full(len(rows), SOURCE_NOISE, np.int8)])[order],
            "n_scatter": np.concatenate([d_scatter[inside], np.full(len(rows), -1, np.int32)])[order],
            "path_length": np.concatenate([d_path[inside], np.full(len(rows), np.nan)])[order],
            "is_noise": is_noise,
        }
        assert len(hits["source"]) == n_sig + len(rows)
    merged = merge_pulses(event, row, time_, is_noise, ro.merge_window)
    p_time = merged["t_mean"] if ro.pulse_time == "mean" else merged["t"]
    p_order = np.lexsort((p_time, merged["event"]))  # (event, t) also for mean times
    first_row = row[merged["first_hit"]][p_order]
    pulses = {
        "event": ids[merged["event"][p_order]],
        "string": geom.string[first_row],
        "module": geom.module[first_row],
        "x": geom.x[first_row],
        "z": geom.z[first_row],
        "t": p_time[p_order],
        "charge": merged["charge"][p_order],
        "n_noise": merged["n_noise"][p_order],
    }
    return hits, pulses, out_counts


def _assemble(parts: list[dict[str, np.ndarray]], n: int, cls: Any) -> Any:
    """Concatenate per-chunk tables (each sorted by (event, t)) into one ragged table sorted by (event, t)."""
    names = [name for name in cls.DTYPES if name != "offset"]
    if parts:
        event = np.concatenate([p["event"] for p in parts])
        order = np.argsort(event, kind="stable")
        columns = {name: np.concatenate([p[name] for p in parts])[order] for name in names}
        event = event[order]
    else:
        event = np.zeros(0, np.int64)
        columns = {name: np.zeros(0, dtype=cls.DTYPES[name]) for name in names}
    offset = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(event, minlength=n), out=offset[1:])
    return cls.from_arrays({"offset": offset, **columns})


def _paths_chunk(points: Any, photons: Any, ids: np.ndarray, first_event_id: int) -> dict[str, np.ndarray]:
    """Collect the path points of one chunk with the event id and the number within the event of their photon."""
    photon = points.photon
    event_local = photons.event[photon]
    first_of_event = torch.searchsorted(photons.event, event_local)  # photons are ordered by event
    rank = photon - first_of_event
    out = {
        "photon_key": photon.cpu().numpy(),
        "event_id": first_event_id + ids[event_local.cpu().numpy()],
        "rank": rank.cpu().numpy(),
        "source": photons.source[photon].cpu().numpy(),
        "x": points.x.cpu().numpy(),
        "z": points.z.cpu().numpy(),
        "t": points.t.cpu().numpy(),
        "kind": points.kind.cpu().numpy(),
    }
    return out


def _assemble_paths(parts: list[dict[str, np.ndarray]]) -> Paths:
    """Flatten the recorded points of all chunks into :class:`Paths` (ordered by event id and photon number)."""
    if not parts:
        return Paths.from_arrays(
            {"event_id": [], "photon": [], "source": [], "offset": [0], "x": [], "z": [], "t": [], "kind": []}
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.lexsort((np.arange(len(cat["rank"])), cat["rank"], cat["event_id"]))
    cat = {k: v[order] for k, v in cat.items()}
    key = cat["event_id"].astype(np.int64) * (1 << 31) + cat["rank"]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    return Paths.from_arrays(
        {
            "event_id": cat["event_id"][starts],
            "photon": cat["rank"][starts],
            "source": cat["source"][starts],
            "offset": np.r_[starts, len(key)],
            "x": cat["x"],
            "z": cat["z"],
            "t": cat["t"],
            "kind": cat["kind"],
        }
    )


# ----------------------------------------------------------------------------------------------------------------------
# Checks, progress and provenance


def _check_int(value: Any, minimum: int, name: str) -> int:
    """Return ``value`` as a plain int (NumPy integers included), at least ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


class _Progress:
    """Minimal progress report on stderr: a bar on a terminal, a log line every ~10 % otherwise."""

    def __init__(self, total: int, enabled: bool) -> None:
        self.total, self.enabled = total, enabled
        self.start = time.perf_counter()
        self.tty = enabled and sys.stderr.isatty()
        self.next_report = 0.1

    def update(self, done: int) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.start
        rate = done / max(elapsed, 1e-9)
        frac = done / self.total
        text = f"{done}/{self.total} events, {rate:.3g} events/s, {elapsed:.0f} s"
        if self.tty:
            bar = "#" * int(30 * frac)
            sys.stderr.write(f"\r[{bar:<30}] {100 * frac:3.0f} %  {text}")
            sys.stderr.flush()
        elif frac >= self.next_report or done == self.total:
            log.info("progress: %3.0f %% (%s)", 100 * frac, text)
            self.next_report = math.floor(frac * 10 + 1) / 10

    def close(self) -> None:
        if self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()


@functools.lru_cache(maxsize=1)
def _git_commit() -> str | None:
    """Commit of the source checkout this package runs from (with ``-dirty`` for local changes), else None."""
    package_dir = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=package_dir, capture_output=True, text=True, timeout=10, check=True)
        return done.stdout.strip()

    try:
        top = Path(git("rev-parse", "--show-toplevel")).resolve()
        if (top / "src" / "nutel").resolve() != package_dir:
            return None
        commit = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return None
    return commit + ("-dirty" if dirty else "")


def _provenance(*, device: torch.device, wall_time: float, **settings: Any) -> dict[str, Any]:
    """Seed, device, chunking, versions, creation time, wall time and command line of a run."""
    import h5py

    from .. import __version__

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or platform.machine()
    return {
        **settings,
        "device": str(device),
        "device_name": device_name,
        "torch_threads": torch.get_num_threads(),
        "package_version": __version__,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "h5py_version": h5py.__version__,
        "platform": platform.platform(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": round(wall_time, 3),
        "command_line": " ".join(sys.argv),
    }
