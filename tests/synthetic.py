"""Small synthetic data sets built from the containers directly (no photon transport), for the analysis-tool tests.

Events are straight tracks or cascades; every module within ``max_distance`` of the track (or vertex) gets a few hits
at the direct-light time plus a small delay, merged into pulses as the readout does. The numbers are not physical,
but the structure (offsets, sorting, truth, detectors, paths) is that of real data.
"""

from __future__ import annotations

import numpy as np

from nutel.toy.config import Readout, SimConfig, medium
from nutel.toy.data import (
    POINT_TYPES,
    Bursts,
    Detector,
    Hits,
    Paths,
    PhotonCounts,
    Pulses,
    SimData,
    Truth,
)
from nutel.toy.events import zenith_azimuth
from nutel.toy.geometry import Geometry, make_geometry
from nutel.toy.reco import expected_arrival_time, track_coordinates, unit_vector


def default_detectors() -> list[Detector]:
    return [Detector(geometry=make_geometry(), medium=medium("water"), readout=Readout(noise_rate=5000.0))]


def two_detectors() -> list[Detector]:
    """A line layout in water and a ragged one in ice."""
    return [
        Detector(geometry=make_geometry(), medium=medium("water"), readout=Readout(noise_rate=5000.0)),
        Detector(
            geometry=make_geometry(layout="ragged", n_strings=5, layout_seed=3),
            medium=medium("ice"),
            readout=Readout(noise_rate=0.0),
        ),
    ]


def synthetic_data(
    n_events: int = 8,
    *,
    seed: int = 0,
    detectors: list[Detector] | None = None,
    hits: bool = True,
    bursts: bool = True,
    paths: int = 2,
    first_event_id: int = 0,
    noise_events: bool = True,
) -> SimData:
    rng = np.random.default_rng(seed)
    detectors = detectors or default_detectors()
    gi = (np.arange(n_events) % len(detectors)).astype(np.int32)
    types = (np.arange(n_events) % 4).astype(np.int8)
    angle = rng.uniform(-np.pi, np.pi, n_events)
    d = unit_vector(angle)
    x0, z0 = rng.uniform(-60, 60, (2, n_events))
    t0 = rng.uniform(1000, 2000, n_events)
    energy = 10 ** rng.uniform(2, 3, n_events)
    hit_rows: dict[str, list[np.ndarray]] = {k: [] for k in ("string", "module", "x", "z", "t", "source", "n_scatter")}
    hit_offset = [0]
    for i in range(n_events):
        g = detectors[gi[i]].geometry
        n = detectors[gi[i]].medium.n
        positions = np.stack([g.x, g.z], axis=1).astype(np.float64)
        if types[i] == 2:
            distance = np.hypot(*(positions - [x0[i], z0[i]]).T)
        else:
            distance = np.abs(track_coordinates(positions, [x0[i], z0[i]], d[i])[1])
        near = np.flatnonzero(distance < 45.0)
        t_direct = expected_arrival_time(positions[near], [x0[i], z0[i]], d[i], t0[i], n, event_type=types[i])
        charge = rng.integers(1, 4, len(near))
        rows = np.repeat(np.arange(len(near)), charge)
        t = np.repeat(t_direct, charge) + rng.exponential(3.0, len(rows))
        scatter = (rng.random(len(rows)) < 0.3).astype(np.int32)
        source = np.full(len(rows), 2 if types[i] == 2 else 0, dtype=np.int8)
        string, module = g.string[near][rows], g.module[near][rows]
        x, z = g.x[near][rows], g.z[near][rows]
        if noise_events and i == n_events - 1:  # one noise hit far away in time
            string, module = np.append(string, g.string[0]), np.append(module, g.module[0])
            x, z = np.append(x, g.x[0]), np.append(z, g.z[0])
            t, source, scatter = np.append(t, 4500.0), np.append(source, 3), np.append(scatter, -1)
        order = np.argsort(t, kind="stable")
        for key, value in (
            ("string", string),
            ("module", module),
            ("x", x),
            ("z", z),
            ("t", t),
            ("source", source),
            ("n_scatter", scatter),
        ):
            hit_rows[key].append(np.asarray(value)[order])
        hit_offset.append(hit_offset[-1] + len(order))
    h = {k: np.concatenate(v) for k, v in hit_rows.items()}
    is_noise = h["source"] == 3
    path_length = np.where(is_noise, np.nan, 50.0)
    hits_container = _build(
        Hits, {"offset": np.array(hit_offset), "path_length": path_length, "is_noise": is_noise, **h}
    )
    pulses_container = _merge(hits_container, 10.0)
    zenith, azimuth = zenith_azimuth(d[:, 0], d[:, 1])
    track_like = types != 2
    truth = _build(
        Truth,
        {
            "event_id": np.arange(first_event_id, first_event_id + n_events),
            "geometry_index": gi,
            "type": types,
            "dir_x": d[:, 0],
            "dir_z": d[:, 1],
            "zenith": zenith,
            "azimuth": azimuth,
            "angle": angle,
            "x0": x0,
            "z0": z0,
            "t0": t0,
            "energy": energy,
            "energy_track": np.where(track_like, energy, 0.0),
            "energy_cascade": np.where(types == 2, energy, np.where(types == 1, 0.3 * energy, 0.0)),
            "cascade_fraction": np.where(types == 1, 0.3, np.nan),
            "track_length": np.where(track_like, 500.0, 0.0),
            "length_in_area": np.where(track_like, 300.0, 0.0),
            "n_bursts": np.where(track_like, 1, 0),
            "energy_lost": np.where(track_like, 5.0, 0.0),
        },
    )
    n_hits = np.diff(hits_container.offset).astype(np.int32)
    n_noise = np.bincount(hits_container.event_index(), is_noise, minlength=n_events).astype(np.int32)
    n_detected = n_hits - n_noise
    photons = _build(
        PhotonCounts,
        {
            "n_emitted": 10 * n_detected,
            "n_emitted_track": np.where(track_like, 10 * n_detected, 0),
            "n_emitted_burst": np.zeros(n_events),
            "n_emitted_cascade": np.where(track_like, 0, 10 * n_detected),
            "n_detected": n_detected,
            "n_detected_direct": n_detected,
            "n_detected_scattered": np.zeros(n_events),
            "n_absorbed_module": np.zeros(n_events),
            "n_absorbed": 9 * n_detected,
            "n_escaped": np.zeros(n_events),
            "n_lost": np.zeros(n_events),
            "n_noise": n_noise,
            "n_out_of_window": np.zeros(n_events),
            "n_hits": n_hits,
            "n_pulses": np.diff(pulses_container.offset),
        },
    )
    bursts_container = None
    if bursts:
        nb = np.where(track_like, 1, 0)
        bursts_container = _build(
            Bursts,
            {
                "offset": np.concatenate([[0], np.cumsum(nb)]),
                "x": x0[track_like],
                "z": z0[track_like],
                "t": t0[track_like],
                "energy": np.full(track_like.sum(), 5.0),
            },
        )
    paths_container = _paths(truth, paths, rng) if paths else None
    return SimData(
        truth=truth,
        photons=photons,
        pulses=pulses_container,
        detectors=detectors,
        config=SimConfig(),
        provenance={"seed": seed, "device": "cpu", "synthetic": True},
        hits=hits_container if hits else None,
        bursts=bursts_container,
        paths=paths_container,
    )


def _build(cls, arrays):
    """``cls.from_arrays`` with zeros for the fields this builder does not know (new fields of the containers)."""
    arrays = dict(arrays)
    per_point = {"x", "z", "t", "kind"}
    if "offset" in arrays and cls is not Paths:
        length = int(np.asarray(arrays["offset"])[-1])
    elif cls is Paths:
        length = len(arrays["event_id"])
    else:
        length = len(np.asarray(next(iter(arrays.values()))))
    for name, dtype in cls.DTYPES.items():
        if name not in arrays:
            size = int(np.asarray(arrays["offset"])[-1]) if cls is Paths and name in per_point else length
            arrays[name] = np.zeros(size, dtype=dtype)
    return cls.from_arrays(arrays)


def _merge(hits: Hits, tau: float) -> Pulses:
    """Greedy per-module merging of time-sorted hits (the readout rule)."""
    rows: dict[str, list[float]] = {k: [] for k in ("string", "module", "x", "z", "t", "charge", "n_noise")}
    offset = [0]
    for i in range(hits.n_events):
        a, b = hits.offset[i], hits.offset[i + 1]
        pulses = []
        keys = hits.string[a:b].astype(np.int64) * 1000 + hits.module[a:b]
        for key in np.unique(keys):
            sel = np.flatnonzero(keys == key) + a
            sel = sel[np.argsort(hits.t[sel], kind="stable")]
            k = 0
            while k < len(sel):
                start = hits.t[sel[k]]
                group = [sel[k]]
                k += 1
                while k < len(sel) and hits.t[sel[k]] <= start + tau:
                    group.append(sel[k])
                    k += 1
                j = group[0]
                pulses.append(
                    (
                        hits.t[j],
                        hits.string[j],
                        hits.module[j],
                        hits.x[j],
                        hits.z[j],
                        len(group),
                        hits.is_noise[group].sum(),
                    )
                )
        pulses.sort(key=lambda p: p[0])
        for p in pulses:
            for name, value in zip(("t", "string", "module", "x", "z", "charge", "n_noise"), p, strict=True):
                rows[name].append(value)
        offset.append(offset[-1] + len(pulses))
    return _build(Pulses, {"offset": np.array(offset), **{k: np.array(v) for k, v in rows.items()}})


def _paths(truth: Truth, n_events: int, rng: np.random.Generator) -> Paths:
    """Three photons per event for the first events: direct detected, scattered detected, absorbed."""
    code = POINT_TYPES
    kinds = [
        [code["EMIT"], code["DETECT"]],
        [code["EMIT"], code["SCATTER"], code["DETECT"]],
        [code["EMIT"], code["SCATTER"], code["ABSORB"]],
        [code["EMIT"], code["ESCAPE"]],
    ]
    event_id, photon, source, offset, x, z, t, kind = [], [], [], [0], [], [], [], []
    for i in range(min(n_events, len(truth))):
        for k, path_kind in enumerate(kinds):
            m = len(path_kind)
            event_id.append(truth.event_id[i])
            photon.append(k)
            source.append(0)
            x.extend(truth.x0[i] + rng.normal(0, 30, m))
            z.extend(truth.z0[i] + rng.normal(0, 30, m))
            t.extend(truth.t0[i] + np.arange(m) * 50.0)
            kind.extend(path_kind)
            offset.append(offset[-1] + m)
    return _build(
        Paths,
        {
            "event_id": event_id,
            "photon": photon,
            "source": source,
            "offset": offset,
            "x": x,
            "z": z,
            "t": t,
            "kind": kind,
        },
    )


def custom_geometry() -> Geometry:
    return Geometry.from_table([0, 0, 1], [0, 1, 0], [0.0, 0.0, 30.0], [0.0, 20.0, 5.0], radius=0.5)
