"""Simulation driver: conservation, reproducibility, chunking, multi-geometry data sets, paths, physics validation."""

import hashlib
import itertools
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from helpers import N_SIGMA, check_conservation, direct_time, hits_on_own_geometry
from nutel.toy import (
    EVENT_TYPES,
    Detector,
    Geometry,
    PointType,
    generate,
    make_geometry,
    preset,
    resolve_device,
    sample_detectors,
    simulate,
)
from nutel.toy.simulate import _plan_chunks


def arrays(data):
    """Every array of a data set, by name."""
    out = {}
    for group in ("truth", "photons", "pulses", "hits", "bursts", "paths"):
        container = getattr(data, group)
        if container is not None:
            out |= {f"{group}/{k}": v for k, v in container.arrays().items()}
    return out


def assert_same(a, b):
    xa, xb = arrays(a), arrays(b)
    assert xa.keys() == xb.keys()
    for name, value in xa.items():
        np.testing.assert_array_equal(value, xb[name], err_msg=name)


# ----------------------------------------------------------------------------------------------------------------------
# Bookkeeping and determinism


@pytest.mark.parametrize(("absorption", "scattering", "stochastic"), list(itertools.product([False, True], repeat=3)))
def test_photon_conservation_for_every_process_combination(absorption, scattering, stochastic):
    config = preset(
        "full",
        absorption=absorption,
        scattering=scattering,
        stochastic=stochastic,
        efficiency=0.7,
        medium="ice",
        max_steps=40,
    )
    data = generate(40, config=config, seed=1, device="cpu", photons_per_chunk=20_000)
    check_conservation(data)
    p = data.photons
    assert p.n_noise.sum() >= 0
    assert p.n_absorbed_module.sum() > 0
    assert (p.n_absorbed.sum() > 0) == absorption
    assert (p.n_detected_scattered.sum() > 0) == scattering
    assert (p.n_emitted_burst.sum() > 0) == stochastic
    assert (data.truth.n_bursts.sum() > 0) == stochastic
    assert (p.n_lost.sum() > 0) == scattering  # ice, 40 steps
    assert hits_on_own_geometry(data)


def test_full_small_is_consistent(full_small):
    check_conservation(full_small)
    t = full_small.truth
    assert full_small.pulses.t.dtype == np.float32
    assert full_small.hits.x.dtype == np.float32
    assert set(np.unique(t.type)) == {0, 1, 2, 3}
    np.testing.assert_array_equal(t.event_id, np.arange(300))
    # energies and segments
    track = (t.type == 0) | (t.type == 3)
    np.testing.assert_array_equal(t.energy_track[track], t.energy[track])
    start = t.type == 1
    np.testing.assert_allclose(t.energy_track[start] + t.energy_cascade[start], t.energy[start], rtol=1e-5)
    np.testing.assert_allclose(t.energy_cascade[start], t.cascade_fraction[start] * t.energy[start], rtol=1e-5)
    cascade = t.type == 2
    assert np.all(t.track_length[cascade] == 0)
    assert np.all(t.energy_cascade[cascade] == t.energy[cascade])
    assert np.all(np.isnan(t.cascade_fraction[~start]))
    assert np.all(t.length_in_area <= t.track_length + 1e-3)
    assert np.all(t.track_length <= math.hypot(550, 420) + 1e-3)
    p = full_small.photons
    np.testing.assert_array_equal(p.n_emitted_track > 0, t.track_length > 0)
    assert np.all(p.n_emitted_cascade[(t.type == 0) | (t.type == 3)] == 0)
    # bursts
    b = full_small.bursts
    np.testing.assert_array_equal(b.counts(), t.n_bursts)
    np.testing.assert_allclose(np.bincount(b.event_index(), weights=b.energy, minlength=300), t.energy_lost, rtol=1e-4)
    ev = b.event_index()
    rho = np.abs((b.x - t.x0[ev]) * t.dir_z[ev] - (b.z - t.z0[ev]) * t.dir_x[ev])
    assert np.all(rho < 1e-2)  # on the muon line
    assert np.all(t.energy_lost <= t.energy_track + 1e-3)
    np.testing.assert_allclose(np.cos(t.zenith), -t.dir_z, atol=1e-6)
    np.testing.assert_allclose(t.angle, np.arctan2(t.dir_z, t.dir_x), atol=1e-6)


def test_muons_stop_with_continuous_losses():
    data = generate(300, config=preset("absorb", event_type="track", fixed_energy=30.0), seed=2, device="cpu")
    t = data.truth
    assert np.all(t.stopped)
    np.testing.assert_allclose(t.track_length, (30.0 - 1.0) / 0.25, rtol=1e-5)
    high = generate(50, config=preset("absorb", event_type="track", fixed_energy=1e4), seed=2, device="cpu")
    assert not np.any(high.truth.stopped)
    none = generate(
        50, config=preset("absorb", event_type="track", fixed_energy=30.0, energy_loss_a=0.0), seed=2, device="cpu"
    )
    assert not np.any(none.truth.stopped)


def test_same_seed_gives_identical_output():
    kwargs = {"config": preset("full"), "device": "cpu", "save_paths": 2}
    a, b = generate(80, seed=3, **kwargs), generate(80, seed=3, **kwargs)
    assert_same(a, b)
    c = generate(80, seed=4, **kwargs)
    assert not np.array_equal(a.truth.x0, c.truth.x0)


def test_event_quantities_do_not_depend_on_the_chunking():
    a = generate(60, config=preset("full"), seed=6, device="cpu", photons_per_chunk=5_000)
    b = generate(60, config=preset("full"), seed=6, device="cpu", photons_per_chunk=10**8)
    assert a.provenance["n_chunks"] > 5
    assert b.provenance["n_chunks"] == 1
    assert_same_truth = [np.testing.assert_array_equal(v, b.truth.arrays()[k]) for k, v in a.truth.arrays().items()]
    assert assert_same_truth
    for name in ("n_emitted", "n_emitted_track", "n_emitted_burst", "n_emitted_cascade", "n_noise"):
        np.testing.assert_array_equal(getattr(a.photons, name), getattr(b.photons, name))
    np.testing.assert_array_equal(a.bursts.energy, b.bursts.energy)
    check_conservation(a)


def test_chunk_plan():
    assert _plan_chunks(np.array([5, 5, 5, 5]), 10) == [(0, 2), (2, 4)]
    assert _plan_chunks(np.array([50, 1, 1]), 10) == [(0, 1), (1, 3)]
    assert _plan_chunks(np.array([], dtype=int), 10) == []


def test_simulate_reproduces_generate():
    config = preset("full", medium="ice")
    data = generate(50, config=config, seed=8, device="cpu")
    t = data.truth
    events = {k: getattr(t, k) for k in ("type", "angle", "x0", "z0", "t0", "energy", "cascade_fraction")}
    again = simulate(events, config=config, seed=8, device="cpu")
    # the truth is stored in float32: positions and times agree to float32 precision, the counts exactly
    for name in ("photons", "pulses", "hits", "bursts"):
        for k, v in getattr(data, name).arrays().items():
            w = getattr(again, name).arrays()[k]
            if v.dtype.kind == "f":
                np.testing.assert_allclose(v, w, rtol=1e-6, atol=1e-3, err_msg=f"{name}/{k}")
            else:
                np.testing.assert_array_equal(v, w, err_msg=f"{name}/{k}")
    for k, v in t.arrays().items():
        np.testing.assert_allclose(v, again.truth.arrays()[k], rtol=1e-5, atol=1e-4, err_msg=k)


def test_simulate_given_events():
    data = simulate(
        {
            "type": ["track", "cascade", "starting"],
            "dir_x": [1.0, 0.0, 0.0],
            "dir_z": [0.0, 2.0, -1.0],
            "x0": [5.0, 0.0, 10.0],
            "z0": [0.0, 0.0, 10.0],
            "t0": 1500.0,
            "energy": [1e3, 100.0, 500.0],
            "cascade_fraction": [np.nan, np.nan, 0.5],
        },
        config=preset("absorb"),
        seed=1,
        device="cpu",
    )
    t = data.truth
    np.testing.assert_array_equal(t.type, [0, 2, 1])
    np.testing.assert_allclose(t.x0, [0.0, 0.0, 10.0])  # the track anchor is the projection on the line
    np.testing.assert_allclose(t.dir_z, [0.0, 1.0, -1.0])
    np.testing.assert_allclose(t.energy_cascade, [0.0, 100.0, 250.0])
    check_conservation(data)
    with pytest.raises(ValueError, match="cascade_fraction"):
        simulate({"type": "starting", "angle": 0.0, "x0": 0, "z0": 0, "t0": 0, "energy": 100.0}, device="cpu")
    with pytest.raises(ValueError, match="direction"):
        simulate({"type": 0, "x0": 0, "z0": 0, "t0": 0, "energy": 100.0}, device="cpu")
    with pytest.raises(ValueError, match="missing"):
        simulate({"type": 0, "angle": 0.0, "x0": 0, "z0": 0, "energy": 100.0}, device="cpu")
    with pytest.raises(ValueError, match="geometry_index"):
        simulate({"type": 0, "angle": 0.0, "x0": 0, "z0": 0, "t0": 0, "energy": 1.0, "geometry_index": 1}, device="cpu")


def test_levels_paths_and_options():
    data = generate(10, config=preset("scatter"), seed=1, device="cpu", level="pulses")
    assert data.hits is None
    assert len(data.pulses) > 0
    with pytest.raises(ValueError, match="level"):
        generate(5, seed=1, device="cpu", level="photons")
    with pytest.raises(ValueError, match="n_geometries"):
        generate(5, n_geometries=3, device="cpu")
    with pytest.raises(ValueError, match="seed"):
        generate(5, seed=-1, device="cpu")
    with pytest.raises(ValueError, match="unsupported device"):
        resolve_device("tpu")
    shifted = generate(5, seed=1, device="cpu", first_event_id=100)
    np.testing.assert_array_equal(shifted.truth.event_id, np.arange(100, 105))
    narrow = generate(30, config=preset("absorb", window=(1500.0, 1600.0)), seed=2, device="cpu")
    assert narrow.photons.n_out_of_window.sum() > 0
    check_conservation(narrow)


def test_paths(full_small):
    paths = full_small.paths
    assert set(np.unique(paths.event_id)) == {0, 1, 2}
    p = full_small.photons
    for i in range(3):
        k = paths.of_event(i)
        assert len(k) == p.n_emitted[i]
        np.testing.assert_array_equal(np.sort(paths.photon[k]), np.arange(p.n_emitted[i]))
        fate = paths.fate[k]
        assert np.sum(fate == PointType.DETECT) == p.n_detected[i]
        assert np.sum(fate == PointType.ABSORB) == p.n_absorbed[i]
        assert np.sum(fate == PointType.ESCAPE) == p.n_escaped[i]
        assert np.sum((fate == PointType.DETECT) & (paths.n_scatter[k] == 0)) == p.n_detected_direct[i]
        np.testing.assert_array_equal(
            np.bincount(paths.source[k], minlength=3),
            [p.n_emitted_track[i], p.n_emitted_burst[i], p.n_emitted_cascade[i]],
        )
    _x, _z, t, kind = paths.polyline(0)
    assert kind[0] == PointType.EMIT
    assert np.all(np.diff(t) >= 0)
    sub = full_small.select(np.array([2, 5]))
    assert set(np.unique(sub.paths.event_id)) == {2}


def test_select_event_and_views(full_small):
    mask = full_small.truth.type == 2
    sub = full_small.select(mask)
    assert len(sub) == mask.sum()
    assert np.all(sub.truth.type == 2)
    check_conservation(sub)
    i = int(np.flatnonzero(full_small.photons.n_hits > 3)[0])
    ev = full_small.event(i)
    assert ev.type_name == EVENT_TYPES[full_small.truth.type[i]]
    assert len(ev.hits) == full_small.photons.n_hits[i]
    assert len(ev.pulses) == full_small.photons.n_pulses[i]
    assert ev.geometry == full_small.geometry(i)
    assert ev.detector == full_small.detector(i)
    assert ev.truth["event_id"] == i
    assert full_small.event(-1).index == 299
    with pytest.raises(IndexError):
        full_small.event(300)
    reordered = full_small.select(np.array([5, 1]))
    np.testing.assert_array_equal(reordered.truth.event_id, [5, 1])
    assert "300 events" in repr(full_small)


# ----------------------------------------------------------------------------------------------------------------------
# Multi-detector data sets


def test_multi_geometry_data_set(multi_small):
    data = multi_small
    check_conservation(data)
    assert data.n_geometries == 4
    assert len(data) == 160
    np.testing.assert_array_equal(data.truth.geometry_index, np.repeat(np.arange(4), 40))
    assert hits_on_own_geometry(data)
    assert len({g.geometry_id for g in data.geometries}) == 4
    assert data.detectors == sample_detectors(4, "default", seed=3)
    for k, det in enumerate(data.detectors):
        _x0, x1, _z0, _z1 = det.geometry.bounds
        sel = data.truth.geometry_index == k
        vertex = data.truth.type[sel] == 2
        assert np.all(data.truth.x0[sel][vertex] <= x1 + 20 + 1e-3)
        assert data.geometry(int(np.flatnonzero(sel)[0])) == det.geometry
    assert data.provenance["family"]["name"] == "default"
    assert data.provenance["events_per_geometry"] == [40] * 4


def test_same_event_in_several_detectors():
    config = preset(
        "scatter", event_type="track", fixed_angle=200.0, fixed_vertex=(0.0, 10.0), fixed_energy=2000.0, fixed_t0=1500.0
    )
    data = generate(n_geometries=5, events_per_geometry=1, family="default", config=config, seed=4, device="cpu")
    for name in ("angle", "x0", "z0", "t0", "energy"):
        assert np.ptp(getattr(data.truth, name)) == 0
    assert len(set(data.photons.n_detected.tolist())) > 1  # different detectors see different light
    assert hits_on_own_geometry(data)


def test_explicit_detectors_and_uneven_split():
    dets = [
        Detector(make_geometry(), preset().medium, preset().readout),
        Detector(make_geometry(layout="core"), preset(medium="ice").medium, preset(noise_rate=0.0).readout),
    ]
    data = generate(7, detectors=dets, seed=5, device="cpu")
    np.testing.assert_array_equal(np.bincount(data.truth.geometry_index), [4, 3])
    assert data.detectors[0].readout.noise_rate == 50_000.0  # resolved from the medium
    assert data.photons.n_noise[data.truth.geometry_index == 1].sum() == 0
    with pytest.raises(ValueError, match="events_per_geometry"):
        generate(7, detectors=dets, events_per_geometry=3, device="cpu")


# ----------------------------------------------------------------------------------------------------------------------
# Physics validation (the numbers are quoted in the core report)


def test_direct_light_arrives_at_the_cherenkov_time():
    # tiny modules: the first hit of a module is at most r / c_n = 0.09 ns before the time at the module centre
    config = preset(
        "straight", event_type="track", cherenkov_spread=0.0, module_radius=0.02, track_yield=200.0, fixed_energy=1e4
    )
    data = generate(15, config=config, seed=3, device="cpu")
    h, t, m = data.hits, data.truth, data.detectors[0].medium
    ev = h.event_index()
    expected = direct_time(h.x, h.z, t.x0[ev], t.z0[ev], t.dir_x[ev], t.dir_z[ev], t.t0[ev], m.n)
    residual = h.t - expected
    key = ev * 10_000 + h.string * 100 + h.module
    _, first = np.unique(key, return_index=True)
    assert len(first) > 100
    assert np.abs(residual[first]).max() < 0.1
    # default modules: every direct photon arrives within [-r / c_n, 0] of the centre time (off-track modules)
    config = preset("straight", event_type="track", cherenkov_spread=0.0, fixed_energy=1e4, n_group=1.38)
    data = generate(20, config=config, seed=4, device="cpu")
    h, t, m = data.hits, data.truth, data.detectors[0].medium
    ev = h.event_index()
    expected = direct_time(h.x, h.z, t.x0[ev], t.z0[ev], t.dir_x[ev], t.dir_z[ev], t.t0[ev], m.n, m.n_group)
    rho = np.abs((h.x - t.x0[ev]) * t.dir_z[ev] - (h.z - t.z0[ev]) * t.dir_x[ev])
    off = rho > 1.0
    res = (h.t - expected)[off]
    assert res.min() > -0.5 / m.speed_of_light - 2e-3
    assert res.max() < 2e-3


def _fit_distance_law(distance, counts, n_events):
    ok = counts > 0
    y, w = np.log(counts[ok] / n_events), np.sqrt(counts[ok])
    x = np.stack([np.ones(ok.sum()), -np.log(distance[ok]), -distance[ok]], 1)
    beta, *_ = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)
    return beta[1], 1.0 / beta[2]


@pytest.mark.parametrize("medium", ["water", "ice"])
def test_cascade_light_falls_as_exp_over_rho(medium):
    k = np.arange(40)
    rho = np.geomspace(10, 200, 40)
    phi = 2 * np.pi * ((k * 17) % 40) / 40  # distinct angles: no module shadows another
    g = Geometry.from_table(np.zeros(40, int), k, rho * np.cos(phi), rho * np.sin(phi))
    config = preset(
        "absorb", medium=medium, event_type="cascade", fixed_vertex=(0.0, 0.0), fixed_energy=50.0, custom_geometry=g
    )
    data = generate(8000, config=config, seed=5, device="cpu")
    counts = np.bincount(g.module_index(data.hits.string, data.hits.module), minlength=40).astype(float)
    alpha, lam = _fit_distance_law(rho, counts, 8000)
    assert alpha == pytest.approx(1.0, abs=0.1)
    assert lam == pytest.approx(data.detectors[0].medium.absorption_length, rel=0.1)


@pytest.mark.parametrize("medium", ["water", "ice"])
def test_track_light_has_no_geometric_fall_off(medium):
    theta = preset(medium=medium).medium.cherenkov_angle
    rho = np.geomspace(5, 150, 30)
    k = np.arange(30)
    g = Geometry.from_table(np.zeros(30, int), k, rho / math.tan(theta) + 8.0 * k, rho)
    config = preset(
        "absorb",
        medium=medium,
        event_type="track",
        fixed_vertex=(0.0, 0.0),
        fixed_angle=0.0,
        fixed_energy=1e4,
        custom_geometry=g,
        track_yield=20.0,
    )
    data = generate(200, config=config, seed=6, device="cpu")
    counts = np.bincount(g.module_index(data.hits.string, data.hits.module), minlength=30).astype(float)
    alpha, lam = _fit_distance_law(rho / math.sin(theta), counts, 200)
    assert abs(alpha) < 0.1
    assert lam == pytest.approx(data.detectors[0].medium.absorption_length, rel=0.1)


def test_scattering_delays_light_more_in_ice():
    medians = {}
    for medium in ("water", "ice"):
        config = preset("scatter", medium=medium, event_type="track", fixed_energy=1e4)
        data = generate(150, config=config, seed=7, device="cpu")
        h, t, m = data.hits, data.truth, data.detectors[0].medium
        ev = h.event_index()
        expected = direct_time(h.x, h.z, t.x0[ev], t.z0[ev], t.dir_x[ev], t.dir_z[ev], t.t0[ev], m.n)
        scattered = h.n_scatter > 0
        medians[medium] = float(np.median((h.t - expected)[scattered]))
        assert medians[medium] > 0
        assert np.median((h.t - expected)[~scattered]) < 0.01
    assert medians["ice"] > medians["water"]


def test_noise_hits_are_poisson():
    data = generate(
        3000,
        config=preset("full", event_type="cascade", fixed_energy=10.0, noise_rate=20_000.0),
        seed=8,
        device="cpu",
        level="pulses",
    )
    n = data.photons.n_noise
    mean = 20_000 * 5000e-9 * 96
    assert abs(n.mean() - mean) < N_SIGMA * math.sqrt(mean / len(n))
    assert n.var() == pytest.approx(mean, rel=0.1)


# ----------------------------------------------------------------------------------------------------------------------
# Slow: fresh-process reproducibility, the calibration, timing, CUDA

SCRIPT = """
import hashlib, sys
import numpy as np, torch
torch.set_num_threads(4)
from nutel.toy import generate, preset
data = generate(150, config=preset("full", medium="ice"), seed=21, device="cpu", photons_per_chunk=200_000)
h = hashlib.sha256()
for group in ("truth", "photons", "pulses", "hits"):
    for k, v in getattr(data, group).arrays().items():
        h.update(k.encode()); h.update(np.ascontiguousarray(v).tobytes())
print(h.hexdigest())
"""


@pytest.mark.slow
def test_fresh_process_reproducibility():
    env = dict(os.environ)
    runs = [
        subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env, check=True)
        for _ in range(2)
    ]
    digests = [r.stdout.strip().splitlines()[-1] for r in runs]
    assert digests[0] == digests[1]
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        data = generate(150, config=preset("full", medium="ice"), seed=21, device="cpu", photons_per_chunk=200_000)
    finally:
        torch.set_num_threads(threads)
    h = hashlib.sha256()
    for group in ("truth", "photons", "pulses", "hits"):
        for k, v in getattr(data, group).arrays().items():
            h.update(k.encode())
            h.update(np.ascontiguousarray(v).tobytes())
    assert h.hexdigest() == digests[0]


@pytest.mark.slow
def test_calibration_median_signal_photons():
    data = generate(2000, config=preset("full"), seed=2026, device="cpu", level="pulses")
    median = np.median(data.photons.n_detected)
    assert 25 <= median <= 35


@pytest.mark.slow
def test_timing_per_preset_and_medium(capsys):
    for name, medium in itertools.product(("straight", "absorb", "scatter", "full"), ("water", "ice")):
        start = time.perf_counter()
        generate(300, config=preset(name, medium=medium), seed=1, device="cpu", level="pulses")
        with capsys.disabled():
            print(f"\n{name:<8} {medium:<5} {1e3 * (time.perf_counter() - start) / 300:.2f} ms/event")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu_statistics():
    cpu = generate(400, config=preset("full"), seed=9, device="cpu", level="pulses")
    gpu = generate(400, config=preset("full"), seed=9, device="cuda", level="pulses")
    np.testing.assert_array_equal(cpu.photons.n_emitted, gpu.photons.n_emitted)
    a, b = cpu.photons.n_detected.sum(), gpu.photons.n_detected.sum()
    assert abs(a - b) < N_SIGMA * math.sqrt(a + b)
    assert hashlib is not None
