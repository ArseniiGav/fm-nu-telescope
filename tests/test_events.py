"""Events: prior statistics per type, muon segments, stochastic losses, the cascade profile, photon emission."""

import math

import numpy as np
import pytest
import torch

from helpers import KS_LIMIT, N_SIGMA, assert_binomial, ks_distance
from nutel.toy import Emission, Physics, make_geometry, preset, sample_prior
from nutel.toy.events import (
    ATMOSPHERIC,
    CASCADE,
    STARTING,
    TRACK,
    Sources,
    angle_from_zenith_azimuth,
    atmospheric_cos_zenith,
    box_interval,
    emit_photons,
    icecube_inelasticity,
    muon_segments,
    radel_wiebusch_angle,
    sample_bursts,
    type_codes,
    zenith_azimuth,
)

N = 40_000


@pytest.fixture(scope="module")
def prior():
    return sample_prior(N, preset("full"), rng=1)


def test_type_mix(prior):
    for code, p in zip((TRACK, STARTING, CASCADE, ATMOSPHERIC), (0.4, 0.2, 0.2, 0.2), strict=True):
        assert_binomial(np.sum(prior["type"] == code), N, p)
    only = sample_prior(100, preset(event_type="cascade"), rng=2)
    assert np.all(only["type"] == CASCADE)


def test_tracks_are_isotropic_lines_through_a_disc(prior):
    g = make_geometry()
    sel = prior["type"] == TRACK
    angle = np.mod(prior["angle"][sel], 2 * np.pi)
    assert ks_distance(angle, lambda a: a / (2 * np.pi)) < KS_LIMIT
    np.testing.assert_allclose(np.hypot(prior["dir_x"], prior["dir_z"]), 1.0, atol=1e-12)
    # the anchor is the closest point to the origin: perpendicular to the direction, signed b uniform
    x0, z0, dx, dz = (prior[k][sel] for k in ("x0", "z0", "dir_x", "dir_z"))
    np.testing.assert_allclose(x0 * dx + z0 * dz, 0.0, atol=1e-9)
    b = -x0 * dz + z0 * dx
    r_gen = g.half_diagonal + 20.0
    assert np.abs(b).max() <= r_gen
    assert ks_distance(b, lambda v: (v + r_gen) / (2 * r_gen)) < KS_LIMIT


def test_vertices_are_uniform_in_the_area(prior):
    sel = (prior["type"] == CASCADE) | (prior["type"] == STARTING)
    x, z = prior["x0"][sel], prior["z0"][sel]
    assert ks_distance(x, lambda v: (v + 195) / 390) < KS_LIMIT
    assert ks_distance(z, lambda v: (v + 130) / 260) < KS_LIMIT


def test_energies_are_log_uniform(prior):
    for code, (lo, hi) in (
        (TRACK, (1e2, 1e4)),
        (STARTING, (1e2, 1e3)),
        (CASCADE, (10, 1e3)),
        (ATMOSPHERIC, (1e2, 1e3)),
    ):
        e = prior["energy"][prior["type"] == code]
        assert e.min() >= lo
        assert e.max() <= hi
        assert ks_distance(np.log(e), lambda v, lo=lo, hi=hi: (v - math.log(lo)) / math.log(hi / lo)) < KS_LIMIT


def test_atmospheric_muons_are_down_going_and_concentrated_near_vertical(prior):
    sel = prior["type"] == ATMOSPHERIC
    zen, az = zenith_azimuth(prior["dir_x"][sel], prior["dir_z"][sel])
    cos_z = np.cos(zen)
    assert np.all(prior["dir_z"][sel] < 0)
    assert cos_z.min() >= 0.1 - 1e-12
    assert 0.80 < np.median(cos_z) < 0.86  # Becherini et al. at 2 km w.e. (literature: 0.83)
    assert_binomial(np.sum(az == 0), sel.sum(), 0.5)
    uniform = sample_prior(20_000, preset(event_type="atmospheric", atmospheric_zenith="uniform"), rng=3)
    cz = -uniform["dir_z"]
    assert ks_distance(cz, lambda v: (v - 0.1) / 0.9) < KS_LIMIT


def test_atmospheric_cos_zenith_cdf():
    u = np.random.default_rng(0).random(50_000)
    c = atmospheric_cos_zenith(u, 2.0, (0.1, 1.0))
    k1 = -0.581 * 2.0 + 0.034
    grid = np.linspace(0.1, 1.0, 20001)
    pdf = grid * np.exp(k1 / grid)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    assert ks_distance(c, lambda v: np.interp(v, grid, cdf)) < KS_LIMIT


def test_starting_tracks_split_the_energy():
    p = sample_prior(20_000, preset(event_type="starting"), rng=4)
    y = p["cascade_fraction"]
    assert np.all((y >= 0) & (y <= 1))
    mean = 1 - 0.75 / (4 / 3)
    sigma = y.std() / math.sqrt(len(y))
    assert abs(y.mean() - mean) < N_SIGMA * sigma
    assert np.all(np.isnan(sample_prior(100, preset(event_type="cascade"), rng=4)["cascade_fraction"]))
    uniform = sample_prior(20_000, preset(event_type="starting", inelasticity="uniform"), rng=5)["cascade_fraction"]
    assert ks_distance(uniform, lambda v: (v - 0.05) / 0.7) < KS_LIMIT


def test_icecube_inelasticity_cdf():
    u = np.random.default_rng(1).random(50_000)
    for eps in (0.0, 1.0, 3.0):
        y = icecube_inelasticity(u, eps)
        w = 1 - y
        # CDF of y: 1 - F_w(1 - y), F_w(w) = (w + eps w^3 / 3) / (1 + eps / 3)
        assert ks_distance(w, lambda v, e=eps: (v + e * v**3 / 3) / (1 + e / 3)) < KS_LIMIT


def test_fixed_values_replace_the_distributions():
    config = preset(
        fixed_angle=30.0, fixed_vertex=(10.0, -5.0), fixed_energy=500.0, fixed_t0=1234.0, fixed_inelasticity=0.3
    )
    p = sample_prior(200, config, rng=6)
    np.testing.assert_allclose(p["angle"], math.radians(30.0))
    np.testing.assert_allclose(p["energy"], 500.0)
    np.testing.assert_allclose(p["t0"], 1234.0)
    vertex = p["type"] == CASCADE
    np.testing.assert_allclose(p["x0"][vertex], 10.0)
    line = (p["type"] == TRACK) | (p["type"] == ATMOSPHERIC)
    # tracks: the anchor is the projection of the fixed point onto the line
    d = np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
    point = np.array([10.0, -5.0])
    anchor = point - (point @ d) * d
    np.testing.assert_allclose(p["x0"][line], anchor[0])
    np.testing.assert_allclose(p["cascade_fraction"][p["type"] == STARTING], 0.3)
    # the other parameters are the same as without the fixed values
    free = sample_prior(200, preset(fixed_energy=500.0), rng=6)
    base = sample_prior(200, preset(), rng=6)
    np.testing.assert_array_equal(free["angle"], base["angle"])
    np.testing.assert_array_equal(free["type"], base["type"])


def test_zenith_azimuth_conventions():
    zen, az = zenith_azimuth(np.array([0.0, 0.0, -1.0, 1.0]), np.array([-1.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(zen, [0, np.pi, np.pi / 2, np.pi / 2])
    np.testing.assert_allclose(az, [np.pi, np.pi, 0, np.pi])
    angles = np.random.default_rng(2).uniform(-np.pi, np.pi, 100)
    z, a = zenith_azimuth(np.cos(angles), np.sin(angles))
    np.testing.assert_allclose(angle_from_zenith_azimuth(z, a), angles, atol=1e-9)


def test_type_codes():
    np.testing.assert_array_equal(type_codes(["track", "cascade"]), [0, 2])
    np.testing.assert_array_equal(type_codes(3), [3])
    with pytest.raises(ValueError, match="unknown event types"):
        type_codes(["muon"])
    with pytest.raises(ValueError, match="codes must be"):
        type_codes([4])


def test_box_interval_and_segments():
    box = (-10.0, 10.0, -5.0, 5.0)
    lo, hi = box_interval(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 20.0, 3.0]), np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), box
    )
    np.testing.assert_allclose(lo[[0, 2]], [-10.0, -8.0])
    np.testing.assert_allclose(hi[[0, 2]], [10.0, 2.0])
    assert lo[1] > hi[1]  # misses the box
    kind = np.array([TRACK, STARTING, CASCADE])
    a, b = muon_segments(kind, np.zeros(3), np.zeros(3), np.ones(3), np.zeros(3), box)
    np.testing.assert_allclose(a, [-10.0, 0.0, 0.0])
    np.testing.assert_allclose(b, [10.0, 10.0, 0.0])


def test_burst_rate_and_fraction_distribution():
    em = Emission(stochastic_b=3e-3, stochastic_v_min=1e-3, energy_loss_a=0.0, muon_stop_energy=0.0)
    n, length = 4000, 500.0
    bursts = sample_bursts(np.zeros(n), np.full(n, length), np.full(n, 1e6), em, Physics(), np.random.default_rng(3))
    expected = em.burst_rate * length * n
    assert abs(len(bursts.energy) - expected) < N_SIGMA * math.sqrt(expected)
    assert np.all((bursts.s >= 0) & (bursts.s <= length))
    # the fractions follow p(v) ~ 1/v: log v uniform on [log v_min, 0]
    order = np.lexsort((bursts.s, bursts.event))
    ev, e = bursts.event[order], bursts.energy[order]
    before = np.full(len(e), 1e6)
    for i in range(1, len(e)):
        if ev[i] == ev[i - 1]:
            before[i] = before[i - 1] - e[i - 1]
    v = e / before
    assert ks_distance(np.log(v), lambda x: (x - math.log(1e-3)) / -math.log(1e-3)) < KS_LIMIT
    none = sample_bursts(
        np.zeros(5), np.full(5, 100.0), np.full(5, 1e3), em, Physics(stochastic=False), np.random.default_rng(3)
    )
    assert len(none.energy) == 0


def test_continuous_loss_stops_muons():
    em = Emission(energy_loss_a=0.25, stochastic_b=0.0)
    out = sample_bursts(
        np.zeros(3), np.full(3, 1000.0), np.array([50.0, 1000.0, 0.5]), em, Physics(), np.random.default_rng(0)
    )
    np.testing.assert_allclose(out.s_end, [(50.0 - 1.0) / 0.25, 1000.0, 0.0])
    np.testing.assert_array_equal(out.stopped, [True, False, True])
    np.testing.assert_allclose(out.energy_end, [1.0, 750.0, 0.5])


def test_radel_wiebusch_profile():
    u = np.random.default_rng(4).random(200_000)
    theta = np.degrees(radel_wiebusch_angle(u, 1.33))
    assert np.median(theta) == pytest.approx(42.0, abs=0.5)  # literature report: median 42.0 deg (n = 1.33)
    assert np.mean(np.abs(theta - np.degrees(np.arccos(1 / 1.33))) < 10) == pytest.approx(0.65, abs=0.02)
    assert np.mean(theta > 90) == pytest.approx(0.042, abs=0.005)


def _sources(n_seg, n_pt, profile_source=2):
    return Sources(
        seg_event=np.array([0]),
        seg_x=np.array([-50.0]),
        seg_z=np.array([0.0]),
        seg_dx=np.array([1.0]),
        seg_dz=np.array([0.0]),
        seg_length=np.array([100.0]),
        seg_t=np.array([10.0]),
        seg_n=np.array([n_seg]),
        pt_event=np.array([0, 1]),
        pt_x=np.array([5.0, -3.0]),
        pt_z=np.array([1.0, 2.0]),
        pt_dx=np.array([0.0, 1.0]),
        pt_dz=np.array([1.0, 0.0]),
        pt_t=np.array([20.0, 30.0]),
        pt_n=np.array([n_pt, n_pt]),
        pt_source=np.array([1, profile_source]),
    )


def test_track_emission_geometry_and_times():
    n_seg = 50_000
    generator = torch.Generator().manual_seed(0)
    ph = emit_photons(_sources(n_seg, 10), Emission(cherenkov_spread=0.0), 1.35, generator)
    assert len(ph) == n_seg + 20 == ph.source.numel()
    track = ph.source == 0
    x, t = ph.x[track].numpy(), ph.t[track].numpy()
    np.testing.assert_allclose(ph.z[track].numpy(), 0.0)
    assert ks_distance(x, lambda v: (v + 50) / 100) < KS_LIMIT
    np.testing.assert_allclose(t, 10.0 + (x + 50.0) / 0.299792458)
    angle = np.abs(np.degrees(np.arctan2(ph.uz[track].numpy(), ph.ux[track].numpy())))
    np.testing.assert_allclose(angle, np.degrees(np.arccos(1 / 1.35)), atol=1e-9)
    assert_binomial(np.sum(ph.uz[track].numpy() > 0), n_seg, 0.5)
    np.testing.assert_allclose(np.hypot(ph.ux.numpy(), ph.uz.numpy()), 1.0, atol=1e-12)
    assert torch.all(ph.event[1:] >= ph.event[:-1])  # ordered by event


def test_cascade_emission_profiles():
    n_pt = 50_000
    for profile, check in (("gaussian", 10.0), ("radel-wiebusch", None)):
        generator = torch.Generator().manual_seed(1)
        ph = emit_photons(_sources(0, n_pt), Emission(cascade_profile=profile), 1.33, generator)
        burst = ph.source == 1  # axis +z
        theta = np.degrees(np.abs(np.arctan2(ph.ux[burst].numpy(), ph.uz[burst].numpy())))
        np.testing.assert_allclose(ph.x[burst].numpy(), 5.0)
        np.testing.assert_allclose(ph.t[burst].numpy(), 20.0)
        if check is not None:
            dev = theta - np.degrees(np.arccos(1 / 1.33))
            assert np.std(dev) == pytest.approx(check, rel=0.03)
        else:
            assert np.median(theta) == pytest.approx(42.0, abs=0.6)
