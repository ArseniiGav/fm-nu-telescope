import math
from types import SimpleNamespace

import numpy as np
import pytest

from nutel.toy.config import C_VACUUM
from nutel.toy.reco import (
    angular_error,
    causal_cleaning,
    direction_angle,
    emission_coordinate,
    energy_from_charge,
    event_index,
    expected_arrival_time,
    fit_charge_calibration,
    linefit,
    linefit_events,
    time_residuals_arrays,
    track_coordinates,
    unit_vector,
)

C = C_VACUUM


# ---------------------------------------------------------------------------------------------------------- geometry


def test_event_index_and_offsets():
    np.testing.assert_array_equal(event_index([0, 2, 2, 5]), [0, 0, 2, 2, 2])
    with pytest.raises(ValueError, match="offset"):
        event_index([1, 2])
    with pytest.raises(ValueError, match="offset"):
        event_index([0, 3, 2])


def test_direction_helpers():
    angle = np.array([0.0, 0.5, -2.0, np.pi])
    d = unit_vector(angle)
    np.testing.assert_allclose(np.hypot(d[:, 0], d[:, 1]), 1.0)
    np.testing.assert_allclose(direction_angle(d[:, 0], d[:, 1]), [0.0, 0.5, -2.0, np.pi], atol=1e-12)


def test_track_coordinates():
    s, rho = track_coordinates([[3.0, 4.0], [3.0, -4.0]], [0.0, 0.0], [2.0, 0.0])
    np.testing.assert_allclose(s, [3.0, 3.0])
    np.testing.assert_allclose(rho, [4.0, -4.0])  # positive on the left of the direction


# ----------------------------------------------------------------------------------------------------------- linefit


def test_linefit_recovers_a_perfect_line():
    rng = np.random.default_rng(1)
    angle = 2.3
    d = unit_vector(angle)
    t = np.sort(rng.uniform(0, 1000, 30))
    anchor = np.array([12.0, -7.0])
    points = anchor + np.outer(t - 300.0, 0.25 * d)
    w = rng.uniform(1, 5, 30)
    fit = linefit(points[:, 0], points[:, 1], t, w)
    np.testing.assert_allclose(fit.direction, d, atol=1e-12)
    assert float(fit.speed) == pytest.approx(0.25)
    assert float(fit.angle) == pytest.approx(angle)
    # the anchor is the weighted mean position, the point of the line at the weighted mean time
    t_bar = np.sum(w * t) / np.sum(w)
    assert float(fit.time) == pytest.approx(t_bar)
    np.testing.assert_allclose(fit.anchor, anchor + (t_bar - 300.0) * 0.25 * d)


def test_linefit_events_matches_single_and_handles_degenerate_events():
    rng = np.random.default_rng(2)
    sizes = [5, 0, 1, 3, 4]
    offset = np.concatenate([[0], np.cumsum(sizes)])
    x, z, t, q = (rng.normal(size=offset[-1]) for _ in range(4))
    q = np.abs(q)
    t[offset[4] : offset[5]] = 7.0  # no spread in time
    fit = linefit_events(offset, x, z, t, q)
    assert fit.direction.shape == (5, 2)
    single = linefit(x[:5], z[:5], t[:5], q[:5])
    np.testing.assert_allclose(fit.direction[0], single.direction)
    np.testing.assert_allclose(fit.speed[0], single.speed)
    assert np.all(np.isnan(fit.direction[[1, 2, 4]]))
    assert np.isfinite(fit.direction[3]).all()
    assert np.isnan(fit.anchor[1]).all()
    assert np.isfinite(fit.anchor[2]).all()
    with pytest.raises(ValueError, match="non-negative"):
        linefit_events(offset, x, z, t, -q)
    with pytest.raises(ValueError, match="length"):
        linefit_events(offset, x[:-1], z, t)


# ------------------------------------------------------------------------------------------------------------ energy


def test_charge_calibration():
    rng = np.random.default_rng(3)
    energy = 10 ** rng.uniform(2, 4, 2000)
    charge = energy / 7.0 * np.exp(rng.normal(0, 0.1, energy.size))
    charge[:10] = 0.0  # ignored
    for method in ("log_median", "log_mean", "least_squares"):
        k = fit_charge_calibration(charge, energy, method=method)
        assert k == pytest.approx(7.0, rel=0.03), method
    np.testing.assert_allclose(energy_from_charge([0.0, 2.0], 7.0), [0.0, 14.0])
    with pytest.raises(ValueError, match="method"):
        fit_charge_calibration(charge, energy, method="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        fit_charge_calibration([0.0], [1.0])
    with pytest.raises(ValueError, match="positive"):
        energy_from_charge([1.0], 0.0)


# ------------------------------------------------------------------------------------------------ direct-light times


def test_expected_arrival_time_by_hand():
    n = 1.33
    # a track along +x through the origin at t0 = 100 ns; a module at (30, 20)
    t = expected_arrival_time([30.0, 20.0], [0.0, 0.0], [1.0, 0.0], 100.0, n)
    theta_c = math.acos(1 / n)
    s_e = 30.0 - 20.0 / math.tan(theta_c)  # emission point on the track
    by_hand = 100.0 + s_e / C + n * (20.0 / math.sin(theta_c)) / C
    assert float(t) == pytest.approx(by_hand, abs=1e-9)
    assert float(t) == pytest.approx(100.0 + (30.0 + 20.0 * math.sqrt(n**2 - 1)) / C, abs=1e-9)
    # the same module below the track: the other ray, the same time
    t_below = expected_arrival_time([30.0, -20.0], [0.0, 0.0], [1.0, 0.0], 100.0, n)
    assert float(t_below) == pytest.approx(float(t))
    # the module radius: the front touches the disc n r / c earlier
    t_r = expected_arrival_time([30.0, 20.0], [0.0, 0.0], [1.0, 0.0], 100.0, n, radius=0.5)
    assert float(t - t_r) == pytest.approx(n * 0.5 / C)


def test_expected_arrival_time_with_a_group_index():
    n, n_g = 1.35, 1.376
    theta_c = math.acos(1 / n)
    s_e = 30.0 - 20.0 / math.tan(theta_c)  # the photon leaves at the phase Cherenkov angle ...
    by_hand = 100.0 + s_e / C + n_g * (20.0 / math.sin(theta_c)) / C  # ... and travels at c / n_group
    t = expected_arrival_time([30.0, 20.0], [0.0, 0.0], [1.0, 0.0], 100.0, n, n_group=n_g)
    assert float(t) == pytest.approx(by_hand, abs=1e-9)
    point = expected_arrival_time([30.0, 40.0], [0.0, 0.0], [1.0, 0.0], 0.0, n, n_group=n_g, event_type=2)
    assert float(point) == pytest.approx(n_g * 50.0 / C)
    same = expected_arrival_time([30.0, 20.0], [0.0, 0.0], [1.0, 0.0], 100.0, n, n_group=n)
    assert float(same) == pytest.approx(float(expected_arrival_time([30.0, 20.0], [0.0, 0.0], [1.0, 0.0], 100.0, n)))
    with pytest.raises(ValueError, match="group"):
        expected_arrival_time([30.0, 20.0], [0.0, 0.0], [1.0, 0.0], 100.0, n, n_group=0.5)


def test_expected_arrival_time_is_the_earliest_photon_time():
    """Brute force: the minimum over emission points of the photon arrival time equals the formula."""
    n = 1.32
    anchor, d = np.array([5.0, -3.0]), unit_vector(0.7)
    module = np.array([[40.0, 60.0]])
    s = np.linspace(-500, 500, 400_001)
    emit = anchor + np.outer(s, d)
    times = 50.0 + s / C + n * np.hypot(*(module - emit).T) / C
    assert float(expected_arrival_time(module, anchor, d, 50.0, n)[0]) == pytest.approx(times.min(), abs=1e-6)


def test_expected_arrival_time_by_event_type():
    n = 1.33
    anchor, d = np.zeros(2), np.array([1.0, 0.0])
    modules = np.array([[50.0, 10.0], [-50.0, 10.0]])  # ahead of and behind the vertex
    track = expected_arrival_time(modules, anchor, d, 0.0, n)
    point = n * np.hypot(*modules.T) / C
    np.testing.assert_allclose(expected_arrival_time(modules, anchor, d, 0.0, n, event_type=[0, 0]), track)
    np.testing.assert_allclose(expected_arrival_time(modules, anchor, d, 0.0, n, event_type=[3, 3]), track)
    np.testing.assert_allclose(expected_arrival_time(modules, anchor, d, 0.0, n, event_type=[2, 2]), point)
    starting = expected_arrival_time(modules, anchor, d, 0.0, n, event_type=[1, 1])
    assert starting[0] == pytest.approx(track[0])  # ahead: track light
    assert starting[1] == pytest.approx(point[1])  # behind the vertex: no track light, the cascade's
    assert np.all(track <= point + 1e-9)  # Fermat: track light is never later than light from a point on it
    s_e = emission_coordinate(modules, anchor, d, n)
    assert s_e[0] > 0 > s_e[1]
    with pytest.raises(ValueError, match="refractive"):
        expected_arrival_time(modules, anchor, d, 0.0, 1.0)
    with pytest.raises(ValueError, match="exceed 1"):
        emission_coordinate(modules, anchor, d, 1.0)


def test_time_residuals_arrays_are_zero_for_direct_light():
    n = 1.33
    rng = np.random.default_rng(4)
    modules = rng.uniform(-100, 100, (12, 2))
    truth = {"event_type": np.array([0, 2]), "x0": np.array([0.0, 10.0]), "z0": np.array([5.0, -5.0])}
    truth |= {"dir_x": np.array([0.6, 1.0]), "dir_z": np.array([-0.8, 0.0]), "t0": np.array([100.0, 300.0])}
    offset = np.array([0, 7, 12])
    expected = np.concatenate(
        [
            expected_arrival_time(modules[:7], [0.0, 5.0], [0.6, -0.8], 100.0, n),
            expected_arrival_time(modules[7:], [10.0, -5.0], [1.0, 0.0], 300.0, n, event_type=2),
        ]
    )
    residuals = time_residuals_arrays(offset, modules[:, 0], modules[:, 1], expected + 3.0, n=n, **truth)
    np.testing.assert_allclose(residuals, 3.0, atol=1e-9)


# ----------------------------------------------------------------------------------------------------------- angular


def test_angular_error_wraps():
    np.testing.assert_allclose(
        angular_error(np.radians([359.0, 90.0, 10.0]), np.radians([1.0, -90.0, 10.0])), [2, 180, 0]
    )
    np.testing.assert_allclose(angular_error([170.0, -170.0], [-170.0, 170.0], degrees=True), [20.0, 20.0])
    assert np.isnan(angular_error(np.nan, 0.0))
    rng = np.random.default_rng(5)
    a, b = rng.uniform(-10, 10, (2, 1000))
    err = angular_error(a, b)
    assert np.all((err >= 0) & (err <= 180))
    np.testing.assert_allclose(err, angular_error(b, a))


# ------------------------------------------------------------------------------------------------------------ causal


def test_causal_cleaning_removes_isolated_noise():
    # event 0: a muon along +x at z = 0 seen by modules at z = 20; plus one noise hit 2 us later
    xs = np.arange(0.0, 200.0, 25.0)
    modules = np.stack([xs, np.full_like(xs, 20.0)], axis=1)
    t_signal = expected_arrival_time(modules, [0.0, 0.0], [1.0, 0.0], 100.0, 1.33)
    x = np.concatenate([xs, [100.0], [0.0, 30.0]])
    z = np.concatenate([np.full_like(xs, 20.0), [-40.0], [0.0, 0.0]])
    t = np.concatenate([t_signal, [2100.0], [10.0, 4000.0]])  # event 1: two unrelated noise hits
    hits = SimpleNamespace(offset=np.array([0, len(xs) + 1, len(xs) + 3]), x=x, z=z, t=t)
    keep = causal_cleaning(hits, slack=10.0, max_distance=math.inf, min_neighbours=1)
    assert keep[: len(xs)].all()
    assert not keep[len(xs)]
    assert not keep[len(xs) + 1 :].any()
    # locality: with a short max_distance the signal hits 25 m apart are still connected
    assert causal_cleaning(hits, slack=10.0, max_distance=30.0, min_neighbours=1)[: len(xs)].all()
    assert not causal_cleaning(hits, slack=10.0, max_distance=20.0, min_neighbours=1)[: len(xs)].any()
    # the defaults: 75 m and two partners; the chain of modules 25 m apart keeps its inner hits
    default = causal_cleaning(hits)
    assert default[1 : len(xs) - 1].all()
    assert not default[len(xs) :].any()
    # more neighbours required: the ends of the chain need two partners within 30 m and lose them
    keep2 = causal_cleaning(hits, slack=10.0, max_distance=30.0, min_neighbours=2)
    assert not keep2[0]
    assert keep2[1:7].all()
    assert not keep2[7]
    with pytest.raises(ValueError, match="slack"):
        causal_cleaning(hits, slack=-1.0)


def test_time_residuals_of_a_data_set():
    from nutel.toy.reco import time_residuals
    from synthetic import synthetic_data, two_detectors

    data = synthetic_data(8, seed=9, detectors=two_detectors())
    centre = time_residuals(data, to_surface=False)
    assert centre.shape == (len(data.pulses),)
    signal = data.pulses.n_noise == 0
    # the synthetic pulses are direct-light times plus a short exponential delay
    assert np.all(centre[signal] >= -1e-3)
    assert np.all(centre[signal] < 60.0)
    surface = time_residuals(data)
    n_g = np.array([d.medium.group_index for d in data.detectors])
    n = n_g[data.truth.geometry_index][data.pulses.event_index()]
    np.testing.assert_allclose(surface - centre, n * 0.5 / C, rtol=1e-5)
    hits = time_residuals(data, level="hits", to_surface=False)
    assert hits.shape == (len(data.hits),)
    with pytest.raises(ValueError, match="no hits level"):
        time_residuals(synthetic_data(2, hits=False), level="hits")
