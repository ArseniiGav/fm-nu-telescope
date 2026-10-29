"""Readout: pulse merging, noise statistics, window and jitter, the noise augmentation."""

import math

import numpy as np
import pytest

from helpers import KS_LIMIT, N_SIGMA, check_conservation, ks_distance
from nutel.toy import add_noise, generate, preset
from nutel.toy.readout import apply_jitter_and_window, merge_pulses, pulses_from_hits, sample_noise


def test_greedy_merging_by_hand():
    #            event 0, module 7: 0, 2, 3.5, 4, 20 | event 0, module 8: 1 | event 1, module 7: 5
    event = np.array([0, 0, 0, 0, 0, 0, 1])
    key = np.array([7, 7, 7, 8, 7, 7, 7])
    t = np.array([3.5, 0.0, 2.0, 1.0, 20.0, 4.0, 5.0])
    noise = np.array([False, True, False, False, False, False, True])
    out = merge_pulses(event, key, t, noise, 3.5)
    # pulses sorted by (event, t): [0, 3.5] (t0 = 0: hits 0, 2, 3.5), [1] (module 8), [4] (module 7), [20], event 1 [5]
    np.testing.assert_array_equal(out["event"], [0, 0, 0, 0, 1])
    np.testing.assert_allclose(out["t"], [0.0, 1.0, 4.0, 20.0, 5.0])
    np.testing.assert_array_equal(out["charge"], [3, 1, 1, 1, 1])
    np.testing.assert_array_equal(out["n_noise"], [1, 0, 0, 0, 1])
    np.testing.assert_allclose(out["t_mean"], [5.5 / 3, 1.0, 4.0, 20.0, 5.0])
    np.testing.assert_array_equal(out["pulse_of_hit"], [0, 0, 0, 1, 3, 2, 4])
    np.testing.assert_array_equal(key[out["first_hit"]], [7, 8, 7, 7, 7])
    empty = merge_pulses([], [], [], [], 3.5)
    assert len(empty["t"]) == 0


def test_merge_window_zero_and_large():
    t = np.array([0.0, 1.0, 1.0, 50.0])
    z = np.zeros(4, int)
    assert len(merge_pulses(z, z, t, np.zeros(4, bool), 0.0)["t"]) == 3  # equal times still merge
    assert merge_pulses(z, z, t, np.zeros(4, bool), 1e9)["charge"].tolist() == [4]


def test_charge_sum_equals_hits_per_module(full_small):
    h, p = full_small.hits, full_small.pulses
    kh = h.event_index() * 10_000 + h.string * 100 + h.module
    kp = p.event_index() * 10_000 + p.string * 100 + p.module
    keys, counts = np.unique(kh, return_counts=True)
    pk, inverse = np.unique(kp, return_inverse=True)
    charge = np.bincount(inverse, weights=p.charge)
    np.testing.assert_array_equal(keys, pk)
    np.testing.assert_array_equal(counts, charge)
    # pulses of one module are separated by more than the merge window
    order = np.lexsort((p.t, kp))
    same = np.diff(kp[order]) == 0
    assert np.all(np.diff(p.t[order].astype(np.float64))[same] > full_small.detectors[0].readout.merge_window - 1e-3)
    rebuilt = pulses_from_hits(h, full_small.detectors[0].readout.merge_window)
    for name, value in rebuilt.arrays().items():
        np.testing.assert_array_equal(value, getattr(p, name), err_msg=name)


def test_noise_statistics():
    rng = np.random.default_rng(1)
    n = 20_000
    event, module, t = sample_noise(np.full(n, 96), np.full(n, 50_000.0), np.zeros(n), np.full(n, 5000.0), rng)
    counts = np.bincount(event, minlength=n)
    mean = 96 * 50_000 * 5000e-9
    assert abs(counts.mean() - mean) < N_SIGMA * math.sqrt(mean / n)
    assert counts.var() == pytest.approx(mean, rel=0.05)  # Poisson: variance = mean
    assert ks_distance(t, lambda v: v / 5000.0) < KS_LIMIT
    assert module.min() >= 0
    assert module.max() == 95
    assert np.all(np.diff(event) >= 0)


def test_jitter_and_window():
    rng = np.random.default_rng(2)
    t = np.full(100_000, 100.0)
    jittered, inside = apply_jitter_and_window(t, 2.0, (0.0, 101.0), rng)
    assert jittered.std() == pytest.approx(2.0, rel=0.02)
    assert inside.mean() == pytest.approx(0.6915, abs=0.01)  # P(N(0, 2) < 1)
    same, _ = apply_jitter_and_window(t, 0.0, (0.0, 5000.0), rng)
    np.testing.assert_array_equal(same, t)


def test_full_preset_noise_and_window(full_small):
    check_conservation(full_small)
    p = full_small.photons
    mean = 96 * 50_000 * 5000e-9
    assert abs(p.n_noise.mean() - mean) < N_SIGMA * math.sqrt(mean / len(p))
    h = full_small.hits
    assert np.all((h.t >= 0) & (h.t < 5000))
    assert np.all(h.source[h.is_noise] == 3)


def test_add_noise_augmentation():
    clean = generate(200, config=preset("scatter"), seed=3, device="cpu")
    assert clean.photons.n_noise.sum() == 0
    noisy = add_noise(clean, 20_000.0, rng=4)
    check_conservation(noisy)
    mean = 96 * 20_000 * 5000e-9
    assert abs(noisy.photons.n_noise.mean() - mean) < N_SIGMA * math.sqrt(mean / 200)
    assert noisy.detectors[0].readout.noise_rate == 20_000.0
    assert noisy.provenance["augmentation"] == [{"add_noise_rate_hz": 20_000.0, "seed": 4}]
    # the signal hits are unchanged
    np.testing.assert_array_equal(np.sort(noisy.hits.t[~noisy.hits.is_noise]), np.sort(clean.hits.t))
    np.testing.assert_array_equal(noisy.photons.n_detected, clean.photons.n_detected)
    again = add_noise(clean, 20_000.0, rng=4)
    np.testing.assert_array_equal(again.hits.t, noisy.hits.t)
    with pytest.raises(ValueError, match="needs the hits"):
        add_noise(generate(5, config=preset("scatter"), seed=1, level="pulses", device="cpu"), 1.0)


def test_pulse_time_mean():
    first = generate(100, config=preset("full", medium="ice"), seed=5, device="cpu")
    mean = generate(100, config=preset("full", medium="ice", pulse_time="mean"), seed=5, device="cpu")
    check_conservation(mean)
    np.testing.assert_array_equal(first.pulses.counts(), mean.pulses.counts())
    # per event: sum of charge x mean time = sum of the hit times; the first times are earlier
    ev = mean.pulses.event_index()
    weighted = np.bincount(ev, weights=mean.pulses.charge * mean.pulses.t.astype(np.float64), minlength=100)
    hit_sum = np.bincount(mean.hits.event_index(), weights=mean.hits.t.astype(np.float64), minlength=100)
    np.testing.assert_allclose(weighted, hit_sum, rtol=1e-5)
    first_sum = np.bincount(first.pulses.event_index(), weights=first.pulses.charge * first.pulses.t.astype(float))
    assert np.all(first_sum <= weighted + 1e-3)
    assert np.any(first_sum < weighted - 1)
    rebuilt = pulses_from_hits(mean.hits, 3.5, mean_time=True)
    np.testing.assert_allclose(rebuilt.t, mean.pulses.t, rtol=1e-6)  # rebuilt from float32 hit times
