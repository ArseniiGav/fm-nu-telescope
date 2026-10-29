"""Detector families: ranges, presets, determinism, medium interpolation."""

import math

import numpy as np
import pytest

from nutel.toy import FAMILIES, MEDIA, Detector, DetectorFamily, family, preset, sample_detector, sample_detectors


def _ranges_hold(det: Detector, fam: DetectorFamily) -> None:
    g, m, r = det.geometry, det.medium, det.readout
    p = g.params
    assert fam.n_strings_range[0] <= g.n_strings <= fam.n_strings_range[1]
    assert fam.modules_per_string_range[0] <= p["modules_per_string"] <= fam.modules_per_string_range[1]
    assert fam.string_spacing_range[0] <= p["string_spacing"] <= fam.string_spacing_range[1] + 1e-3
    assert fam.module_spacing_range[0] <= p["module_spacing"] <= fam.module_spacing_range[1] + 1e-3
    assert fam.module_radius_range[0] - 1e-4 <= g.radius <= fam.module_radius_range[1] + 1e-4
    assert fam.absorption_length_range[0] - 1e-2 <= m.absorption_length <= fam.absorption_length_range[1] + 1e-2
    lo, hi = fam.scattering_length_eff_range
    assert lo - 1e-2 <= m.scattering_length_eff <= hi + 1e-2
    assert fam.efficiency_range[0] - 1e-4 <= r.efficiency <= fam.efficiency_range[1] + 1e-4
    assert fam.time_jitter_range[0] - 1e-4 <= r.time_jitter <= fam.time_jitter_range[1] + 1e-4
    assert fam.noise_rate_range[0] - 0.1 <= r.noise_rate <= fam.noise_rate_range[1] + 0.1


def test_default_family_ranges_and_layout_mix():
    fam = family("default")
    dets = sample_detectors(200, fam, seed=1)
    for det in dets:
        _ranges_hold(det, fam)
    layouts = {d.geometry.layout for d in dets}
    assert layouts == {"line", "core", "irregular"}
    assert len({d.geometry_id for d in dets}) == 200
    # medium between water and ice, n and g interpolated
    n = np.array([d.medium.refractive_index for d in dets])
    assert np.all((n >= 1.32 - 1e-9) & (n <= 1.35 + 1e-9))
    lam = np.array([d.medium.scattering_length_eff for d in dets])
    assert np.corrcoef(np.log(lam), n)[0, 1] > 0.99  # n follows lambda_eff (water-like = long scattering)


def test_tridentlike_family():
    fam = FAMILIES["tridentlike"]
    dets = sample_detectors(30, "tridentlike", seed=2)
    for det in dets:
        _ranges_hold(det, fam)
        assert det.geometry.layout == "irregular"
        assert det.medium.absorption_length == MEDIA["water"].absorption_length
        assert det.geometry.n_modules >= 16 * 16
    assert DetectorFamily.preset("tridentlike") == fam


def test_sampling_is_deterministic():
    a = sample_detectors(5, "default", seed=7)
    b = sample_detectors(5, "default", seed=7)
    c = sample_detectors(5, "default", seed=8)
    assert a == b
    assert [d.geometry_id for d in a] != [d.geometry_id for d in c]
    assert sample_detector(3) == sample_detector(np.random.default_rng(3))


def test_readout_window_comes_from_the_config():
    config = preset("full", window=(0.0, 2000.0), merge_window=5.0)
    det = sample_detector(1, "default", config)
    assert det.readout.window == (0.0, 2000.0)
    assert det.readout.merge_window == 5.0


def test_fixed_values_and_overrides():
    fam = family("default", n_strings_range=(5, 5), layout_weights=(1, 0, 0, 0), noise_rate_range=(0.0, 0.0))
    det = sample_detector(0, fam)
    assert det.geometry.n_strings == 5
    assert det.geometry.layout == "line"
    assert det.readout.noise_rate == 0.0
    ragged = sample_detector(0, family("default", layout_weights=(0, 0, 0, 1)))
    assert ragged.geometry.layout == "ragged"


def test_family_validation_and_description():
    with pytest.raises(ValueError, match="MIN <= MAX"):
        DetectorFamily(n_strings_range=(5, 3))
    with pytest.raises(ValueError, match="overlap"):
        DetectorFamily(module_radius_range=(0.3, 8.0))
    with pytest.raises(ValueError, match="unknown detector family"):
        family("hexagon")
    with pytest.raises(ValueError, match="unknown detector family"):
        sample_detector(0, "hexagon")
    text = family().describe()
    assert "absorption_length" in text
    assert "layout weights" in text
    assert DetectorFamily.from_dict(family().to_dict()) == family()
    assert math.isclose(sum(family().layout_weights), 1.0)


def test_sampled_medium_carries_the_sampled_noise_rate():
    for det in sample_detectors(5, "default", seed=9):
        assert det.readout.noise_rate == det.medium.background_rate
