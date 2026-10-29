"""Configuration: presets, overrides of every parameter, validation, JSON round trip, descriptions."""

import dataclasses
import json
import math

import numpy as np
import pytest

from nutel.toy import (
    MEDIA,
    PRESETS,
    Geometry,
    Medium,
    Readout,
    SimConfig,
    config_fields,
    describe_presets,
    matching_preset,
    medium,
    medium_name,
    preset,
)


def _changed(f: dataclasses.Field, part_name: str):
    """A valid value different from the default of a configuration field."""
    meta, default = f.metadata, f.default
    if "choices" in meta:
        return next(c for c in meta["choices"] if c != default)
    if isinstance(default, bool):
        return not default
    special = {
        "fixed_angle": 30.0,
        "fixed_vertex": (1.0, 2.0),
        "fixed_energy": 50.0,
        "fixed_t0": 1500.0,
        "fixed_inelasticity": 0.3,
        "n_group": 1.38,
        "noise_rate": 1234.0,
        "event_type": "cascade",
        "type_fractions": (1.0, 0.0, 0.0, 0.0),
        "inelasticity_range": (0.1, 0.5),
        "atmospheric_cos_zenith_range": (0.2, 0.9),
        "window": (-100.0, 4000.0),
        "t0_range": (0.0, 500.0),
        "stochastic_v_min": 0.01,
        "g": 0.5,
        "n_core": 2,
        "min_string_distance": 30.0,
        "ragged_min_modules": 3,
        "core_spacing": 25.0,
        "module_radius": 0.3,
        "efficiency": 0.5,
        "refractive_index": 1.4,
    }
    if f.name in special:
        return special[f.name]
    if f.metadata.get("nargs") == 2:
        lo, hi = default
        return (lo * 1.5, hi * 1.5) if lo > 0 else (lo, hi * 1.5)
    if isinstance(default, int):
        return default + 1
    return default * 1.5 if default else 0.5


@pytest.mark.parametrize(("part", "f"), list(config_fields()), ids=[f.name for _, f in config_fields()])
def test_every_parameter_can_be_overridden(part, f):
    value = _changed(f, part)
    config = SimConfig().replace(**{f.name: value})
    got = getattr(getattr(config, part), f.name)
    assert got == (tuple(value) if isinstance(value, tuple) else value)
    assert got != f.default
    # every other parameter keeps its value
    for other_part, g in config_fields():
        if g.name != f.name:
            assert getattr(getattr(config, other_part), g.name) == getattr(getattr(SimConfig(), other_part), g.name)
    assert SimConfig.from_json(config.to_json()) == config
    assert f.metadata["help"]


def test_parameter_names_are_unique_and_documented():
    names = [f.name for _, f in config_fields()]
    assert len(names) == len(set(names))
    for _, f in config_fields():
        assert isinstance(f.metadata["help"], str)
        assert f.metadata["help"]


def test_presets_form_the_ladder():
    assert list(PRESETS) == ["straight", "absorb", "scatter", "full"]
    flags = [(c.physics.absorption, c.physics.scattering, c.physics.stochastic) for c in PRESETS.values()]
    assert flags == [(False, False, False), (True, False, False), (True, True, False), (True, True, True)]
    for name in ("straight", "absorb", "scatter"):
        r = PRESETS[name].readout
        assert (r.time_jitter, r.efficiency, r.noise_rate) == (0.0, 1.0, 0.0)
    r = PRESETS["full"].readout
    assert r.time_jitter == 1.0
    assert r.efficiency == 1.0
    assert r.noise_rate is None
    assert preset() == SimConfig() == PRESETS["full"]
    assert preset("full").medium == MEDIA["water"]


def test_preset_with_medium_and_overrides():
    config = preset("scatter", medium="ice", noise_rate=20e3, absorption_length=40.0)
    assert config.physics.scattering
    assert not config.physics.stochastic
    assert config.medium == medium("ice", absorption_length=40.0)
    assert config.readout.noise_rate == 20e3
    assert matching_preset(preset("absorb", medium="ice")) == "absorb"
    assert matching_preset(config) is None
    assert medium_name(config.medium) == "custom"
    assert medium_name(preset(medium="ice").medium) == "ice"
    assert preset("full", medium=MEDIA["ice"]).medium == MEDIA["ice"]
    with pytest.raises(ValueError, match="unknown preset"):
        preset("nope")
    with pytest.raises(ValueError, match="unknown medium"):
        preset("full", medium="oil")


def test_noise_rate_follows_the_medium_unless_set():
    water, ice = preset("full", medium="water"), preset("full", medium="ice")
    assert water.detector().readout.noise_rate == 50_000.0
    assert ice.detector().readout.noise_rate == 500.0
    assert ice.replace(noise_rate=7.0).detector().readout.noise_rate == 7.0
    assert preset("scatter", medium="water").detector().readout.noise_rate == 0.0
    with pytest.raises(ValueError, match="not set"):
        _ = Readout().mean_noise_hits
    assert water.effective_readout().mean_noise_hits == pytest.approx(50_000 * 5000e-9)


def test_media_presets_and_derived_quantities():
    w, i = MEDIA["water"], MEDIA["ice"]
    assert (w.refractive_index, w.absorption_length, w.scattering_length_eff, w.g) == (1.35, 60.0, 300.0, 0.8)
    assert (i.refractive_index, i.absorption_length, i.scattering_length_eff, i.g) == (1.32, 100.0, 25.0, 0.8)
    assert w.scattering_length == pytest.approx(60.0)
    assert i.scattering_length == pytest.approx(5.0)
    assert math.cos(w.cherenkov_angle) == pytest.approx(1 / 1.35)
    assert w.speed_of_light == pytest.approx(0.299792458 / 1.35)
    assert w.n == w.refractive_index == w.group_index
    m = Medium(n_group=1.38)
    assert m.speed_of_light == pytest.approx(0.299792458 / 1.38)
    assert math.cos(m.cherenkov_angle) == pytest.approx(1 / 1.35)


def test_switched_off_processes_have_infinite_lengths():
    config = preset("straight")
    eff = config.effective_medium()
    assert math.isinf(eff.absorption_length)
    assert math.isinf(eff.scattering_length_eff)
    assert config.medium.absorption_length == 60.0  # the medium itself keeps its values
    eff = preset("absorb").effective_medium()
    assert eff.absorption_length == 60.0
    assert math.isinf(eff.scattering_length_eff)


def test_infinite_lengths_and_json():
    config = SimConfig().replace(absorption_length=math.inf)
    assert math.isinf(config.medium.absorption_length)
    assert SimConfig.from_json(config.to_json()) == config


def test_json_round_trip_with_custom_geometry():
    g = Geometry.from_table([0, 0, 1], [0, 1, 0], [0.0, 0.0, 30.0], [0.0, 20.0, 5.0], radius=0.4)
    config = preset("absorb", medium="ice", custom_geometry=g, layout="core")
    back = SimConfig.from_json(config.to_json(indent=2))
    assert back == config
    assert back.geometry() == g
    assert json.loads(config.to_json())["custom_geometry"]["radius"] == 0.4


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"absorption_length": -1.0}, "absorption_length"),
        ({"g": 1.0}, "g must be"),
        ({"refractive_index": 0.9}, "refractive_index"),
        ({"efficiency": 1.5}, "efficiency"),
        ({"window": (10.0, 5.0)}, "window"),
        ({"module_radius": 15.0}, "overlap"),
        ({"n_core": 20}, "n_core"),
        ({"track_energy_range": (10.0, 1.0)}, "track_energy_range"),
        ({"type_fractions": (0.0, 0.0, 0.0, 0.0)}, "type_fractions"),
        ({"inelasticity_range": (0.5, 1.5)}, "inelasticity_range"),
        ({"stochastic_v_min": 1.0}, "stochastic_v_min"),
        ({"max_steps": 0}, "max_steps"),
        ({"vertex_margin": 200.0}, "vertex_margin"),
        ({"fixed_energy": -1.0}, "fixed_energy"),
        ({"absorption_length": math.nan}, "finite"),
    ],
)
def test_invalid_values_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        SimConfig().replace(**overrides)


def test_type_errors_and_unknown_names():
    with pytest.raises(TypeError, match="Did you mean 'absorption_length'"):
        SimConfig().replace(absorbtion_length=3.0)
    with pytest.raises(TypeError, match="integer"):
        SimConfig().replace(n_strings=2.5)
    with pytest.raises(TypeError, match="True or False"):
        SimConfig().replace(scattering=1)
    with pytest.raises(ValueError, match="one of"):
        SimConfig().replace(layout="hexagon")
    with pytest.raises(ValueError, match="sequence of 2"):
        SimConfig().replace(window=5000.0)
    with pytest.raises(TypeError):
        SimConfig(medium="water")  # type: ignore[arg-type]


def test_numpy_values_are_converted():
    config = SimConfig().replace(n_strings=np.int64(6), absorption_length=np.float32(50.0), window=np.array([0, 10]))
    assert type(config.layout.n_strings) is int
    assert type(config.medium.absorption_length) is float
    assert config.readout.window == (0.0, 10.0)
    json.dumps(config.to_dict())


def test_type_probabilities():
    assert preset().prior.type_probabilities == pytest.approx((0.4, 0.2, 0.2, 0.2))
    assert preset(event_type="starting").prior.type_probabilities == (0.0, 1.0, 0.0, 0.0)
    assert preset(type_fractions=(2, 2, 0, 0)).prior.type_probabilities == (0.5, 0.5, 0.0, 0.0)


def test_descriptions_mention_the_key_values():
    text = preset("scatter", medium="ice").describe()
    assert "lambda_eff = 25 m" in text
    assert "stochastic losses off" in text
    assert "50000 Hz (from the medium)" in SimConfig().describe()
    table = describe_presets()
    for name in ("straight", "absorb", "scatter", "full", "water", "ice"):
        assert name in table
    assert "custom table" in SimConfig(custom_geometry=Geometry.from_table([0], [0], [0.0], [0.0])).describe()


def test_from_dict_rejects_unknown_entries():
    with pytest.raises(ValueError, match="sections"):
        SimConfig.from_dict({"optics": {}})
    with pytest.raises(ValueError, match="unknown parameters"):
        SimConfig.from_dict({"medium": {"colour": "blue"}})
    assert SimConfig.from_dict({}) == SimConfig()
