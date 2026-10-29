import json

import h5py
import numpy as np
import pytest

from nutel.toy.io import FORMAT_VERSION, describe, load, merge, save
from synthetic import synthetic_data, two_detectors


def assert_same(a, b, groups=("truth", "photons", "pulses", "hits", "bursts", "paths")):
    for group in groups:
        ca, cb = getattr(a, group), getattr(b, group)
        assert (ca is None) == (cb is None), group
        if ca is None:
            continue
        for name, values in ca.arrays().items():
            other = cb.arrays()[name]
            assert values.dtype == other.dtype, f"{group}/{name}"
            np.testing.assert_array_equal(values, other, err_msg=f"{group}/{name}")


def test_round_trip(tmp_path):
    data = synthetic_data(8, seed=1)
    path = save(data, tmp_path / "sub" / "a.h5")
    assert path.exists()
    assert not (tmp_path / "sub" / "a.h5.part").exists()
    back = load(path)
    assert_same(data, back)
    assert back.config == data.config
    assert back.provenance == data.provenance
    assert [d.geometry.geometry_id for d in back.detectors] == [d.geometry.geometry_id for d in data.detectors]
    assert back.detectors == data.detectors
    with h5py.File(path) as f:
        assert f.attrs["format_version"] == FORMAT_VERSION
        assert f.attrs["n_geometries"] == 1
        assert f["pulses/t"].attrs["units"] == "ns"
        assert f["truth/energy"].attrs["units"] == "GeV"
        assert f["truth/s_start"].attrs["units"] == "m"
        assert "NaN" in f["truth/s_end"].attrs["description"]
        assert f["geometries/geometry_id"].dtype == np.dtype("S12")
        assert json.loads(f["paths"].attrs["point_types"])["DETECT"] == 2
    with pytest.raises(FileExistsError):
        save(data, path, overwrite=False)


def test_level_selection(tmp_path):
    data = synthetic_data(5, seed=2)
    path = save(data, tmp_path / "a.h5")
    assert load(path, level="pulses").hits is None
    assert load(path, level="hits").hits is not None
    small = load(path, level="pulses", bursts=False, paths=False)
    assert small.bursts is None
    assert small.paths is None
    pulses_only = save(synthetic_data(5, seed=2, hits=False), tmp_path / "b.h5")
    assert load(pulses_only).hits is None
    with pytest.raises(ValueError, match="no hit level"):
        load(pulses_only, level="hits")
    with pytest.raises(ValueError, match="level"):
        load(path, level="photons")  # type: ignore[arg-type]


def test_multi_geometry_round_trip(tmp_path):
    data = synthetic_data(9, seed=3, detectors=two_detectors())
    back = load(save(data, tmp_path / "multi.h5"))
    assert_same(data, back)
    assert back.n_geometries == 2
    assert back.detectors[1].medium.n == pytest.approx(1.32)
    assert back.detectors[1].readout.noise_rate == 0.0
    assert back.geometry(1).layout == "ragged"
    for i in range(len(back)):
        g = back.geometry(i)
        ev = back.event(i)
        assert np.isin(g.module_index(ev.pulses.string, ev.pulses.module), np.arange(g.n_modules)).all()


def test_merge_single_and_multi_geometry(tmp_path):
    a = save(synthetic_data(6, seed=1), tmp_path / "a.h5")
    b = save(synthetic_data(5, seed=2, first_event_id=6), tmp_path / "b.h5")
    c = save(synthetic_data(7, seed=3, detectors=two_detectors(), first_event_id=100), tmp_path / "c.h5")
    out = merge([a, b, c], tmp_path / "abc.h5")
    merged = load(out)
    parts = [load(a), load(b), load(c)]
    assert len(merged) == 18
    assert merged.n_geometries == 2  # the water line detector of a, b and c is stored once
    np.testing.assert_array_equal(merged.truth.event_id, np.concatenate([p.truth.event_id for p in parts]))
    for k, part in enumerate(parts):
        start = sum(len(p) for p in parts[:k])
        for i in range(len(part)):
            assert merged.detector(start + i) == part.detector(i)
            np.testing.assert_array_equal(merged.event(start + i).pulses.t, part.event(i).pulses.t)
            np.testing.assert_array_equal(merged.event(start + i).hits.t, part.event(i).hits.t)
    assert len(merged.paths) == sum(len(p.paths) for p in parts)
    np.testing.assert_array_equal(merged.paths.x, np.concatenate([p.paths.x for p in parts]))
    assert [s["file"] for s in merged.provenance["merged_from"]] == ["a.h5", "b.h5", "c.h5"]


def test_merge_checks_and_renumbering(tmp_path):
    a = save(synthetic_data(4, seed=1), tmp_path / "a.h5")
    same_ids = save(synthetic_data(4, seed=2), tmp_path / "b.h5")
    with pytest.raises(ValueError, match="event ids"):
        merge([a, same_ids], tmp_path / "x.h5")
    renumbered = load(merge([a, same_ids], tmp_path / "ab.h5", renumber=True))
    np.testing.assert_array_equal(renumbered.truth.event_id, np.arange(8))
    assert set(renumbered.paths.event_id) <= set(range(8))
    np.testing.assert_array_equal(renumbered.paths.event_id[: len(load(a).paths)], load(a).paths.event_id)
    same_seed = save(synthetic_data(4, seed=1, first_event_id=10), tmp_path / "c.h5")
    with pytest.raises(ValueError, match="seed"):
        merge([a, same_seed], tmp_path / "x.h5")
    no_hits = save(synthetic_data(4, seed=5, hits=False, first_event_id=20), tmp_path / "d.h5")
    with pytest.raises(ValueError, match="groups"):
        merge([a, no_hits], tmp_path / "x.h5")
    other = synthetic_data(4, seed=6, first_event_id=30)
    other.config = other.config.replace(track_yield=7.0)
    with pytest.raises(ValueError, match=r"emission\.track_yield"):
        merge([a, save(other, tmp_path / "e.h5")], tmp_path / "x.h5")
    with pytest.raises(ValueError, match="also an input"):
        merge([a, same_ids], a)
    with pytest.raises(ValueError, match="no input"):
        merge([], tmp_path / "x.h5")


def test_describe_and_bad_files(tmp_path):
    path = save(synthetic_data(9, seed=4, detectors=two_detectors()), tmp_path / "a.h5")
    text = describe(path)
    assert "9 events, 2 detector(s)" in text
    assert "/pulses/t" in text
    assert "ragged" in text
    assert "track 3, starting 2, cascade 2, atmospheric 2" in text
    not_hdf5 = tmp_path / "x.h5"
    not_hdf5.write_text("hello")
    with pytest.raises(ValueError, match="not a readable HDF5"):
        load(not_hdf5)
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f["x"] = 1
    with pytest.raises(ValueError, match="not a nutel"):
        load(other)


def test_file_without_a_truth_field_is_refused(tmp_path):
    path = save(synthetic_data(3, seed=5), tmp_path / "old.h5")
    with h5py.File(path, "a") as f:
        del f["truth/s_start"]
    with pytest.raises(ValueError, match="no s_start; the file was written by an older version"):
        load(path)
