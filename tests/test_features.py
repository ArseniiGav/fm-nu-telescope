from types import SimpleNamespace

import numpy as np
import pytest

from nutel.toy.features import FEATURE_NAMES, features_from_arrays, image_from_arrays, image_grid
from nutel.toy.reco import unit_vector


def test_features_by_hand():
    # event 0: three pulses on two modules of one string, event 1: empty, event 2: one pulse
    offset = np.array([0, 3, 3, 4])
    x = np.array([0.0, 0.0, 0.0, 50.0])
    z = np.array([0.0, 20.0, 0.0, 10.0])
    t = np.array([100.0, 130.0, 160.0, 7.0])
    q = np.array([1.0, 3.0, 2.0, 4.0])
    f = features_from_arrays(offset, x, z, t, q, string=np.array([0, 0, 0, 1]))
    assert set(f) == set(FEATURE_NAMES)
    np.testing.assert_array_equal(f["n_pulses"], [3, 0, 1])
    np.testing.assert_array_equal(f["n_hits"], [6, 0, 4])
    np.testing.assert_array_equal(f["total_charge"], [6.0, 0.0, 4.0])
    np.testing.assert_array_equal(f["n_modules"], [2, 0, 1])
    np.testing.assert_array_equal(f["n_strings"], [1, 0, 1])
    np.testing.assert_allclose(f["t_first"], [100.0, np.nan, 7.0])
    np.testing.assert_allclose(f["duration"], [60.0, np.nan, 0.0])
    np.testing.assert_allclose(f["z_cog"], [10.0, np.nan, 10.0])
    np.testing.assert_allclose(f["z_rms"], [10.0, np.nan, 0.0])
    np.testing.assert_allclose(f["x_rms"], [0.0, np.nan, 0.0])
    assert np.isfinite(f["linefit_angle"][0])
    assert np.isnan(f["linefit_angle"][1:]).all()


def test_features_linefit_columns_follow_a_moving_source():
    rng = np.random.default_rng(0)
    angles = rng.uniform(-np.pi, np.pi, 20)
    offset = np.arange(21) * 10
    t = rng.uniform(0, 500, 200)
    d = unit_vector(np.repeat(angles, 10))
    x, z = 0.3 * t * d[:, 0], 0.3 * t * d[:, 1]
    f = features_from_arrays(offset, x, z, t)
    np.testing.assert_allclose(f["linefit_angle"], angles, atol=1e-9)
    np.testing.assert_allclose(f["linefit_speed"], 0.3)
    np.testing.assert_array_equal(f["n_hits"], 10)


def test_features_input_checks():
    with pytest.raises(ValueError, match="length"):
        features_from_arrays([0, 2], [0.0], [0.0, 1.0], [0.0, 1.0])


def test_image_from_arrays():
    offset = np.array([0, 3, 4])
    s_idx = np.array([0, 0, 1, 1])
    m_idx = np.array([2, 2, 0, 1])
    t = np.array([50.0, 40.0, 70.0, 5.0])
    q = np.array([1.0, 3.0, 2.0, 1.0])
    img = image_from_arrays(
        offset, s_idx, m_idx, t, q, (2, 3), channels=("t_first", "charge", "n_pulses", "t_mean", "t_last")
    )
    assert img.shape == (2, 5, 2, 3)
    assert img.dtype == np.float32
    assert img[0, 0, 0, 2] == 40.0
    assert img[0, 1, 0, 2] == 4.0
    assert img[0, 2, 0, 2] == 2.0
    assert img[0, 3, 0, 2] == pytest.approx((50 + 3 * 40) / 4)
    assert img[0, 4, 0, 2] == 50.0
    assert img[0, 0, 1, 0] == 70.0
    assert np.isnan(img[0, 0, 0, 0])  # empty module: no time
    assert img[0, 1, 0, 0] == 0.0
    assert img[1, 0, 1, 1] == 5.0
    assert np.nansum(img[1, 1]) == 1.0
    rel = image_from_arrays(offset, s_idx, m_idx, t, q, (2, 3), relative_time=True, empty_time=-1.0)
    assert rel[0, 0, 0, 2] == 0.0
    assert rel[0, 0, 1, 0] == 30.0
    assert rel[0, 0, 1, 1] == -1.0
    mask = np.array([[[1, 1, 1], [1, 1, 0]]], dtype=bool)
    with_mask = image_from_arrays(offset, s_idx, m_idx, t, q, (2, 3), channels=("charge",), module_mask=mask)
    assert with_mask.shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(with_mask[:, -1], np.broadcast_to(mask, (2, 2, 3)))
    with pytest.raises(ValueError, match="channels"):
        image_from_arrays(offset, s_idx, m_idx, t, q, (2, 3), channels=("energy",))
    with pytest.raises(ValueError, match="grid"):
        image_from_arrays(offset, s_idx, m_idx, t, q, (1, 3))


def test_image_grid_of_ragged_and_several_geometries():
    a = SimpleNamespace(string=np.array([3, 3, 3, 7, 7]), module=np.array([0, 1, 2, 0, 1]))
    b = SimpleNamespace(string=np.array([0, 1, 2, 2]), module=np.array([0, 0, 0, 1]))
    shape, ids, mask = image_grid([a, b])
    assert shape == (3, 3)
    np.testing.assert_array_equal(ids[0], [3, 7])
    np.testing.assert_array_equal(mask[0], [[1, 1, 1], [1, 1, 0], [0, 0, 0]])
    np.testing.assert_array_equal(mask[1], [[1, 0, 0], [1, 0, 0], [1, 1, 0]])
    with pytest.raises(ValueError, match="no geometry"):
        image_grid([])


# --------------------------------------------------------------------------------------------- adapters for SimData


def test_event_features_of_a_data_set():
    from nutel.toy.features import event_features
    from synthetic import synthetic_data, two_detectors

    data = synthetic_data(9, seed=7, detectors=two_detectors())
    table = event_features(data)
    assert list(table)[:2] == ["event_id", "geometry_index"]
    assert all(len(v) == 9 for v in table.values())
    np.testing.assert_array_equal(table["n_hits"], data.photons.n_hits)
    np.testing.assert_array_equal(table["n_pulses"], data.pulses.counts())
    np.testing.assert_array_equal(table["total_charge"], np.bincount(data.pulses.event_index(), data.pulses.charge))
    np.testing.assert_array_equal(table["geometry_index"], data.truth.geometry_index)
    hits = event_features(data, level="hits")
    np.testing.assert_array_equal(hits["n_pulses"], data.hits.counts())
    np.testing.assert_array_equal(hits["n_modules"], table["n_modules"])
    pandas = pytest.importorskip("pandas")
    assert pandas.DataFrame(table).shape == (9, 2 + len(FEATURE_NAMES))


def test_image_of_a_multi_geometry_data_set():
    from nutel.toy.features import image
    from synthetic import synthetic_data, two_detectors

    data = synthetic_data(9, seed=8, detectors=two_detectors())
    shape, _, grid_mask = image_grid(data.geometries)
    img = image(data, ("charge", "t_first"), mask=True)
    assert img.shape == (9, 3, *shape)
    np.testing.assert_allclose(img[:, 0].sum(axis=(1, 2)), np.bincount(data.pulses.event_index(), data.pulses.charge))
    for i in range(9):
        g = data.truth.geometry_index[i]
        np.testing.assert_array_equal(img[i, 2], grid_mask[g])
        assert np.all(img[i, 0][~grid_mask[g]] == 0)  # no charge in the padding
        first = data.pulses.t[data.pulses.offset[i] : data.pulses.offset[i + 1]].min()
        assert np.nanmin(img[i, 1]) == pytest.approx(first)
    rel = image(data, ("t_first",), relative_time=True)
    assert np.nanmin(rel[:, 0], axis=(1, 2)) == pytest.approx(np.zeros(9))
