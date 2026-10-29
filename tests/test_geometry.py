"""Geometry: layouts, geometry ids, custom tables, serialisation."""

import numpy as np
import pytest

from nutel.toy import Geometry, Layout, SimConfig, make_geometry


def test_default_line_layout():
    g = make_geometry()
    assert (g.layout, g.n_strings, g.n_modules, g.radius) == ("line", 8, 96, 0.5)
    assert g.bounds == (-175.0, 175.0, -110.0, 110.0)
    assert g.size == (350.0, 220.0)
    assert g.center == (0.0, 0.0)
    assert g.half_diagonal == pytest.approx(0.5 * np.hypot(350, 220))
    np.testing.assert_array_equal(g.modules_per_string, np.full(8, 12))
    xs = np.unique(g.x)
    np.testing.assert_allclose(np.diff(xs), 50.0)
    one = g.string == 3
    np.testing.assert_allclose(np.diff(g.z[one]), 20.0)
    assert g.module[one][np.argmin(g.z[one])] == 0  # module 0 is the deepest
    assert g.positions.shape == (96, 2)
    assert g.positions.dtype == np.float32
    assert g.x.dtype == np.float32
    assert g.string.dtype == np.int32


def test_layout_parameters_are_respected():
    g = make_geometry(n_strings=5, modules_per_string=7, string_spacing=30.0, module_spacing=10.0, module_radius=0.2)
    assert (g.n_strings, g.n_modules, g.radius) == (5, 35, 0.2)
    assert g.size == (120.0, 60.0)
    assert SimConfig().replace(n_strings=5).geometry().n_strings == 5


def test_core_layout():
    g = make_geometry(layout="core", n_strings=8, n_core=4, core_spacing=20.0, string_spacing=50.0)
    xs = np.unique(g.x)
    assert len(xs) == 8
    gaps = np.diff(xs)
    np.testing.assert_allclose(sorted(gaps), [20, 20, 20, 50, 50, 50, 50])
    np.testing.assert_allclose(gaps[2:5], 20.0)  # the core is in the middle
    assert g.params["n_core"] == 4


def test_irregular_layout():
    for seed in range(20):
        g = make_geometry(layout="irregular", layout_seed=seed, min_string_distance=25.0)
        xs = np.unique(g.x)
        assert len(xs) == 8
        assert np.diff(xs).min() >= 25.0 - 1e-4
        assert xs.max() - xs.min() == pytest.approx(350.0, abs=1e-3)
    a = make_geometry(layout="irregular", layout_seed=1)
    b = make_geometry(layout="irregular", layout_seed=1)
    c = make_geometry(layout="irregular", layout_seed=2)
    assert a == b
    assert a != c


def test_ragged_layout():
    g = make_geometry(layout="ragged", ragged_min_modules=4, ragged_max_offset=15.0, layout_seed=4)
    counts = g.modules_per_string
    assert counts.min() >= 4
    assert counts.max() <= 12
    assert len(np.unique(counts)) > 1
    for s in g.strings:
        z = g.z[g.string == s]
        np.testing.assert_allclose(np.diff(np.sort(z)), 20.0, atol=1e-4)
    assert g.center == pytest.approx((0.0, 0.0), abs=1e-4)
    assert "per string" in g.describe()


def test_custom_table_and_module_index():
    g = Geometry.from_table([0, 0, 2], [0, 1, 0], [0.0, 0.0, 30.0], [-10.0, 10.0, 0.0], radius=0.3)
    assert g.layout == "custom"
    assert g.n_strings == 2
    assert g.n_modules == 3
    np.testing.assert_array_equal(g.module_index([2, 0, 5], [0, 1, 0]), [2, 1, -1])
    assert Geometry.from_dict(g.to_dict()) == g
    assert SimConfig(custom_geometry=g).geometry() is g


def test_geometry_id_is_stable():
    g = make_geometry()
    assert len(g.geometry_id) == 12
    assert int(g.geometry_id, 16) >= 0
    assert g.geometry_id == "a1f8a22bd674"  # changes only if the default layout or the hash definition changes
    shifted = Geometry.from_table(g.string, g.module, g.x + 0.001, g.z, radius=g.radius)
    assert shifted.geometry_id == g.geometry_id  # rounded to cm
    moved = Geometry.from_table(g.string, g.module, g.x + 0.2, g.z, radius=g.radius)
    assert moved.geometry_id != g.geometry_id
    other_radius = Geometry.from_table(g.string, g.module, g.x, g.z, radius=0.4)
    assert other_radius.geometry_id != g.geometry_id
    assert hash(g) == hash(Geometry.from_dict(g.to_dict()))


def test_invalid_tables():
    with pytest.raises(ValueError, match="unique"):
        Geometry.from_table([0, 0], [1, 1], [0.0, 10.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="overlap"):
        Geometry.from_table([0, 1], [0, 0], [0.0, 0.5], [0.0, 0.0], radius=0.5)
    with pytest.raises(ValueError, match="same length"):
        Geometry.from_table([0], [0, 1], [0.0], [0.0])
    with pytest.raises(ValueError, match="positive"):
        Geometry.from_table([0], [0], [0.0], [0.0], radius=0.0)
    with pytest.raises(ValueError, match="finite"):
        Geometry.from_table([0], [0], [np.nan], [0.0])


def test_arrays_are_read_only():
    g = make_geometry()
    with pytest.raises(ValueError, match="read-only"):
        g.x[0] = 1.0


def test_layout_validation():
    with pytest.raises(ValueError, match="min_string_distance"):
        Layout(min_string_distance=60.0)
    with pytest.raises(ValueError, match="ragged_min_modules"):
        Layout(ragged_min_modules=13)
