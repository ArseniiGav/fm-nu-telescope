import subprocess
import sys

import numpy as np
import pytest

from synthetic import synthetic_data, two_detectors

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nutel.toy.plot import (  # noqa: E402
    EventView,
    animate_event,
    draw_family,
    draw_geometry,
    event_display,
    gallery,
    palette,
    photon_paths,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def data():
    return synthetic_data(8, seed=3, detectors=two_detectors())


def test_core_imports_without_matplotlib():
    code = (
        "import sys; sys.modules['matplotlib'] = None\n"
        "import nutel.toy.plot, nutel.toy.features, nutel.toy.reco, nutel.toy.io\n"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_palette_is_hex():
    assert all(len(v) == 7 and v.startswith("#") for v in palette.values())


def test_draw_geometry(data):
    _fig, ax = plt.subplots()
    circles = draw_geometry(ax, data.geometry(0))
    g = data.geometry(0)
    assert len(circles.get_offsets()) == g.n_modules
    x0, x1 = ax.get_xlim()
    assert x0 < g.x.min()
    assert x1 > g.x.max()
    assert ax.get_xlabel() == "x [m]"
    assert ax.get_ylabel() == "z [m]"


@pytest.mark.parametrize("i", range(4))
def test_event_display_every_type(data, i):
    fig, ax = plt.subplots()
    markers = event_display(ax, data, i, linefit=True, mark_noise=True)
    view = EventView.from_data(data, i)
    assert len(markers.get_offsets()) == len(view.modules_hit()["x"])
    title = ax.get_title()
    assert f"event {int(data.truth.event_id[i])}" in title
    assert ("track", "starting", "cascade", "atmospheric")[i] in title
    assert f"{len(view.x)} pulses" in title
    assert "detector" in title  # multi-geometry file
    labels = ax.get_legend_handles_labels()[1]
    assert "module" in labels
    assert ("true vertex" if i in (1, 2) else "true track") in labels
    assert fig.axes[-1].get_ylabel() == "first pulse time [ns]"  # the colour bar


def test_event_view_by_hand_and_options(tmp_path):
    view = EventView(
        module_x=np.array([0.0, 50.0, 100.0]),
        module_z=np.zeros(3),
        radius=0.5,
        x=np.array([0.0, 50.0, 50.0]),
        z=np.zeros(3),
        t=np.array([10.0, 30.0, 35.0]),
        charge=np.array([2.0, 1.0, 3.0]),
        n_noise=np.array([2.0, 0.0, 0.0]),
    )
    hit = view.modules_hit()
    np.testing.assert_array_equal(hit["charge"], [2.0, 4.0])
    np.testing.assert_array_equal(hit["t_first"], [10.0, 30.0])
    np.testing.assert_array_equal(hit["noise_only"], [True, False])
    assert view.title() == "3 pulses"
    fig, ax = plt.subplots()
    event_display(ax, view, color="charge", size="none", truth=False, legend=False)
    assert fig.axes[-1].get_ylabel() == "module charge [p.e.]"
    with pytest.raises(TypeError, match="index"):
        event_display(ax, synthetic_data(2), None)
    with pytest.raises(IndexError):
        EventView.from_data(synthetic_data(2), 5)
    empty = EventView(np.zeros(1), np.zeros(1), 0.5, *(np.zeros(0) for _ in range(4)))
    fig, ax = plt.subplots()
    event_display(ax, empty)


def test_event_display_hits_level(data):
    _fig, ax = plt.subplots()
    event_display(ax, data, 0, level="hits")
    assert f"{int(np.diff(data.hits.offset)[0])} hits" in ax.get_title()


def test_photon_paths(data):
    _fig, ax = plt.subplots()
    counts = photon_paths(ax, data, 0)
    assert counts == {"direct": 1, "scattered": 1, "absorbed": 1, "other": 1}
    assert len(ax.collections) >= 4  # modules + one collection per fate
    with pytest.raises(ValueError, match="no stored photon paths"):
        photon_paths(ax, data, 5)
    no_paths = synthetic_data(3, paths=0)
    with pytest.raises(ValueError, match="save_paths"):
        photon_paths(ax, no_paths, 0)


def test_gallery_and_family(data):
    fig = gallery(data, [0, 1, 2, 3, 4], ncols=3, titles=["a", "b", "c", "d", "e"])
    assert len(fig.axes) >= 6
    assert fig.axes[0].get_title().startswith("a\n")
    fig = draw_family(data.detectors, ncols=4)
    titles = [ax.get_title() for ax in fig.axes]
    assert any("detector 1" in t and "lambda" in t for t in titles)
    with pytest.raises(ValueError, match="no event"):
        gallery(data, [])
    with pytest.raises(ValueError, match="no detector"):
        draw_family([])


@pytest.mark.parametrize("i", [0, 1, 2])
def test_animate_event(data, tmp_path, i):
    from PIL import Image

    out = animate_event(data, i, tmp_path / "gif" / "a.gif", n_frames=4, hold_frames=1, dpi=40)
    assert out.exists()
    with Image.open(out) as im:
        assert im.n_frames >= 2


def test_event_display_mask_and_noise(data):
    p = data.pulses
    n0 = int(p.offset[1] - p.offset[0])
    keep_all = np.ones(len(p.t), dtype=bool)
    keep_all[int(p.offset[0])] = False  # remove the first pulse of event 0
    _fig, ax = plt.subplots()
    markers = event_display(ax, data, 0, mask=keep_all, linefit=True)
    removed = [c for c in ax.collections if c.get_label() == "removed by the cleaning"]
    assert len(removed) == 1
    assert len(removed[0].get_offsets()) == 1
    assert len(markers.get_offsets()) <= n0
    _fig, ax = plt.subplots()
    event_display(ax, data, 0, mask=np.ones(n0, dtype=bool), noise=True)  # a per-event mask
    with pytest.raises(ValueError, match="mask has"):
        event_display(ax, data, 0, mask=np.ones(n0 + 1, dtype=bool))
    view = EventView(
        module_x=np.array([0.0, 10.0, 20.0]),
        module_z=np.zeros(3),
        radius=0.5,
        x=np.array([0.0, 10.0, 20.0]),
        z=np.zeros(3),
        t=np.array([0.0, 50.0, 4000.0]),
        charge=np.array([1.0, 2.0, 1.0]),
        n_noise=np.array([0.0, 0.0, 1.0]),
    )
    _fig, ax = plt.subplots()
    markers = event_display(ax, view, noise=True)
    assert len(markers.get_offsets()) == 2
    assert markers.get_array().max() == 50.0  # the colour range ignores the noise pulse
    assert any(c.get_label() == "noise pulse" for c in ax.collections)


def test_gallery_panel_options_and_family_fill(data):
    fig = gallery(data, [0, 1], panel_options=[None, {"truth": False, "data": data}])
    assert len(fig.axes) >= 2
    fig = draw_family(data.detectors, fill=True)
    assert "noise" in fig.axes[0].get_title()
    _fig, ax = plt.subplots()
    circles = draw_geometry(ax, data.geometry(0), fill=True)
    assert circles.get_facecolors().shape[0] >= 1


def test_energy_in_tev_from_1_tev():
    from nutel.toy.plot import _format_energy

    assert _format_energy(999.0) == "999 GeV"
    assert _format_energy(1040.0) == "1.04 TeV"
