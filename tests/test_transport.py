"""Transport: the 2-D Henyey-Greenstein sampler, the segment-disc test, the world box, the cull grid, the fates."""

import math

import numpy as np
import pytest
import torch

from helpers import KS_LIMIT, N_SIGMA, assert_binomial, ks_distance
from nutel.toy import PointType
from nutel.toy.events import Photons
from nutel.toy.transport import ModuleGrid, exit_distance, first_module_hit, rotate, sample_hg_angle, transport

F64 = torch.float64


def hg_pdf(theta: np.ndarray, g: float) -> np.ndarray:
    return (1 - g**2) / (2 * np.pi * (1 + g**2 - 2 * g * np.cos(theta)))


@pytest.mark.parametrize("g", [0.0, 0.5, 0.8, 0.95])
def test_hg_sampler(g):
    theta = sample_hg_angle(200_000, g, torch.Generator().manual_seed(1)).numpy()
    assert np.all(np.abs(theta) <= np.pi)
    mean_cos = np.cos(theta).mean()
    sigma = np.cos(theta).std() / math.sqrt(len(theta))
    assert abs(mean_cos - g) < N_SIGMA * sigma
    grid = np.linspace(-np.pi, np.pi, 200_001)
    pdf = hg_pdf(grid, g)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    assert cdf[-1] == pytest.approx(1.0, rel=1e-6)  # normalisation
    assert ks_distance(theta[:20_000], lambda v: np.interp(v, grid, cdf)) < KS_LIMIT


def test_rotate():
    ux, uz = rotate(
        torch.tensor([1.0], dtype=F64), torch.tensor([0.0], dtype=F64), torch.tensor([math.pi / 2], dtype=F64)
    )
    assert ux.item() == pytest.approx(0.0, abs=1e-15)
    assert uz.item() == pytest.approx(1.0)


def test_segment_disc_intersection_against_fine_sampling():
    rng = np.random.default_rng(5)
    n, m, r = 3000, 12, 1.5
    modules = rng.uniform(-20, 20, (m, 2))
    x, z = rng.uniform(-25, 25, n), rng.uniform(-25, 25, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    ux, uz = np.cos(phi), np.sin(phi)
    length = rng.exponential(15.0, n)
    t = lambda a: torch.as_tensor(a, dtype=F64)  # noqa: E731
    dist, mod = first_module_hit(t(x), t(z), t(ux), t(uz), t(length), t(modules), r, block_elements=1000)
    dist, mod = dist.numpy(), mod.numpy()
    # brute force: points every 1 mm along the segment, first point inside a disc
    step = 1e-3
    n_hit = 0
    for i in range(0, n, 10):
        s = np.arange(0, length[i] + step, step)
        px, pz = x[i] + s * ux[i], z[i] + s * uz[i]
        d2 = (px[:, None] - modules[None, :, 0]) ** 2 + (pz[:, None] - modules[None, :, 1]) ** 2
        inside = d2 < r * r
        if inside.any():
            k = np.argmax(inside.any(axis=1))
            assert mod[i] == np.argmax(inside[k]) or abs(dist[i] - s[k]) < 2 * step
            assert abs(dist[i] - s[k]) < 2 * step
            n_hit += 1
        else:
            assert mod[i] == -1 or dist[i] > length[i] - 2 * step
    assert n_hit > 20


def test_segment_starting_inside_a_disc_hits_at_zero():
    t = lambda *a: torch.tensor(a, dtype=F64)  # noqa: E731
    dist, mod = first_module_hit(t(0.1), t(0.0), t(1.0), t(0.0), t(5.0), torch.tensor([[0.0, 0.0]], dtype=F64), 0.5)
    assert dist.item() == 0.0
    assert mod.item() == 0
    dist, mod = first_module_hit(t(2.0), t(0.0), t(1.0), t(0.0), t(5.0), torch.tensor([[0.0, 0.0]], dtype=F64), 0.5)
    assert mod.item() == -1  # the disc is behind


def test_exit_distance():
    t = lambda *a: torch.tensor(a, dtype=F64)  # noqa: E731
    box = (-10.0, 10.0, -5.0, 5.0)
    d = exit_distance(t(0.0, 0.0, 0.0, 20.0), t(0.0, 0.0, 0.0, 0.0), t(1.0, 0.0, -0.6, 1.0), t(0.0, 1.0, 0.8, 0.0), box)
    np.testing.assert_allclose(d.numpy(), [10.0, 5.0, 5.0 / 0.8, 0.0])


def test_module_grid_is_a_lower_bound():
    rng = np.random.default_rng(6)
    modules = torch.as_tensor(rng.uniform(-100, 100, (40, 2)), dtype=F64)
    box = (-150.0, 150.0, -120.0, 120.0)
    grid = ModuleGrid(modules, 0.5, box, max_cells=5000)
    pts = torch.as_tensor(np.stack([rng.uniform(-150, 150, 20000), rng.uniform(-120, 120, 20000)], 1), dtype=F64)
    true = torch.cdist(pts, modules).min(dim=1).values - 0.5
    bound = grid.lower_bound(pts[:, 0], pts[:, 1])
    assert torch.all(bound <= true + 1e-9)
    assert (true - bound).max() < 2 * grid.h


def _photons(n, x, z, angle, generator=None):
    angle = torch.as_tensor(angle, dtype=F64).expand(n).clone()
    return Photons(
        event=torch.zeros(n, dtype=torch.int64),
        source=torch.zeros(n, dtype=torch.int8),
        x=torch.full((n,), float(x), dtype=F64),
        z=torch.full((n,), float(z), dtype=F64),
        ux=torch.cos(angle),
        uz=torch.sin(angle),
        t=torch.zeros(n, dtype=F64),
    )


def _run(photons, **kw):
    args = {
        "modules": torch.tensor([[10.0, 0.0], [30.0, 0.0]], dtype=F64),
        "radius": 0.5,
        "absorption_length": math.inf,
        "scattering_length": math.inf,
        "g": 0.8,
        "speed_of_light": 0.2,
        "efficiency": 1.0,
        "world": (-50.0, 50.0, -50.0, 50.0),
        "max_steps": 200,
        "generator": torch.Generator().manual_seed(2),
    }
    args.update(kw)
    modules, radius = args.pop("modules"), args.pop("radius")
    return transport(photons, modules, radius, **args)


def test_straight_photons_hit_the_first_module_or_escape():
    res = _run(_photons(3, 0.0, 0.0, torch.tensor([0.0, math.pi, 0.0])))
    np.testing.assert_array_equal(res.fate.numpy(), [PointType.DETECT, PointType.ESCAPE, PointType.DETECT])
    np.testing.assert_array_equal(res.module.numpy(), [0, -1, 0])  # the first module shadows the second
    np.testing.assert_allclose(res.path_length.numpy(), [9.5, 50.0, 9.5])
    assert res.n_scatter.sum() == 0


def test_absorption_budget_and_efficiency():
    n = 50_000
    res = _run(_photons(n, 0.0, 0.0, 0.0), absorption_length=20.0, efficiency=0.6)
    fate = res.fate.numpy()
    p_reach = math.exp(-9.5 / 20.0)
    reached = np.isin(fate, [PointType.DETECT, PointType.ABSORB_MODULE]).sum()
    assert_binomial(reached, n, p_reach)
    assert_binomial(np.sum(fate == PointType.DETECT), reached, 0.6)
    absorbed = res.path_length.numpy()[fate == PointType.ABSORB]
    assert ks_distance(absorbed, lambda v: (1 - np.exp(-v / 20)) / (1 - p_reach)) < KS_LIMIT


def test_scattering_conserves_photons_and_records_paths():
    n = 2000
    photons = _photons(n, 0.0, 5.0, torch.rand(n, generator=torch.Generator().manual_seed(3), dtype=F64) * 6.3)
    record = torch.zeros(n, dtype=torch.bool)
    record[:20] = True
    res = _run(photons, absorption_length=30.0, scattering_length=3.0, max_steps=30, record=record)
    fate = res.fate.numpy()
    assert set(np.unique(fate)) <= {PointType.DETECT, PointType.ABSORB, PointType.ESCAPE, PointType.LOST}
    assert np.sum(fate == PointType.LOST) > 0
    assert res.n_scatter.max() == 30
    assert np.all(res.n_scatter.numpy()[fate != PointType.LOST] < 30)
    paths = res.paths
    photon = paths.photon.numpy()
    assert set(np.unique(photon)) == set(range(20))
    for k in range(20):
        kinds = paths.kind.numpy()[photon == k]
        assert kinds[0] == PointType.EMIT
        assert kinds[-1] == fate[k]
        assert np.all(kinds[1:-1] == PointType.SCATTER)
        assert len(kinds) - 2 == res.n_scatter[k].item()
        xs, zs = paths.x.numpy()[photon == k], paths.z.numpy()[photon == k]
        path = np.sum(np.hypot(np.diff(xs), np.diff(zs)))
        assert path == pytest.approx(res.path_length[k].item(), rel=1e-9)
        ts = paths.t.numpy()[photon == k]
        np.testing.assert_allclose(np.diff(ts), np.hypot(np.diff(xs), np.diff(zs)) / 0.2, rtol=1e-9, atol=1e-9)


def test_grid_does_not_change_the_result():
    n = 3000
    photons = _photons(n, 0.0, 5.0, torch.rand(n, generator=torch.Generator().manual_seed(4), dtype=F64) * 6.3)
    modules = torch.tensor([[10.0, 0.0], [30.0, 0.0], [-12.0, 8.0]], dtype=F64)
    kw = {"modules": modules, "absorption_length": 40.0, "scattering_length": 4.0}
    a = _run(photons, generator=torch.Generator().manual_seed(9), **kw)
    grid = ModuleGrid(modules, 0.5, (-50.0, 50.0, -50.0, 50.0))
    b = _run(photons, generator=torch.Generator().manual_seed(9), grid=grid, **kw)
    for name in ("fate", "module", "path_length", "n_scatter"):
        assert torch.equal(getattr(a, name), getattr(b, name))
