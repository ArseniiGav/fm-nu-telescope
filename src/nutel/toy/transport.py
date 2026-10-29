"""Photon transport in the two-dimensional medium (torch, vectorised over all photons of a chunk of events).

Each photon carries its position ``p``, direction ``u``, emission time, path length, number of scatterings and an
absorption budget ``L_a ~ Exp(lambda_a)`` drawn at emission (the total path it can travel). One pass of the loop in
:func:`transport` moves every photon still in flight along one straight segment::

    1. step       ell ~ Exp(lambda_s) (inf without scattering); the segment is [p, p + min(ell, L_a, s_exit) u],
                  s_exit = distance to the border of the world box
    2. modules    the FIRST module disc the segment enters (exact segment-disc intersection, all photons x all modules
                  of the geometry): the photon stops on the disc (modules are opaque); it is detected with probability
                  epsilon (efficiency), else absorbed by the module
    3. otherwise  absorbed if the budget ends on the segment; escaped if the segment reaches the world border;
                  else scattered: the direction turns by theta ~ 2-D Henyey-Greenstein, the budget shrinks by ell
    4. end        photons still in flight after max_steps steps are lost

A grid of lower bounds of the distance to the nearest module (:class:`ModuleGrid`) skips the exact module test for
photons whose segment cannot reach any module; this changes nothing in the result, only the speed.

Every random number comes from the ``generator`` argument (a :class:`torch.Generator` on the device of the
computation), so the result is reproducible for a given seed, device type and chunk on the same hardware and PyTorch
build (see :func:`warm_up_cpu_kernels` for a CPU detail).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .data import PointType
from .events import Photons

__all__ = [
    "ModuleGrid",
    "PathPoints",
    "TransportResult",
    "exit_distance",
    "first_module_hit",
    "rotate",
    "sample_hg_angle",
    "transport",
    "warm_up_cpu_kernels",
]

#: Number of photon x module pairs tested at once in the module test (sets the memory use).
BLOCK_ELEMENTS = 2**21

_cpu_kernels_warm = False


def warm_up_cpu_kernels() -> None:
    """Run the vectorised CPU math kernels once, single-threaded, before the first multi-threaded transport.

    In a fresh process the first multi-threaded call of a vectorised math kernel (``sin``, ``cos``, ``atan``, ``log``,
    ...) can compute one thread's block with a different code path than the others while the vector math library
    initialises (seen with the MKL-backed CPU builds of PyTorch). That can flip the fate of a photon at a module edge
    and, through the number of later random draws, change the whole realisation compared with a process that has used
    the kernels before. One small single-threaded call of each kernel beforehand removes the race. :func:`transport`
    calls this function for CPU tensors; it does its work once per process.
    """
    global _cpu_kernels_warm
    if _cpu_kernels_warm:
        return
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for dtype in (torch.float32, torch.float64):
            x = torch.linspace(0.1, 0.9, 64, dtype=dtype)
            for fn in (torch.sin, torch.cos, torch.tan, torch.atan, torch.sqrt, torch.log, torch.exp, torch.acos):
                fn(x)
            torch.atan2(x, x)
            torch.hypot(x, x)
            torch.log1p(-x)
    finally:
        torch.set_num_threads(threads)
    _cpu_kernels_warm = True


# ----------------------------------------------------------------------------------------------------------------------
# Geometry helpers


def rotate(ux: torch.Tensor, uz: torch.Tensor, angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the vectors ``(ux, uz)`` counter-clockwise by ``angle`` [rad]."""
    cos, sin = torch.cos(angle), torch.sin(angle)
    return ux * cos - uz * sin, ux * sin + uz * cos


def sample_hg_angle(n: int, g: float, generator: torch.Generator) -> torch.Tensor:
    """Scattering angles of the two-dimensional Henyey-Greenstein phase function.

    ``p(theta) = (1 - g^2) / (2 pi (1 + g^2 - 2 g cos theta))`` on ``(-pi, pi]`` has ``<cos theta> = g``; inverse
    transform sampling gives ``theta = 2 atan((1 - g) / (1 + g) tan(pi (u - 1/2)))`` with ``u ~ U(0, 1)``.

    Returns
    -------
    torch.Tensor
        ``n`` angles [rad] (float64) on the generator's device.
    """
    u = torch.rand(n, generator=generator, device=generator.device, dtype=torch.float64)
    return 2.0 * torch.atan((1.0 - g) / (1.0 + g) * torch.tan(math.pi * (u - 0.5)))


def exit_distance(
    x: torch.Tensor, z: torch.Tensor, ux: torch.Tensor, uz: torch.Tensor, box: tuple[float, float, float, float]
) -> torch.Tensor:
    """Distance along ``u`` from ``(x, z)`` to the border of the box ``(x_min, x_max, z_min, z_max)``; 0 outside."""
    x_min, x_max, z_min, z_max = box
    inf = torch.full_like(x, math.inf)
    tx = torch.where(ux > 0, (x_max - x) / ux, torch.where(ux < 0, (x_min - x) / ux, inf))
    tz = torch.where(uz > 0, (z_max - z) / uz, torch.where(uz < 0, (z_min - z) / uz, inf))
    outside = (x < x_min) | (x > x_max) | (z < z_min) | (z > z_max)
    return torch.where(outside, torch.zeros_like(x), torch.clamp(torch.minimum(tx, tz), min=0.0))


def first_module_hit(
    x: torch.Tensor,
    z: torch.Tensor,
    ux: torch.Tensor,
    uz: torch.Tensor,
    length: torch.Tensor,
    modules: torch.Tensor,
    radius: float,
    block_elements: int = BLOCK_ELEMENTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First module disc entered by each segment ``[p, p + length u]`` (exact segment-disc intersection).

    For a module centre ``c``: ``a = (c - p) . u`` (distance along the ray to the closest approach), ``q = (c - p) x u``
    (perpendicular distance); the ray enters the disc at ``s_in = a - sqrt(r^2 - q^2)`` if ``|q| < r``. The segment
    hits the module if ``s_in <= length`` and the exit point ``s_out = a + sqrt(r^2 - q^2) > 0`` (a segment that
    starts inside a disc hits it at distance 0).

    Parameters
    ----------
    x, z, ux, uz
        Start points and unit directions of the segments (float64 tensors ``[N]``).
    length
        Lengths of the segments ``[N]`` (may be ``inf``).
    modules
        Module centres ``[M, 2]`` (x, z), float64.
    radius
        Module radius [m].
    block_elements
        Number of segment x module pairs computed at once.

    Returns
    -------
    distance : torch.Tensor
        Distance to the hit point ``[N]`` (``inf`` where no module is hit).
    module : torch.Tensor
        Row of the module in the geometry table ``[N]`` (int64, -1 where no module is hit).
    """
    n, m = x.numel(), modules.shape[0]
    distance = torch.full((n,), math.inf, dtype=x.dtype, device=x.device)
    module = torch.full((n,), -1, dtype=torch.int64, device=x.device)
    if n == 0 or m == 0:
        return distance, module
    mx, mz = modules[:, 0], modules[:, 1]
    r2 = radius * radius
    block = max(1, block_elements // m)
    for a0 in range(0, n, block):
        sl = slice(a0, min(n, a0 + block))
        dx = mx[None, :] - x[sl, None]
        dz = mz[None, :] - z[sl, None]
        bux, buz = ux[sl, None], uz[sl, None]
        along = dx * bux + dz * buz
        perp = dx * buz - dz * bux
        half = torch.sqrt(torch.clamp(r2 - perp * perp, min=0.0))
        s_in = along - half
        valid = (perp.abs() < radius) & (along + half > 0) & (s_in <= length[sl, None])
        s_hit = torch.where(valid, torch.clamp(s_in, min=0.0), math.inf)
        best, index = torch.min(s_hit, dim=1)
        distance[sl] = best
        module[sl] = torch.where(torch.isfinite(best), index, -1)
    return distance, module


class ModuleGrid:
    """Grid of lower bounds of the distance from any point of a cell to the nearest module disc.

    For a cell with centre ``q`` and half-diagonal ``h / sqrt 2``: ``D = min_j |q - c_j| - h / sqrt 2 - r``. A photon
    in the cell cannot reach a module along a segment shorter than ``D``, so the exact test can be skipped.

    Parameters
    ----------
    modules
        Module centres ``[M, 2]`` (float64 tensor).
    radius
        Module radius [m].
    box
        World box ``(x_min, x_max, z_min, z_max)`` covered by the grid.
    max_cells
        Upper limit of the number of cells (sets the cell size).
    """

    def __init__(
        self, modules: torch.Tensor, radius: float, box: tuple[float, float, float, float], max_cells: int = 2**18
    ) -> None:
        x_min, x_max, z_min, z_max = box
        width, height = x_max - x_min, z_max - z_min
        self.h = max(math.sqrt(width * height / max_cells), 0.5)
        self.x0, self.z0 = x_min, z_min
        self.nx = max(1, math.ceil(width / self.h))
        self.nz = max(1, math.ceil(height / self.h))
        device = modules.device
        cx = self.x0 + (torch.arange(self.nx, device=device, dtype=torch.float64) + 0.5) * self.h
        cz = self.z0 + (torch.arange(self.nz, device=device, dtype=torch.float64) + 0.5) * self.h
        centres = torch.stack(torch.meshgrid(cx, cz, indexing="ij"), dim=-1).reshape(-1, 2)
        nearest = torch.empty(centres.shape[0], dtype=torch.float64, device=device)
        block = max(1, BLOCK_ELEMENTS // max(1, modules.shape[0]))
        for a0 in range(0, centres.shape[0], block):
            nearest[a0 : a0 + block] = torch.cdist(centres[a0 : a0 + block], modules).min(dim=1).values
        self.bound = nearest - self.h / math.sqrt(2.0) - radius

    def lower_bound(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Lower bound of the distance from ``(x, z)`` to the nearest module disc (points inside the grid)."""
        ix = torch.clamp(((x - self.x0) / self.h).floor().long(), 0, self.nx - 1)
        iz = torch.clamp(((z - self.z0) / self.h).floor().long(), 0, self.nz - 1)
        return self.bound[ix * self.nz + iz]


# ----------------------------------------------------------------------------------------------------------------------
# Transport


@dataclass
class PathPoints:
    """Points of the recorded photon paths, ordered by photon and, for each photon, along its path."""

    photon: torch.Tensor  #: index of the photon in the chunk's photon list (int64)
    x: torch.Tensor  #: point coordinates [m] (float64)
    z: torch.Tensor
    t: torch.Tensor  #: time at the point [ns] (float64)
    kind: torch.Tensor  #: PointType code (int8)


@dataclass
class TransportResult:
    """Fate of every photon of a chunk (tensors of length n_photons, in the order of the input photons)."""

    fate: torch.Tensor  #: PointType code of the last point (int8)
    module: torch.Tensor  #: row of the module of the last point in the geometry table, -1 if none (int64)
    path_length: torch.Tensor  #: total path length [m] (float64)
    n_scatter: torch.Tensor  #: number of scatterings (int32)
    paths: PathPoints | None = None  #: recorded paths, if requested


def transport(
    photons: Photons,
    modules: torch.Tensor,
    radius: float,
    *,
    absorption_length: float,
    scattering_length: float,
    g: float,
    speed_of_light: float,
    efficiency: float,
    world: tuple[float, float, float, float],
    max_steps: int,
    generator: torch.Generator,
    grid: ModuleGrid | None = None,
    record: torch.Tensor | None = None,
) -> TransportResult:
    """Track all photons of a chunk from emission to their fate.

    Parameters
    ----------
    photons
        Emitted photons (positions, directions and times in float64).
    modules
        Module centres of the chunk's geometry ``[M, 2]`` (float64, same device).
    radius
        Module radius [m].
    absorption_length, scattering_length
        lambda_a and the geometric scattering length lambda_s [m]; ``inf`` switches the process off.
    g
        Anisotropy of the Henyey-Greenstein phase function.
    speed_of_light
        Speed of light in the medium [m/ns] (for the times of recorded path points).
    efficiency
        Detection probability of a photon that reaches a module.
    world
        World box ``(x_min, x_max, z_min, z_max)`` [m]; photons that leave it escape.
    max_steps
        Maximum number of straight segments per photon.
    generator
        Source of all random numbers, on the device of the tensors.
    grid
        Optional :class:`ModuleGrid` of the geometry over the world box (speed only).
    record
        Boolean mask of the photons whose path points are recorded, or ``None``.

    Returns
    -------
    TransportResult
    """
    device = photons.x.device
    if device.type == "cpu":
        warm_up_cpu_kernels()
    n = len(photons)
    f64 = torch.float64
    fate = torch.full((n,), PointType.LOST, dtype=torch.int8, device=device)
    hit_module = torch.full((n,), -1, dtype=torch.int64, device=device)
    path_length = torch.zeros(n, dtype=f64, device=device)
    n_scatter_out = torch.zeros(n, dtype=torch.int32, device=device)

    index = torch.arange(n, device=device)
    x, z = photons.x.clone(), photons.z.clone()
    ux, uz = photons.ux.clone(), photons.uz.clone()
    t_emit = photons.t
    path = torch.zeros(n, dtype=f64, device=device)
    n_scatter = torch.zeros(n, dtype=torch.int32, device=device)
    if math.isfinite(absorption_length):
        budget = -absorption_length * torch.log1p(-torch.rand(n, generator=generator, device=device, dtype=f64))
    else:
        budget = torch.full((n,), math.inf, dtype=f64, device=device)
    scatter_on = math.isfinite(scattering_length)
    tracked = record.clone() if record is not None else None
    points: list[tuple[torch.Tensor, ...]] = []

    def add_points(sel: torch.Tensor, kind: torch.Tensor) -> None:
        assert tracked is not None
        i = (tracked & sel).nonzero().squeeze(1)
        if i.numel():
            points.append((index[i], x[i], z[i], t_emit[index[i]] + path[i] / speed_of_light, kind[i].to(torch.int8)))

    if tracked is not None:
        add_points(torch.ones(n, dtype=torch.bool, device=device), torch.full((n,), int(PointType.EMIT), device=device))

    for _ in range(max_steps):
        m = index.numel()
        if m == 0:
            break
        # 1. the straight segment
        if scatter_on:
            ell = -scattering_length * torch.log1p(-torch.rand(m, generator=generator, device=device, dtype=f64))
        else:
            ell = torch.full((m,), math.inf, dtype=f64, device=device)
        s_exit = exit_distance(x, z, ux, uz, world)
        seg = torch.minimum(torch.minimum(ell, budget), s_exit)
        # 2. the first module on the segment
        s_hit = torch.full((m,), math.inf, dtype=f64, device=device)
        mod = torch.full((m,), -1, dtype=torch.int64, device=device)
        cand = (
            (seg >= grid.lower_bound(x, z)).nonzero().squeeze(1) if grid is not None else torch.arange(m, device=device)
        )
        if cand.numel():
            d, k = first_module_hit(x[cand], z[cand], ux[cand], uz[cand], seg[cand], modules, radius)
            s_hit[cand], mod[cand] = d, k
        hit = mod >= 0
        # 3. what happens at the end of the segment
        absorbed = ~hit & (budget <= ell) & (budget <= s_exit)
        escaped = ~hit & ~absorbed & (s_exit <= ell)
        what = torch.full((m,), int(PointType.SCATTER), dtype=torch.int64, device=device)
        what[escaped] = int(PointType.ESCAPE)
        what[absorbed] = int(PointType.ABSORB)
        if hit.any():
            h = hit.nonzero().squeeze(1)
            if efficiency >= 1.0:
                what[h] = int(PointType.DETECT)
            else:
                u = torch.rand(h.numel(), generator=generator, device=device, dtype=f64)
                what[h] = torch.where(u < efficiency, int(PointType.DETECT), int(PointType.ABSORB_MODULE))
        step = torch.where(hit, s_hit, seg)
        x = x + step * ux
        z = z + step * uz
        path = path + step
        budget = budget - step
        if tracked is not None:
            add_points(torch.ones(m, dtype=torch.bool, device=device), what)
        ended = what != int(PointType.SCATTER)
        e = ended.nonzero().squeeze(1)
        k = index[e]
        fate[k] = what[e].to(torch.int8)
        hit_module[k] = mod[e]
        path_length[k] = path[e]
        n_scatter_out[k] = n_scatter[e]
        # 4. the others scatter
        keep = (~ended).nonzero().squeeze(1)
        index, x, z, ux, uz = index[keep], x[keep], z[keep], ux[keep], uz[keep]
        path, budget, n_scatter = path[keep], budget[keep], n_scatter[keep] + 1
        if tracked is not None:
            tracked = tracked[keep]
        if keep.numel():
            ux, uz = rotate(ux, uz, sample_hg_angle(keep.numel(), g, generator))

    # photons still in flight after max_steps are lost (their fate is LOST already)
    path_length[index] = path
    n_scatter_out[index] = n_scatter
    if tracked is not None:
        add_points(
            torch.ones(index.numel(), dtype=torch.bool, device=device),
            torch.full((index.numel(),), int(PointType.LOST), device=device),
        )
    paths = None
    if record is not None:
        if points:
            photon, px, pz, pt, kind = (torch.cat(column) for column in zip(*points, strict=True))
        else:
            empty = torch.zeros(0, dtype=f64, device=device)
            photon, px, pz, pt = torch.zeros(0, dtype=torch.int64, device=device), empty, empty, empty
            kind = torch.zeros(0, dtype=torch.int8, device=device)
        order = torch.sort(photon, stable=True).indices
        paths = PathPoints(photon[order], px[order], pz[order], pt[order], kind[order])
    return TransportResult(fate=fate, module=hit_module, path_length=path_length, n_scatter=n_scatter_out, paths=paths)
