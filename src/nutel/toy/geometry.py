"""Detector geometry: a table of optical modules on vertical strings.

A :class:`Geometry` is a table with one row per module: ``string`` (int), ``module`` (int, 0 = deepest), ``x`` and
``z`` (m), plus one module radius for the whole detector (modules are discs). Layouts::

    line       N_str strings with spacing d_s, N_mod modules per string with spacing d_m (the default: 8 x 12,
               50 m x 20 m, i.e. an instrumented area of 350 m x 220 m)
    core       a dense core: the n_core central strings with the spacing core_spacing, the others at d_s
    irregular  random string positions over the width of the line layout, gaps >= min_string_distance
    ragged     the line layout with a random number of modules and a random vertical offset per string
    custom     any table, Geometry.from_table(string, module, x, z, radius)

Generated layouts are centred on the origin (the centre of the bounding box of the modules). The instrumented area
is the bounding box of the module centres (:attr:`Geometry.bounds`). Hits carry the module positions in metres, so
any analysis can read any layout.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .config import Layout

__all__ = ["Geometry", "make_geometry"]


@dataclass(frozen=True, eq=False)
class Geometry:
    """Table of optical modules (one row per module) and the module radius.

    Parameters
    ----------
    string, module
        String number and module number on the string (0 = deepest) of every module (int32 arrays).
    x, z
        Module centres [m] (float32 arrays).
    radius
        Radius of the disc-shaped modules [m].
    layout
        Name of the layout (``"line"``, ``"core"``, ``"irregular"``, ``"ragged"`` or ``"custom"``).
    params
        Parameters of the layout (JSON-serialisable; empty for custom tables).

    Notes
    -----
    Two geometries are equal if their :attr:`geometry_id` is equal: the first 12 hex characters of the SHA-1 of the
    table (string, module, x and z rounded to cm) and the radius (rounded to mm).
    """

    string: np.ndarray
    module: np.ndarray
    x: np.ndarray
    z: np.ndarray
    radius: float = 0.5
    layout: str = "custom"
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        string = np.ascontiguousarray(self.string, dtype=np.int32).ravel()
        module = np.ascontiguousarray(self.module, dtype=np.int32).ravel()
        x = np.ascontiguousarray(self.x, dtype=np.float32).ravel()
        z = np.ascontiguousarray(self.z, dtype=np.float32).ravel()
        if not (len(string) == len(module) == len(x) == len(z)) or len(x) == 0:
            raise ValueError("string, module, x and z must be non-empty arrays of the same length")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z))):
            raise ValueError("module positions must be finite")
        radius = float(self.radius)
        if not radius > 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        keys = string.astype(np.int64) * (1 << 32) + module
        if len(np.unique(keys)) != len(keys):
            raise ValueError("every (string, module) pair must be unique")
        for name, value in (("string", string), ("module", module), ("x", x), ("z", z)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "layout", str(self.layout))
        object.__setattr__(self, "params", json.loads(json.dumps(dict(self.params))))
        if 1 < len(x) <= 4000:  # overlap check (quadratic; skipped for very large tables)
            d2 = (x[:, None].astype(np.float64) - x[None, :]) ** 2 + (z[:, None].astype(np.float64) - z[None, :]) ** 2
            np.fill_diagonal(d2, np.inf)
            if np.min(d2) < (2 * radius) ** 2:
                raise ValueError(f"modules overlap: two module centres are closer than 2 x radius = {2 * radius} m")

    # ---------------------------------------------------------------------------------------------- construction
    @classmethod
    def from_table(
        cls,
        string: ArrayLike,
        module: ArrayLike,
        x: ArrayLike,
        z: ArrayLike,
        radius: float = 0.5,
    ) -> Geometry:
        """Geometry of the layout ``"custom"`` from an explicit module table.

        Parameters
        ----------
        string, module
            String and module numbers (integers; every pair unique).
        x, z
            Module centres [m].
        radius
            Module radius [m].
        """
        return cls(
            string=np.asarray(string),
            module=np.asarray(module),
            x=np.asarray(x),
            z=np.asarray(z),
            radius=radius,
            layout="custom",
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python dictionary (JSON-serialisable) with the table, radius, layout and parameters."""
        return {
            "string": self.string.tolist(),
            "module": self.module.tolist(),
            "x": self.x.tolist(),
            "z": self.z.tolist(),
            "radius": self.radius,
            "layout": self.layout,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Geometry:
        """Inverse of :meth:`to_dict`."""
        return cls(
            string=np.asarray(data["string"]),
            module=np.asarray(data["module"]),
            x=np.asarray(data["x"]),
            z=np.asarray(data["z"]),
            radius=float(data.get("radius", 0.5)),
            layout=str(data.get("layout", "custom")),
            params=dict(data.get("params") or {}),
        )

    # ---------------------------------------------------------------------------------------------- properties
    @property
    def geometry_id(self) -> str:
        """First 12 hex characters of the SHA-1 of the table rounded to cm (and the radius rounded to mm)."""
        table = np.stack(
            [
                self.string.astype(np.int64),
                self.module.astype(np.int64),
                np.round(self.x.astype(np.float64) * 100).astype(np.int64),
                np.round(self.z.astype(np.float64) * 100).astype(np.int64),
            ],
            axis=1,
        )
        digest = hashlib.sha1(np.ascontiguousarray(table, dtype="<i8").tobytes())
        digest.update(str(round(self.radius * 1000)).encode())
        return digest.hexdigest()[:12]

    @property
    def n_modules(self) -> int:
        """Total number of modules."""
        return len(self.x)

    @property
    def n_strings(self) -> int:
        """Number of strings."""
        return len(np.unique(self.string))

    @property
    def strings(self) -> np.ndarray:
        """Sorted string numbers (int32)."""
        return np.unique(self.string)

    @property
    def modules_per_string(self) -> np.ndarray:
        """Number of modules of every string, in the order of :attr:`strings` (int64)."""
        return np.unique(self.string, return_counts=True)[1]

    @property
    def positions(self) -> np.ndarray:
        """Module centres as an array ``[n_modules, 2]`` with the axis order (x, z) (float32)."""
        return np.stack([self.x, self.z], axis=1)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Instrumented area, the bounding box of the module centres: ``(x_min, x_max, z_min, z_max)`` [m]."""
        return float(np.min(self.x)), float(np.max(self.x)), float(np.min(self.z)), float(np.max(self.z))

    @property
    def size(self) -> tuple[float, float]:
        """Width and height of the instrumented area [m]."""
        x0, x1, z0, z1 = self.bounds
        return x1 - x0, z1 - z0

    @property
    def center(self) -> tuple[float, float]:
        """Centre of the instrumented area [m]."""
        x0, x1, z0, z1 = self.bounds
        return 0.5 * (x0 + x1), 0.5 * (z0 + z1)

    @property
    def half_diagonal(self) -> float:
        """Half of the diagonal of the instrumented area [m]."""
        w, h = self.size
        return 0.5 * float(np.hypot(w, h))

    def module_index(self, string: ArrayLike, module: ArrayLike) -> np.ndarray:
        """Row of the table of every ``(string, module)`` pair (-1 where the pair does not exist)."""
        keys = self.string.astype(np.int64) * (1 << 32) + self.module
        order = np.argsort(keys)
        query = np.asarray(string, dtype=np.int64) * (1 << 32) + np.asarray(module, dtype=np.int64)
        pos = np.clip(np.searchsorted(keys[order], query), 0, len(keys) - 1)
        found = keys[order][pos] == query
        return np.where(found, order[pos], -1)

    def describe(self) -> str:
        """One-paragraph description (used by ``nutelsim info``)."""
        x0, x1, z0, z1 = self.bounds
        counts = self.modules_per_string
        per_string = f"{counts[0]}" if np.all(counts == counts[0]) else f"{np.min(counts)}-{np.max(counts)}"
        params = ", ".join(f"{k} = {v}" for k, v in self.params.items())
        return (
            f"geometry {self.geometry_id} ({self.layout}{': ' + params if params else ''}): {self.n_strings} strings, "
            f"{self.n_modules} modules ({per_string} per string) of radius {self.radius:g} m; instrumented area "
            f"x in [{x0:.1f}, {x1:.1f}] m, z in [{z0:.1f}, {z1:.1f}] m"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Geometry):
            return NotImplemented
        return self.geometry_id == other.geometry_id

    def __hash__(self) -> int:
        return hash(self.geometry_id)

    def __repr__(self) -> str:
        return f"Geometry({self.geometry_id}, {self.layout}, {self.n_strings} strings, {self.n_modules} modules)"


# ----------------------------------------------------------------------------------------------------------------------
# Layouts


def make_geometry(spec: Layout | None = None, **overrides: Any) -> Geometry:
    """Build the module table of a layout.

    Parameters
    ----------
    spec
        Layout parameters (default: :class:`~nutel.toy.config.Layout`, the 8 x 12 line).
    **overrides
        Layout parameters to change, e.g. ``make_geometry(layout="core", n_core=4)``.

    Returns
    -------
    Geometry

    Examples
    --------
    >>> g = make_geometry()
    >>> g.n_strings, g.n_modules, g.bounds
    (8, 96, (-175.0, 175.0, -110.0, 110.0))
    """
    import dataclasses

    p = spec if spec is not None else Layout()
    if overrides:
        p = dataclasses.replace(p, **overrides)
    n_str, n_mod = p.n_strings, p.modules_per_string
    rng = np.random.default_rng(p.layout_seed)
    width = (n_str - 1) * p.string_spacing
    counts = np.full(n_str, n_mod)
    offsets = np.zeros(n_str)
    params: dict[str, Any] = {
        "n_strings": n_str,
        "modules_per_string": n_mod,
        "string_spacing": p.string_spacing,
        "module_spacing": p.module_spacing,
    }
    if p.layout == "line":
        xs = np.arange(n_str) * p.string_spacing
    elif p.layout == "core":
        n_core = max(p.n_core, 0)
        core = np.arange(n_core) * p.core_spacing
        core = core - core.mean() if n_core else core
        n_outer = n_str - n_core
        left, right = n_outer // 2, n_outer - n_outer // 2
        edge = np.max(core) if n_core else -0.5 * p.string_spacing
        right_x = edge + p.string_spacing * np.arange(1, right + 1)
        left_x = (np.min(core) if n_core else 0.5 * p.string_spacing) - p.string_spacing * np.arange(1, left + 1)
        xs = np.sort(np.concatenate([left_x, core, right_x]))
        params |= {"n_core": n_core, "core_spacing": p.core_spacing}
    elif p.layout == "irregular":
        if n_str == 1:
            xs = np.zeros(1)
        else:
            free = width - (n_str - 1) * p.min_string_distance
            gaps = p.min_string_distance + free * rng.dirichlet(np.ones(n_str - 1))
            xs = np.concatenate([[0.0], np.cumsum(gaps)])
        params |= {"min_string_distance": p.min_string_distance, "layout_seed": p.layout_seed}
    elif p.layout == "ragged":
        xs = np.arange(n_str) * p.string_spacing
        counts = rng.integers(p.ragged_min_modules, n_mod + 1, size=n_str)
        offsets = rng.uniform(-p.ragged_max_offset, p.ragged_max_offset, size=n_str)
        params |= {
            "ragged_min_modules": p.ragged_min_modules,
            "ragged_max_offset": p.ragged_max_offset,
            "layout_seed": p.layout_seed,
        }
    else:  # pragma: no cover - Layout validates the choice
        raise ValueError(f"unknown layout {p.layout!r}")
    string = np.repeat(np.arange(n_str), counts)
    module = np.concatenate([np.arange(c) for c in counts])
    x = np.repeat(xs, counts)
    z = module * p.module_spacing + np.repeat(offsets, counts)
    # centre the bounding box of the module centres on the origin
    x = x - 0.5 * (x.min() + x.max())
    z = z - 0.5 * (np.min(z) + np.max(z))
    params["module_radius"] = p.module_radius
    return Geometry(
        string=string,
        module=module,
        x=np.round(x, 6),
        z=np.round(z, 6),
        radius=p.module_radius,
        layout=p.layout,
        params=params,
    )
