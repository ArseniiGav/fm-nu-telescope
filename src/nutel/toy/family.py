"""Families of detectors: layouts, media and readouts sampled from ranges (for multi-detector data sets).

A :class:`DetectorFamily` holds the RANGES of the detector parameters; :func:`sample_detector` draws one
:class:`~nutel.toy.data.Detector` (geometry, medium, readout) from it::

    layout       line, core or irregular (or ragged) with probabilities layout_weights
    geometry     n_strings, modules_per_string (uniform integers), string and module spacing, module radius (uniform);
                 core: the central fraction core_fraction of the strings at core_spacing_factor x d_s;
                 irregular: random string positions with gaps >= (1 - irregularity) x d_s
    medium       absorption length and effective scattering length log-uniform; the refractive index and g are
                 interpolated between the water and ice presets with the position of lambda_eff between them
    readout      efficiency, time jitter (uniform), noise rate (log-uniform; uniform if the range starts at 0)

Presets (:data:`FAMILIES`): ``default`` (around the default 8 x 12 line in water and ice) and ``tridentlike``
(large, sparse, irregular, water; held out in stage C). The readout window, the merge window and the physics switches
come from the run's configuration.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import MEDIA, Layout, Medium, SimConfig, _param, _Part, _require
from .data import Detector
from .geometry import make_geometry

__all__ = ["FAMILIES", "DetectorFamily", "family", "sample_detector", "sample_detectors"]

_LAYOUT_CHOICES = ("line", "core", "irregular", "ragged")


@dataclass(frozen=True)
class DetectorFamily(_Part):
    """Ranges of the detector parameters of a family (every range ``(min, max)``; ``min == max`` fixes a value)."""

    layout_weights: tuple[float, float, float, float] = _param(
        (0.4, 0.3, 0.3, 0.0),
        "probabilities of the layouts line, core, irregular, ragged",
        nargs=4,
        metavar=("LINE", "CORE", "IRREGULAR", "RAGGED"),
    )
    n_strings_range: tuple[int, int] = _param((6, 10), "number of strings", nargs=2, integer=True)
    modules_per_string_range: tuple[int, int] = _param((8, 16), "modules per string", nargs=2, integer=True)
    string_spacing_range: tuple[float, float] = _param((35.0, 70.0), "string spacing d_s", "m", nargs=2)
    module_spacing_range: tuple[float, float] = _param((12.0, 30.0), "module spacing d_m", "m", nargs=2)
    module_radius_range: tuple[float, float] = _param((0.3, 0.7), "module radius", "m", nargs=2)
    core_fraction_range: tuple[float, float] = _param(
        (0.3, 0.6), "core layout: fraction of strings in the core", nargs=2
    )
    core_spacing_factor_range: tuple[float, float] = _param(
        (0.3, 0.6), "core layout: core spacing in units of d_s", nargs=2
    )
    irregularity_range: tuple[float, float] = _param(
        (0.2, 0.8), "irregular layout: minimum gap = (1 - irregularity) d_s", nargs=2
    )
    absorption_length_range: tuple[float, float] = _param(
        (60.0, 100.0), "absorption length lambda_a (log-uniform)", "m", nargs=2
    )
    scattering_length_eff_range: tuple[float, float] = _param(
        (25.0, 300.0), "effective scattering length lambda_eff (log-uniform)", "m", nargs=2
    )
    efficiency_range: tuple[float, float] = _param((0.8, 1.0), "detection efficiency", nargs=2)
    time_jitter_range: tuple[float, float] = _param((0.5, 2.0), "time jitter", "ns", nargs=2)
    noise_rate_range: tuple[float, float] = _param(
        (500.0, 50000.0), "noise rate per module (log-uniform; uniform if MIN = 0)", "Hz", nargs=2
    )

    def _validate(self) -> None:
        w = self.layout_weights
        _require(all(v >= 0 for v in w) and sum(w) > 0, f"bad layout_weights {w}")
        for f in dataclasses.fields(self):
            if f.name.endswith("_range"):
                lo, hi = getattr(self, f.name)
                _require(lo <= hi, f"{f.name}: need MIN <= MAX, got {(lo, hi)}")
        _require(self.n_strings_range[0] >= 1, "n_strings_range must start at >= 1")
        _require(self.modules_per_string_range[0] >= 1, "modules_per_string_range must start at >= 1")
        _require(self.string_spacing_range[0] > 0 and self.module_spacing_range[0] > 0, "spacings must be positive")
        _require(self.module_radius_range[0] > 0, "module radius must be positive")
        _require(
            2 * self.module_radius_range[1] < self.module_spacing_range[0],
            "modules overlap: 2 x max module radius must be below the min module spacing",
        )
        _require(self.core_fraction_range[0] >= 0 and self.core_fraction_range[1] <= 1, "core_fraction in [0, 1]")
        _require(
            2 * self.module_radius_range[1] < self.core_spacing_factor_range[0] * self.string_spacing_range[0],
            "core strings overlap: core_spacing_factor x d_s must exceed 2 x module radius",
        )
        _require(self.irregularity_range[0] >= 0 and self.irregularity_range[1] < 1, "irregularity in [0, 1)")
        _require(
            2 * self.module_radius_range[1] < (1 - self.irregularity_range[1]) * self.string_spacing_range[0],
            "irregular strings overlap: (1 - irregularity) x d_s must exceed 2 x module radius",
        )
        _require(self.absorption_length_range[0] > 0, "absorption lengths must be positive")
        _require(self.scattering_length_eff_range[0] > 0, "scattering lengths must be positive")
        _require(self.efficiency_range[0] >= 0 and self.efficiency_range[1] <= 1, "efficiency in [0, 1]")
        _require(self.time_jitter_range[0] >= 0, "time jitter must be >= 0")
        _require(self.noise_rate_range[0] >= 0, "noise rate must be >= 0")

    @classmethod
    def preset(cls, name: str = "default", **overrides: Any) -> DetectorFamily:
        """Return a family preset (``"default"`` or ``"tridentlike"``), optionally with ranges changed."""
        return family(name, **overrides)

    def describe(self) -> str:
        """Table of the ranges (used by ``nutelsim family``)."""
        lines = [
            "layout weights: "
            + ", ".join(f"{n} {w:g}" for n, w in zip(_LAYOUT_CHOICES, self.layout_weights, strict=True))
        ]
        for f in dataclasses.fields(self):
            if f.name.endswith("_range"):
                lo, hi = getattr(self, f.name)
                unit = f" {f.metadata['unit']}" if f.metadata["unit"] else ""
                lines.append(f"{f.name[:-6]:<26} [{lo:g}, {hi:g}]{unit}   {f.metadata['help']}")
        return "\n".join(lines)


#: Family presets: ``default`` (around the default line, water to ice) and ``tridentlike`` (large, sparse,
#: irregular, water).
FAMILIES: dict[str, DetectorFamily] = {
    "default": DetectorFamily(),
    "tridentlike": DetectorFamily(
        layout_weights=(0.0, 0.0, 1.0, 0.0),
        n_strings_range=(16, 20),
        modules_per_string_range=(16, 20),
        string_spacing_range=(70.0, 110.0),
        module_spacing_range=(25.0, 35.0),
        module_radius_range=(0.4, 0.6),
        irregularity_range=(0.3, 0.6),
        absorption_length_range=(MEDIA["water"].absorption_length,) * 2,
        scattering_length_eff_range=(MEDIA["water"].scattering_length_eff,) * 2,
        efficiency_range=(0.8, 1.0),
        time_jitter_range=(0.5, 1.5),
        noise_rate_range=(20000.0, 100000.0),
    ),
}


def family(name: str = "default", **overrides: Any) -> DetectorFamily:
    """Return a family preset by name, optionally with ranges changed."""
    fam = _named(name)
    return dataclasses.replace(fam, **overrides) if overrides else fam


def _named(name: str) -> DetectorFamily:
    if name not in FAMILIES:
        raise ValueError(f"unknown detector family {name!r}; choose from {', '.join(FAMILIES)}")
    return FAMILIES[name]


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def _log_uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    if hi <= lo:
        return float(lo)
    if lo <= 0:
        return float(rng.uniform(lo, hi))
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _interpolated_medium(absorption_length: float, scattering_length_eff: float) -> Medium:
    """Medium with n and g interpolated between water and ice with the position of lambda_eff (log scale)."""
    water, ice = MEDIA["water"], MEDIA["ice"]
    a, b = math.log(water.scattering_length_eff), math.log(ice.scattering_length_eff)
    w = min(max((math.log(scattering_length_eff) - a) / (b - a), 0.0), 1.0) if a != b else 0.0
    return Medium(
        refractive_index=round(water.refractive_index + w * (ice.refractive_index - water.refractive_index), 6),
        absorption_length=absorption_length,
        scattering_length_eff=scattering_length_eff,
        g=round(water.g + w * (ice.g - water.g), 6),
    )


def sample_detector(
    rng: np.random.Generator | int, family: DetectorFamily | str = "default", config: SimConfig | None = None
) -> Detector:
    """Draw one detector (geometry, medium, readout) from a family.

    Parameters
    ----------
    rng
        NumPy generator or integer seed.
    family
        The family (or the name of a preset).
    config
        Base configuration: its readout window and merge window are kept (default: :class:`SimConfig`).

    Returns
    -------
    Detector
    """
    fam = _named(family) if isinstance(family, str) else family
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    base = config or SimConfig()
    weights = np.asarray(fam.layout_weights, dtype=np.float64)
    kind = _LAYOUT_CHOICES[int(rng.choice(4, p=weights / weights.sum()))]
    n_strings = int(rng.integers(fam.n_strings_range[0], fam.n_strings_range[1] + 1))
    n_modules = int(rng.integers(fam.modules_per_string_range[0], fam.modules_per_string_range[1] + 1))
    d_s = round(_uniform(rng, fam.string_spacing_range), 3)
    d_m = round(_uniform(rng, fam.module_spacing_range), 3)
    radius = round(_uniform(rng, fam.module_radius_range), 4)
    core_fraction = _uniform(rng, fam.core_fraction_range)
    core_factor = _uniform(rng, fam.core_spacing_factor_range)
    irregularity = _uniform(rng, fam.irregularity_range)
    layout_seed = int(rng.integers(2**31))
    layout = Layout(
        layout=kind,
        n_strings=n_strings,
        modules_per_string=n_modules,
        string_spacing=d_s,
        module_spacing=d_m,
        module_radius=radius,
        n_core=min(n_strings, max(2, round(core_fraction * n_strings))) if kind == "core" else 0,
        core_spacing=round(core_factor * d_s, 3),
        min_string_distance=round((1.0 - irregularity) * d_s, 3),
        ragged_min_modules=max(1, n_modules // 2),
        ragged_max_offset=d_m,
        layout_seed=layout_seed,
    )
    medium = _interpolated_medium(
        round(_log_uniform(rng, fam.absorption_length_range), 3),
        round(_log_uniform(rng, fam.scattering_length_eff_range), 3),
    )
    efficiency = round(_uniform(rng, fam.efficiency_range), 4)
    time_jitter = round(_uniform(rng, fam.time_jitter_range), 4)
    noise_rate = round(_log_uniform(rng, fam.noise_rate_range), 1)
    readout = dataclasses.replace(base.readout, efficiency=efficiency, time_jitter=time_jitter, noise_rate=noise_rate)
    medium = dataclasses.replace(medium, background_rate=noise_rate)  # one noise rate per detector
    return Detector(geometry=make_geometry(layout), medium=medium, readout=readout)


def sample_detectors(
    n: int, family: DetectorFamily | str = "default", seed: int = 0, config: SimConfig | None = None
) -> list[Detector]:
    """Draw ``n`` detectors from a family with the seed ``seed`` (the same seed gives the same detectors)."""
    if n < 1:
        raise ValueError(f"need at least one detector, got {n}")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x6E7574]))
    return [sample_detector(rng, family, config) for _ in range(n)]
