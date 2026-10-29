"""Configuration of the toy telescope: layout, medium, readout, physics switches, emission and the event prior.

A configuration is a frozen :class:`SimConfig` made of six parts, each a frozen dataclass whose fields are the
options of the simulation (every physical constant and detector parameter of the toy is a field here, with its
default, its unit and a help text that the command line uses):

* :class:`Layout`: how the module table is built (strings, modules, spacings, module radius);
* :class:`Medium`: refractive index, absorption length, effective scattering length, anisotropy ``g``;
* :class:`Readout`: efficiency, time jitter, noise rate, readout window, pulse merging window;
* :class:`Physics`: the switches of the physics ladder (absorption, scattering, stochastic losses) and the limits of
  the transport;
* :class:`Emission`: Cherenkov light of tracks, cascades and stochastic losses;
* :class:`Prior`: the distribution of the event parameters per event type.

Parameter names are unique across the parts, so a modified copy is made with flat keywords::

    from nutel.toy import preset

    config = preset("scatter", medium="ice", noise_rate=20e3, absorption_length=40.0)
    config = config.replace(time_jitter=0.0, layout="core")

Every default equals the ``"full"`` preset in water.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from .data import Detector
    from .geometry import Geometry

__all__ = [
    "C_VACUUM",
    "EVENT_TYPES",
    "LAYOUTS",
    "MEDIA",
    "PRESETS",
    "SOURCES",
    "Emission",
    "Layout",
    "Medium",
    "Physics",
    "Prior",
    "Readout",
    "SimConfig",
    "config_fields",
    "describe_presets",
    "matching_preset",
    "medium",
    "medium_name",
    "preset",
]

#: Speed of light in vacuum [m/ns].
C_VACUUM = 0.299792458

#: Event types: code -> name (the code is stored in ``truth.type``).
EVENT_TYPES: tuple[str, ...] = ("track", "starting", "cascade", "atmospheric")

#: Light sources of a hit: code -> name (the code is stored in ``hits.source``; 3 marks noise hits).
SOURCES: tuple[str, ...] = ("track", "burst", "cascade", "noise")

#: Layouts that can be built from the parameters of :class:`Layout` (``"custom"`` comes from an explicit table).
LAYOUTS: tuple[str, ...] = ("line", "core", "irregular", "ragged", "custom")

PhotonStatistics = Literal["poisson", "fixed"]


def _param(default: Any, help: str, unit: str = "", **extra: Any) -> Any:
    """Dataclass field with a help text and a unit (used by ``describe()`` and the command line).

    Extra metadata: ``choices`` (allowed strings), ``nargs`` (length of a tuple value), ``metavar`` (names of the
    values on the command line), ``allow_inf`` (``inf`` is a valid value, e.g. a length that switches a process off),
    ``optional`` (``None`` is a valid value), ``integer`` (the tuple holds integers).
    """
    return field(default=default, metadata={"help": help, "unit": unit, **extra})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, name: str, *, integer: bool, allow_inf: bool) -> float | int:
    """Convert one value to a Python float or int (NumPy scalars included) and check that it is finite."""
    if isinstance(value, (bool, str)) or type(value).__name__ == "bool_":
        raise TypeError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number, got {value!r}") from None
    if math.isnan(number) or (math.isinf(number) and not allow_inf):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if integer:
        if math.isinf(number) or number != int(number):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        return int(number)
    return number


def _coerce(f: dataclasses.Field[Any], value: Any, owner: str) -> Any:
    """Convert ``value`` to the type of the field (also turns NumPy scalars and lists into Python values)."""
    name = f"{owner}.{f.name}"
    meta = f.metadata
    if value is None:
        if meta.get("optional"):
            return None
        raise TypeError(f"{name} must not be None")
    if "choices" in meta:
        if value not in meta["choices"]:
            raise ValueError(f"{name} must be one of {', '.join(map(repr, meta['choices']))}, got {value!r}")
        return str(value)
    if isinstance(f.default, bool):
        if not isinstance(value, bool) and type(value).__name__ != "bool_":
            raise TypeError(f"{name} must be True or False, got {value!r}")
        return bool(value)
    integer = bool(meta.get("integer")) or (isinstance(f.default, int) and not isinstance(f.default, bool))
    allow_inf = bool(meta.get("allow_inf"))
    nargs = meta.get("nargs")
    if nargs:
        if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != nargs:
            raise ValueError(f"{name} must be a sequence of {nargs} numbers, got {value!r}")
        return tuple(_number(v, name, integer=integer, allow_inf=allow_inf) for v in value)
    return _number(value, name, integer=integer, allow_inf=allow_inf)


class _Part:
    """Behaviour shared by the configuration parts: type coercion and validation on construction."""

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            object.__setattr__(self, f.name, _coerce(f, getattr(self, f.name), type(self).__name__))
        self._validate()

    def _validate(self) -> None:
        """Check the parameter values (each part overrides this)."""

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary of plain Python values (tuples become lists; JSON-serialisable)."""
        return {
            f.name: list(getattr(self, f.name)) if isinstance(getattr(self, f.name), tuple) else getattr(self, f.name)
            for f in dataclasses.fields(self)  # type: ignore[arg-type]
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Any:
        """Inverse of :meth:`to_dict`; missing parameters take their default values."""
        names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        extra = set(data) - names
        if extra:
            raise ValueError(f"unknown parameters of {cls.__name__}: {sorted(extra)}")
        return cls(**data)


# ----------------------------------------------------------------------------------------------------------------------
# The parts


@dataclass(frozen=True)
class Layout(_Part):
    """Parameters of the module table (see :mod:`nutel.toy.geometry` for the layouts).

    Strings are vertical columns of optical modules (discs of radius ``module_radius``), module 0 is the deepest.
    ``line``: ``n_strings`` equally spaced strings; ``core``: ``n_core`` central strings with the smaller spacing
    ``core_spacing``, the others at ``string_spacing``; ``irregular``: random string positions with gaps of at least
    ``min_string_distance`` over the same width as ``line``; ``ragged``: ``line`` with a random number of modules
    (``ragged_min_modules`` to ``modules_per_string``) and a random vertical offset (up to ``ragged_max_offset``) per
    string. ``layout_seed`` fixes the random layouts. Generated layouts are centred on the origin.
    """

    layout: str = _param(
        "line",
        "layout of the strings: line, core, irregular or ragged",
        choices=("line", "core", "irregular", "ragged"),
    )
    n_strings: int = _param(8, "number of strings N_str")
    modules_per_string: int = _param(12, "number of modules per string N_mod")
    string_spacing: float = _param(50.0, "horizontal distance between neighbouring strings d_s", "m")
    module_spacing: float = _param(20.0, "vertical distance between neighbouring modules d_m", "m")
    module_radius: float = _param(0.5, "radius r_m of the (disc-shaped) optical modules", "m")
    n_core: int = _param(4, "core layout: number of central strings with the smaller spacing")
    core_spacing: float = _param(20.0, "core layout: horizontal spacing of the central strings", "m")
    min_string_distance: float = _param(20.0, "irregular layout: minimum horizontal distance between strings", "m")
    ragged_min_modules: int = _param(6, "ragged layout: minimum number of modules on a string")
    ragged_max_offset: float = _param(20.0, "ragged layout: maximum vertical offset of a string", "m")
    layout_seed: int = _param(0, "seed of the random layouts (irregular, ragged)")

    def _validate(self) -> None:
        _require(self.n_strings >= 1, f"n_strings must be >= 1, got {self.n_strings}")
        _require(self.modules_per_string >= 1, f"modules_per_string must be >= 1, got {self.modules_per_string}")
        _require(self.string_spacing > 0, f"string_spacing must be positive, got {self.string_spacing}")
        _require(self.module_spacing > 0, f"module_spacing must be positive, got {self.module_spacing}")
        _require(self.module_radius > 0, f"module_radius must be positive, got {self.module_radius}")
        _require(
            2 * self.module_radius < self.module_spacing,
            f"modules overlap: 2 x module_radius = {2 * self.module_radius} m >= module_spacing "
            f"{self.module_spacing} m",
        )
        _require(0 <= self.n_core <= self.n_strings, f"n_core must be in [0, n_strings], got {self.n_core}")
        _require(self.core_spacing > 2 * self.module_radius, f"core_spacing too small: {self.core_spacing} m")
        _require(
            self.min_string_distance > 2 * self.module_radius,
            f"min_string_distance too small: {self.min_string_distance} m",
        )
        _require(
            self.min_string_distance <= self.string_spacing,
            f"min_string_distance ({self.min_string_distance} m) must not exceed string_spacing "
            f"({self.string_spacing} m)",
        )
        _require(
            1 <= self.ragged_min_modules <= self.modules_per_string,
            f"ragged_min_modules must be in [1, modules_per_string], got {self.ragged_min_modules}",
        )
        _require(self.ragged_max_offset >= 0, f"ragged_max_offset must be >= 0, got {self.ragged_max_offset}")
        _require(self.layout_seed >= 0, f"layout_seed must be >= 0, got {self.layout_seed}")


@dataclass(frozen=True)
class Medium(_Part):
    """Optical properties of the medium (order-of-magnitude values, not a fit to any site).

    The Cherenkov angle is ``cos(theta_c) = 1 / n`` with the (phase) refractive index ``n``; light travels at
    ``c / n_group`` (``n_group = None``: equal to ``n``, a simplification). The geometric scattering length is
    ``lambda_s = lambda_eff (1 - g)``. Scattering uses the two-dimensional Henyey-Greenstein phase function with
    ``<cos theta> = g``. ``inf`` switches a process off (as do the switches of :class:`Physics`).
    ``background_rate`` is the noise-hit rate per module that the medium causes (K-40 decays and bioluminescence in
    sea water, mostly dark noise in ice); the readout uses it unless ``noise_rate`` is set.
    """

    refractive_index: float = _param(1.35, "(phase) refractive index n: Cherenkov angle cos(theta_c) = 1/n")
    n_group: float | None = _param(
        None,
        "group refractive index for the speed of light c/n_group (default: equal to n)",
        optional=True,
        metavar="N",
    )
    absorption_length: float = _param(60.0, "absorption length lambda_a", "m", allow_inf=True)
    scattering_length_eff: float = _param(
        300.0, "effective scattering length lambda_eff = lambda_s / (1 - g)", "m", allow_inf=True
    )
    g: float = _param(0.8, "scattering anisotropy g = <cos theta> of the Henyey-Greenstein phase function")
    background_rate: float = _param(50_000.0, "noise-hit rate per module caused by the medium", "Hz")

    @property
    def n(self) -> float:
        """Refractive index (short name of ``refractive_index``)."""
        return self.refractive_index

    @property
    def scattering_length(self) -> float:
        """Geometric scattering length ``lambda_s = lambda_eff (1 - g)`` [m]."""
        return self.scattering_length_eff * (1.0 - self.g)

    @property
    def cherenkov_angle(self) -> float:
        """Cherenkov angle ``theta_c = arccos(1 / n)`` [rad] of a particle with beta = 1."""
        return math.acos(1.0 / self.refractive_index)

    @property
    def group_index(self) -> float:
        """Group refractive index: ``n_group``, or ``n`` if it is not set."""
        return self.refractive_index if self.n_group is None else self.n_group

    @property
    def speed_of_light(self) -> float:
        """Speed of light in the medium ``c_n = c / n_group`` [m/ns]."""
        return C_VACUUM / self.group_index

    def _validate(self) -> None:
        _require(self.refractive_index > 1, f"refractive_index must be > 1, got {self.refractive_index}")
        _require(self.absorption_length > 0, f"absorption_length must be positive, got {self.absorption_length}")
        _require(
            self.scattering_length_eff > 0, f"scattering_length_eff must be positive, got {self.scattering_length_eff}"
        )
        _require(0 <= self.g < 1, f"g must be in [0, 1), got {self.g}")
        if self.n_group is not None:
            _require(self.n_group >= 1, f"n_group must be >= 1, got {self.n_group}")
        _require(self.background_rate >= 0, f"background_rate must be >= 0, got {self.background_rate}")


@dataclass(frozen=True)
class Readout(_Part):
    """Response of the optical modules: hits and pulses.

    A photon that reaches a module is detected with probability ``efficiency`` (else ``absorbed_module``). Its time is
    ``t_arrival + N(0, time_jitter^2)``. Noise adds ``Poisson(noise_rate x window length)`` hits per module and event,
    uniform in the window (``noise_rate = None``: the ``background_rate`` of the medium). Hits outside ``window`` are
    dropped (counted as ``n_out_of_window``). Pulses: per module, a pulse starts at the earliest unassigned hit and
    takes every hit within ``merge_window`` after it; its time is the first or the mean time of its hits.
    """

    efficiency: float = _param(1.0, "probability that a photon reaching a module is detected")
    time_jitter: float = _param(1.0, "Gaussian time resolution sigma_t per detected photon (0 disables)", "ns")
    noise_rate: float | None = _param(
        None,
        "noise-hit rate per module (0 disables; default: the background rate of the medium)",
        "Hz",
        optional=True,
        metavar="RATE",
    )
    window: tuple[float, float] = _param(
        (0.0, 5000.0), "readout window [start, end): hits outside are dropped", "ns", nargs=2, metavar=("START", "END")
    )
    merge_window: float = _param(3.5, "hits of a module within this time after a pulse start join the pulse", "ns")
    pulse_time: str = _param(
        "first", "time of a pulse: 'first' (first hit) or 'mean' (mean of its hits)", choices=("first", "mean")
    )

    @property
    def window_length(self) -> float:
        """Length of the readout window [ns]."""
        return self.window[1] - self.window[0]

    @property
    def mean_noise_hits(self) -> float:
        """Mean number of noise hits per module and event, ``noise_rate x window length`` (rate must be set)."""
        if self.noise_rate is None:
            raise ValueError("noise_rate is not set (it follows the medium): use SimConfig.effective_readout()")
        return self.noise_rate * self.window_length * 1e-9

    def _validate(self) -> None:
        _require(0 <= self.efficiency <= 1, f"efficiency must be in [0, 1], got {self.efficiency}")
        _require(self.time_jitter >= 0, f"time_jitter must be >= 0, got {self.time_jitter}")
        _require(self.noise_rate is None or self.noise_rate >= 0, f"noise_rate must be >= 0, got {self.noise_rate}")
        _require(self.window[1] > self.window[0], f"window must have end > start, got {self.window}")
        _require(self.merge_window >= 0, f"merge_window must be >= 0, got {self.merge_window}")


@dataclass(frozen=True)
class Physics(_Part):
    """Switches of the physics ladder and the limits of the photon transport.

    A process that is off behaves as an infinite length. The world is the instrumented area plus
    ``min(world_margin_factor x lambda_a, world_margin_max)`` on every side (at least ``track_margin`` of the prior,
    so that every emission point lies inside); a photon that leaves it is ``escaped``. Photons still in flight after
    ``max_steps`` scatterings are ``lost``.
    """

    absorption: bool = _param(True, "absorption in the medium")
    scattering: bool = _param(True, "scattering in the medium (2-D Henyey-Greenstein)")
    stochastic: bool = _param(True, "stochastic energy losses (bursts) along muon tracks")
    max_steps: int = _param(200, "maximum number of straight steps per photon; photons still in flight are lost")
    world_margin_factor: float = _param(2.0, "world margin around the instrumented area, in absorption lengths")
    world_margin_max: float = _param(300.0, "maximum world margin around the instrumented area", "m")

    def _validate(self) -> None:
        _require(1 <= self.max_steps <= 100_000, f"max_steps must be in [1, 100000], got {self.max_steps}")
        _require(self.world_margin_factor > 0, f"world_margin_factor must be positive, got {self.world_margin_factor}")
        _require(self.world_margin_max > 0, f"world_margin_max must be positive, got {self.world_margin_max}")


@dataclass(frozen=True)
class Emission(_Part):
    """Cherenkov light of muons and cascades (two-dimensional: the cone becomes two rays).

    A muon segment of length ``L`` emits ``Poisson(track_yield x L)`` photons uniformly along it, at
    ``+-(theta_c + N(0, cherenkov_spread^2))`` to its direction, at the time ``t0 + s / c`` (beta = 1). A cascade of
    energy ``E`` emits ``Poisson(cascade_yield x E)`` photons at its vertex and time (the hadronic vertex cascade of a
    starting track ``hadronic_light_factor`` times that), with ``cascade_yield = cascade_length_equivalent x
    track_yield`` (1 GeV of cascade gives the light of ``cascade_length_equivalent`` metres of track). The emission
    angle to the cascade axis is ``+-theta`` with ``theta`` from the Raedel-Wiebusch angular profile
    (``cascade_profile = "radel-wiebusch"``: ``dn/dOmega = a exp(b |cos theta - 1/n|^c) + d``, sampled as the 3-D
    polar angle) or ``theta = theta_c + N(0, cascade_spread^2)`` (``"gaussian"``).
    Stochastic losses: bursts occur along the muon at the rate ``b ln(1 / v_min)`` per metre; each takes the fraction
    ``v ~ 1/v`` on ``[v_min, 1]`` of the current muon energy and becomes a point-like cascade. The muon also loses
    ``energy_loss_a`` GeV per metre continuously and stops (no more light) below ``muon_stop_energy``.
    """

    track_yield: float = _param(1.75, "photons emitted per metre of muon track, Y_t (calibrated)", "1/m")
    photon_statistics: str = _param(
        "poisson",
        "number of photons per source: 'poisson' (Poisson(mean)) or 'fixed' (round(mean))",
        choices=("poisson", "fixed"),
    )
    cherenkov_spread: float = _param(1.0, "Gaussian spread sigma_c of the track emission angle", "deg")
    cascade_length_equivalent: float = _param(
        5.0, "track length with the same light as 1 GeV of cascade (Y_c = this x Y_t)", "m/GeV"
    )
    cascade_profile: str = _param(
        "radel-wiebusch",
        "angular profile of cascade light: 'radel-wiebusch' (parametrised simulation) or 'gaussian' around theta_c",
        choices=("radel-wiebusch", "gaussian"),
    )
    cascade_spread: float = _param(10.0, "gaussian profile: spread of the cascade emission angle around theta_c", "deg")
    hadronic_light_factor: float = _param(
        1.0, "light of the hadronic vertex cascade of starting tracks relative to an electromagnetic cascade"
    )
    stochastic_b: float = _param(3.3e-4, "b of dE/dx = a + b E: stochastic energy loss per metre and GeV", "1/m")
    stochastic_v_min: float = _param(1e-3, "smallest energy fraction v_min taken by a burst")
    energy_loss_a: float = _param(0.25, "a of dE/dx = a + b E: continuous energy loss of the muon (0: none)", "GeV/m")
    muon_stop_energy: float = _param(1.0, "the muon stops (and stops emitting) below this energy", "GeV")

    @property
    def cascade_yield(self) -> float:
        """Photons emitted per GeV of cascade energy, ``Y_c = cascade_length_equivalent x track_yield`` [1/GeV]."""
        return self.cascade_length_equivalent * self.track_yield

    @property
    def burst_rate(self) -> float:
        """Mean number of bursts per metre of muon track, ``b ln(1 / v_min)`` [1/m]."""
        return self.stochastic_b * math.log(1.0 / self.stochastic_v_min)

    def _validate(self) -> None:
        _require(self.track_yield >= 0, f"track_yield must be >= 0, got {self.track_yield}")
        _require(self.cherenkov_spread >= 0, f"cherenkov_spread must be >= 0, got {self.cherenkov_spread}")
        _require(
            self.cascade_length_equivalent >= 0,
            f"cascade_length_equivalent must be >= 0, got {self.cascade_length_equivalent}",
        )
        _require(self.cascade_spread >= 0, f"cascade_spread must be >= 0, got {self.cascade_spread}")
        _require(self.stochastic_b >= 0, f"stochastic_b must be >= 0, got {self.stochastic_b}")
        _require(0 < self.stochastic_v_min < 1, f"stochastic_v_min must be in (0, 1), got {self.stochastic_v_min}")
        _require(self.energy_loss_a >= 0, f"energy_loss_a must be >= 0, got {self.energy_loss_a}")
        _require(
            self.hadronic_light_factor >= 0, f"hadronic_light_factor must be >= 0, got {self.hadronic_light_factor}"
        )
        _require(self.muon_stop_energy >= 0, f"muon_stop_energy must be >= 0, got {self.muon_stop_energy}")


@dataclass(frozen=True)
class Prior(_Part):
    """Distribution of the event parameters (see :mod:`nutel.toy.events`).

    ``track``: direction angle uniform in [0, 2 pi), the line passes at the signed impact parameter
    ``b ~ U(-R_gen, R_gen)`` from the origin, ``R_gen`` = half-diagonal of the instrumented area + ``impact_margin``;
    ``starting`` and ``cascade``: vertex uniform in the instrumented area + ``vertex_margin``; the inelasticity of
    starting tracks follows ``p(y) ~ 1 + eps (1 - y)^2`` on [0, 1] (``inelasticity = "icecube"``) or is uniform in
    ``inelasticity_range``; ``atmospheric``: a down-going track with ``p(cos zenith) ~ cos z exp(K1 / cos z)``,
    ``K1 = -0.581 h + 0.034`` at the depth ``h`` in km water equivalent (``atmospheric_zenith = "depth"``, Becherini et
    al. 2006) or uniform (``"uniform"``), in ``atmospheric_cos_zenith_range``. Energies are log-uniform.
    Muons emit light inside the generation box (instrumented area + ``track_margin``). ``fixed_*`` values replace
    the corresponding distributions (the random numbers are drawn anyway, so the other parameters do not change).
    """

    type_fractions: tuple[float, float, float, float] = _param(
        (0.4, 0.2, 0.2, 0.2),
        "fractions of the event types track, starting, cascade, atmospheric",
        nargs=4,
        metavar=("TRACK", "STARTING", "CASCADE", "ATMOSPHERIC"),
    )
    event_type: str | None = _param(None, "simulate only this event type", choices=EVENT_TYPES, optional=True)
    impact_margin: float = _param(20.0, "R_gen = half-diagonal of the instrumented area + this margin", "m")
    vertex_margin: float = _param(20.0, "vertices are uniform in the instrumented area + this margin", "m")
    track_margin: float = _param(100.0, "muons emit light inside the instrumented area + this margin", "m")
    t0_range: tuple[float, float] = _param(
        (1000.0, 2000.0),
        "uniform range of t0 (time at the anchor point or vertex)",
        "ns",
        nargs=2,
        metavar=("MIN", "MAX"),
    )
    track_energy_range: tuple[float, float] = _param(
        (1e2, 1e4),
        "log-uniform range of the muon energy of through-going tracks",
        "GeV",
        nargs=2,
        metavar=("MIN", "MAX"),
    )
    starting_energy_range: tuple[float, float] = _param(
        (1e2, 1e3),
        "log-uniform range of the neutrino energy of starting tracks",
        "GeV",
        nargs=2,
        metavar=("MIN", "MAX"),
    )
    cascade_energy_range: tuple[float, float] = _param(
        (1e1, 1e3), "log-uniform range of the cascade energy", "GeV", nargs=2, metavar=("MIN", "MAX")
    )
    atmospheric_energy_range: tuple[float, float] = _param(
        (1e2, 1e3), "log-uniform range of the atmospheric-muon energy", "GeV", nargs=2, metavar=("MIN", "MAX")
    )
    inelasticity: str = _param(
        "icecube",
        "inelasticity y of starting tracks: 'icecube' (p(y) ~ 1 + eps (1 - y)^2) or 'uniform' (in inelasticity_range)",
        choices=("icecube", "uniform"),
    )
    inelasticity_eps: float = _param(1.0, "eps of the 'icecube' inelasticity distribution (mean y 0.44 for 1)")
    inelasticity_range: tuple[float, float] = _param(
        (0.05, 0.75), "'uniform' inelasticity: range of y", nargs=2, metavar=("MIN", "MAX")
    )
    atmospheric_zenith: str = _param(
        "depth",
        "zenith of atmospheric muons: 'depth' (Becherini et al. at depth_kmwe) or 'uniform' in cos(zenith)",
        choices=("depth", "uniform"),
    )
    depth_kmwe: float = _param(2.0, "'depth' zenith distribution: detector depth h", "km w.e.")
    atmospheric_cos_zenith_range: tuple[float, float] = _param(
        (0.1, 1.0), "range of cos(zenith) of atmospheric muons (down-going)", nargs=2, metavar=("MIN", "MAX")
    )
    fixed_angle: float | None = _param(
        None, "use this direction angle atan2(dir_z, dir_x) for every event", "deg", optional=True, metavar="ANGLE"
    )
    fixed_vertex: tuple[float, float] | None = _param(
        None,
        "use this vertex (cascade, starting) or point on the line (track, atmospheric) for every event",
        "m",
        nargs=2,
        optional=True,
        metavar=("X", "Z"),
    )
    fixed_energy: float | None = _param(None, "use this energy for every event", "GeV", optional=True, metavar="E")
    fixed_t0: float | None = _param(None, "use this t0 for every event", "ns", optional=True, metavar="T0")
    fixed_inelasticity: float | None = _param(
        None, "use this inelasticity y for every starting track", optional=True, metavar="Y"
    )

    def _validate(self) -> None:
        fractions = self.type_fractions
        _require(all(f >= 0 for f in fractions) and sum(fractions) > 0, f"bad type_fractions {fractions}")
        for name in ("impact_margin", "vertex_margin", "track_margin"):
            _require(getattr(self, name) >= 0, f"{name} must be >= 0, got {getattr(self, name)}")
        _require(
            self.vertex_margin <= self.track_margin,
            f"vertex_margin ({self.vertex_margin} m) must not exceed track_margin ({self.track_margin} m)",
        )
        _require(self.t0_range[0] <= self.t0_range[1], f"need t0_range MIN <= MAX, got {self.t0_range}")
        for name in ("track_energy_range", "starting_energy_range", "cascade_energy_range", "atmospheric_energy_range"):
            lo, hi = getattr(self, name)
            _require(0 < lo <= hi, f"need 0 < MIN <= MAX for {name}, got {(lo, hi)}")
        lo, hi = self.inelasticity_range
        _require(0 <= lo <= hi <= 1, f"need 0 <= MIN <= MAX <= 1 for inelasticity_range, got {(lo, hi)}")
        _require(self.inelasticity_eps >= 0, f"inelasticity_eps must be >= 0, got {self.inelasticity_eps}")
        _require(self.depth_kmwe > 0, f"depth_kmwe must be positive, got {self.depth_kmwe}")
        lo, hi = self.atmospheric_cos_zenith_range
        _require(0 <= lo <= hi <= 1, f"need 0 <= MIN <= MAX <= 1 for atmospheric_cos_zenith_range, got {(lo, hi)}")
        if self.fixed_energy is not None:
            _require(self.fixed_energy > 0, f"fixed_energy must be positive, got {self.fixed_energy}")
        if self.fixed_inelasticity is not None:
            _require(0 <= self.fixed_inelasticity <= 1, "fixed_inelasticity must be in [0, 1]")

    @property
    def type_probabilities(self) -> tuple[float, float, float, float]:
        """Normalised type fractions (one-hot for a fixed ``event_type``)."""
        if self.event_type is not None:
            k = EVENT_TYPES.index(self.event_type)
            return tuple(1.0 if i == k else 0.0 for i in range(4))  # type: ignore[return-value]
        total = sum(self.type_fractions)
        return tuple(f / total for f in self.type_fractions)  # type: ignore[return-value]


# ----------------------------------------------------------------------------------------------------------------------
# The full configuration

#: Media presets (order-of-magnitude values; see docs/references.md).
MEDIA: dict[str, Medium] = {
    "water": Medium(
        refractive_index=1.35, absorption_length=60.0, scattering_length_eff=300.0, g=0.8, background_rate=50_000.0
    ),
    "ice": Medium(
        refractive_index=1.32, absorption_length=100.0, scattering_length_eff=25.0, g=0.8, background_rate=500.0
    ),
}


def medium(name: str = "water", **overrides: Any) -> Medium:
    """Return a medium preset (``"water"`` or ``"ice"``), optionally with values changed."""
    if name not in MEDIA:
        raise ValueError(f"unknown medium {name!r}; choose from {', '.join(MEDIA)}")
    return dataclasses.replace(MEDIA[name], **overrides) if overrides else MEDIA[name]


def medium_name(value: Medium) -> str:
    """Name of the medium preset equal to ``value`` (``"water"``, ``"ice"``) or ``"custom"``."""
    for name, candidate in MEDIA.items():
        if candidate == value:
            return name
    return "custom"


@dataclass(frozen=True)
class SimConfig:
    """Complete configuration of a simulation (everything except the seed, the device and the run options).

    Parameters
    ----------
    layout, medium, readout, physics, emission, prior
        The six parts; each defaults to the ``"full"`` preset in water.
    custom_geometry
        An explicit module table (:class:`~nutel.toy.geometry.Geometry`) that replaces the layout, or ``None``.

    Examples
    --------
    >>> config = SimConfig().replace(scattering=False, n_strings=6)
    >>> config.physics.scattering, config.layout.n_strings
    (False, 6)
    """

    layout: Layout = field(default_factory=Layout)
    medium: Medium = field(default_factory=Medium)
    readout: Readout = field(default_factory=Readout)
    physics: Physics = field(default_factory=Physics)
    emission: Emission = field(default_factory=Emission)
    prior: Prior = field(default_factory=Prior)
    custom_geometry: Geometry | None = None

    SECTIONS: ClassVar[tuple[str, ...]] = ("layout", "medium", "readout", "physics", "emission", "prior")

    def __post_init__(self) -> None:
        for name, cls in zip(self.SECTIONS, _PART_CLASSES, strict=True):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"SimConfig.{name} must be a {cls.__name__}, got {getattr(self, name)!r}")
        if self.custom_geometry is not None:
            from .geometry import Geometry

            if not isinstance(self.custom_geometry, Geometry):
                raise TypeError(f"SimConfig.custom_geometry must be a Geometry, got {self.custom_geometry!r}")

    # ------------------------------------------------------------------------------------------------------ derived
    def geometry(self) -> Geometry:
        """Return the module table: ``custom_geometry`` if set, else built from :attr:`layout`."""
        from .geometry import make_geometry

        return self.custom_geometry if self.custom_geometry is not None else make_geometry(self.layout)

    def detector(self) -> Detector:
        """Return the detector of this configuration: geometry, medium and readout (noise rate resolved)."""
        from .data import Detector

        return Detector(geometry=self.geometry(), medium=self.medium, readout=self.effective_readout())

    def effective_readout(self, readout: Readout | None = None, medium: Medium | None = None) -> Readout:
        """Return the readout with ``noise_rate`` resolved (``None`` -> the medium's ``background_rate``)."""
        r = self.readout if readout is None else readout
        m = self.medium if medium is None else medium
        return r if r.noise_rate is not None else dataclasses.replace(r, noise_rate=m.background_rate)

    def effective_medium(self, value: Medium | None = None) -> Medium:
        """Return the medium as the transport sees it: lengths of switched-off processes set to ``inf``."""
        m = self.medium if value is None else value
        return dataclasses.replace(
            m,
            absorption_length=m.absorption_length if self.physics.absorption else math.inf,
            scattering_length_eff=m.scattering_length_eff if self.physics.scattering else math.inf,
        )

    # ---------------------------------------------------------------------------------------------------- changes
    def replace(self, **overrides: Any) -> SimConfig:
        """Return a copy with some parameters changed.

        Parameters
        ----------
        **overrides
            Flat parameter names of any part (e.g. ``absorption=False``, ``noise_rate=0``, ``n_strings=6``), a whole
            part (e.g. ``readout=Readout(time_jitter=0.0)``), ``medium="ice"`` (a preset name), or
            ``custom_geometry=Geometry(...)``. ``layout`` is the layout name (a string) or a whole
            :class:`Layout`.

        Returns
        -------
        SimConfig
            The modified copy (the original is unchanged).
        """
        parts: dict[str, Any] = {name: getattr(self, name) for name in self.SECTIONS}
        parts["custom_geometry"] = self.custom_geometry
        changes: dict[str, dict[str, Any]] = {name: {} for name in self.SECTIONS}
        for key, value in overrides.items():
            part_cls = _PART_CLASSES[self.SECTIONS.index(key)] if key in self.SECTIONS else None
            if key == "medium" and isinstance(value, str):
                parts["medium"] = medium(value)
            elif key == "custom_geometry" or (part_cls is not None and isinstance(value, part_cls)):
                parts[key] = value
            elif key in _PARAMETER_PART:
                changes[_PARAMETER_PART[key]][key] = value
            else:
                close = difflib.get_close_matches(key, list(_PARAMETER_PART), n=3)
                hint = f" Did you mean {' or '.join(map(repr, close))}?" if close else ""
                raise TypeError(f"unknown configuration parameter {key!r}.{hint}")
        for name, kwargs in changes.items():
            if kwargs:
                parts[name] = dataclasses.replace(parts[name], **kwargs)
        return SimConfig(**parts)

    # --------------------------------------------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        """Nested dictionary of plain Python values (JSON-serialisable)."""
        out: dict[str, Any] = {name: getattr(self, name).to_dict() for name in self.SECTIONS}
        out["custom_geometry"] = None if self.custom_geometry is None else self.custom_geometry.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimConfig:
        """Inverse of :meth:`to_dict`; missing parameters take their default values."""
        unknown = set(data) - {*cls.SECTIONS, "custom_geometry"}
        if unknown:
            raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
        parts: dict[str, Any] = {
            name: part_cls.from_dict(data.get(name) or {})
            for name, part_cls in zip(cls.SECTIONS, _PART_CLASSES, strict=True)
        }
        if data.get("custom_geometry") is not None:
            from .geometry import Geometry

            parts["custom_geometry"] = Geometry.from_dict(data["custom_geometry"])
        return cls(**parts)

    def to_json(self, indent: int | None = None) -> str:
        """JSON string of :meth:`to_dict` (stored in every data file; ``inf`` is written as ``Infinity``)."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> SimConfig:
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        """Human-readable description of the configuration (used by ``nutelsim presets`` and ``nutelsim info``)."""
        lay, med, r, ph, em, pr = self.layout, self.medium, self.readout, self.physics, self.emission, self.prior
        if self.custom_geometry is not None:
            g = self.custom_geometry
            layout = f"custom table: {g.n_strings} strings, {g.n_modules} modules of radius {g.radius:g} m"
        else:
            layout = (
                f"{lay.layout}: {lay.n_strings} strings x {lay.modules_per_string} modules, d_s = "
                f"{lay.string_spacing:g} m, d_m = {lay.module_spacing:g} m, r_m = {lay.module_radius:g} m"
            )
        absorption = f"lambda_a = {med.absorption_length:g} m" if ph.absorption else "off"
        scattering = (
            f"lambda_eff = {med.scattering_length_eff:g} m (lambda_s = {med.scattering_length:g} m, g = {med.g:g})"
            if ph.scattering
            else "off"
        )
        stochastic = f"b = {em.stochastic_b:g} /m, v_min = {em.stochastic_v_min:g}" if ph.stochastic else "off"
        noise = f"{med.background_rate:g} Hz (from the medium)" if r.noise_rate is None else f"{r.noise_rate:g} Hz"
        cascade = em.cascade_profile + (f" ({em.cascade_spread:g} deg)" if em.cascade_profile == "gaussian" else "")
        if pr.event_type is not None:
            types = f"only {pr.event_type}"
        else:
            types = ", ".join(f"{t} {f:g}" for t, f in zip(EVENT_TYPES, pr.type_fractions, strict=True))
        return "\n".join(
            [
                f"layout:     {layout}",
                f"medium:     {medium_name(med)}: n = {med.refractive_index:g} "
                f"(theta_c = {math.degrees(med.cherenkov_angle):.1f} deg); absorption {absorption}; "
                f"scattering {scattering}",
                f"emission:   Y_t = {em.track_yield:g} /m, Y_c = {em.cascade_yield:g} /GeV ({em.photon_statistics}); "
                f"track spread {em.cherenkov_spread:g} deg; cascade profile {cascade}",
                f"muon:       continuous loss a = {em.energy_loss_a:g} GeV/m, stops below {em.muon_stop_energy:g} GeV; "
                f"stochastic losses {stochastic}",
                f"readout:    efficiency {r.efficiency:g}, jitter {r.time_jitter:g} ns, noise {noise}, "
                f"window [{r.window[0]:g}, {r.window[1]:g}) ns, merge {r.merge_window:g} ns, pulse time {r.pulse_time}",
                f"prior:      {types}; t0 in [{pr.t0_range[0]:g}, {pr.t0_range[1]:g}] ns",
                f"transport:  at most {ph.max_steps} steps per photon; world margin "
                f"min({ph.world_margin_factor:g} lambda_a, {ph.world_margin_max:g} m)",
            ]
        )


_PART_CLASSES: tuple[type[Any], ...] = (Layout, Medium, Readout, Physics, Emission, Prior)
_PARAMETER_PART: dict[str, str] = {}  # parameter name -> name of the part that owns it
for _part, _cls in zip(SimConfig.SECTIONS, _PART_CLASSES, strict=True):
    for _f in dataclasses.fields(_cls):
        if _f.name in _PARAMETER_PART:
            raise RuntimeError(f"duplicate parameter name {_f.name!r}")
        _PARAMETER_PART[_f.name] = _part


def config_fields() -> Iterator[tuple[str, dataclasses.Field[Any]]]:
    """Iterate over ``(part name, dataclass field)`` for every parameter, in definition order.

    The field metadata hold ``help``, ``unit`` and, where relevant, ``choices``, ``nargs``, ``metavar``,
    ``optional`` and ``allow_inf`` (used by the command line to build one flag per parameter).
    """
    for name, cls in zip(SimConfig.SECTIONS, _PART_CLASSES, strict=True):
        for f in dataclasses.fields(cls):
            yield name, f


def _ladder(absorption: bool, scattering: bool, stochastic: bool, readout: bool) -> SimConfig:
    ideal = Readout(efficiency=1.0, time_jitter=0.0, noise_rate=0.0)
    return SimConfig(
        physics=Physics(absorption=absorption, scattering=scattering, stochastic=stochastic),
        readout=Readout() if readout else ideal,
    )


#: The cumulative physics ladder: each preset switches on more physics (the medium is chosen separately).
PRESETS: dict[str, SimConfig] = {
    "straight": _ladder(absorption=False, scattering=False, stochastic=False, readout=False),
    "absorb": _ladder(absorption=True, scattering=False, stochastic=False, readout=False),
    "scatter": _ladder(absorption=True, scattering=True, stochastic=False, readout=False),
    "full": _ladder(absorption=True, scattering=True, stochastic=True, readout=True),
}

_PRESET_TEXT = {
    "straight": "Cherenkov emission, straight photons, ideal readout",
    "absorb": "+ absorption",
    "scatter": "+ scattering",
    "full": "+ readout effects (jitter, noise of the medium) and stochastic losses",
}


def preset(name: str = "full", medium: str | Medium = "water", **overrides: Any) -> SimConfig:
    """Return the configuration of a ladder preset in a medium, optionally with parameters changed on top.

    Parameters
    ----------
    name
        One of ``"straight"``, ``"absorb"``, ``"scatter"``, ``"full"`` (see :data:`PRESETS`).
    medium
        ``"water"``, ``"ice"`` (see :data:`MEDIA`) or a :class:`Medium`.
    **overrides
        Parameters to change, as for :meth:`SimConfig.replace`.

    Returns
    -------
    SimConfig

    Examples
    --------
    >>> preset("scatter", medium="ice", noise_rate=20e3).readout.noise_rate
    20000.0
    """
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    return PRESETS[name].replace(medium=medium, **overrides)


def matching_preset(config: SimConfig) -> str | None:
    """Name of the ladder preset of ``config`` (ignoring the medium), or ``None`` if it was modified."""
    for name, candidate in PRESETS.items():
        if candidate.replace(medium=config.medium) == config:
            return name
    return None


def describe_presets() -> str:
    """Table of the ladder presets and the media (used by ``nutelsim presets``)."""
    lines = ["presets (cumulative physics ladder; the medium is orthogonal):", ""]
    lines.append(
        f"  {'preset':<9} {'absorption':<11} {'scattering':<11} {'stochastic':<11} {'jitter':<7} "
        f"{'efficiency':<11} {'noise':<16}"
    )
    for name, c in PRESETS.items():
        ph, r = c.physics, c.readout
        noise = "from the medium" if r.noise_rate is None else f"{r.noise_rate:g} Hz"
        lines.append(
            f"  {name:<9} {'on' if ph.absorption else 'off':<11} {'on' if ph.scattering else 'off':<11} "
            f"{'on' if ph.stochastic else 'off':<11} {r.time_jitter:<7g} {r.efficiency:<11g} {noise:<16}"
            f"   {_PRESET_TEXT[name]}"
        )
    lines += ["", "media:"]
    for name, m in MEDIA.items():
        lines.append(
            f"  {name:<6} n = {m.refractive_index:g}, lambda_a = {m.absorption_length:g} m, "
            f"lambda_eff = {m.scattering_length_eff:g} m, g = {m.g:g} (lambda_s = {m.scattering_length:g} m), "
            f"noise (background) {m.background_rate:g} Hz"
        )
    lines += ["", "default configuration (full, water):", SimConfig().describe()]
    return "\n".join(line.rstrip() for line in lines)
