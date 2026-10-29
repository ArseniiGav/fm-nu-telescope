"""Containers of simulated data: what :func:`~nutel.toy.generate` returns and :func:`~nutel.toy.load` reads back.

The attributes mirror the groups of the HDF5 file::

    data.truth       event_id, geometry_index, type, direction, vertex, t0, energies, ...   one entry per event
    data.photons     photon bookkeeping (emitted, detected, absorbed, escaped, noise, ...)  one entry per event
    data.pulses      pulses (merged hits) of every event, ragged with offsets                always
    data.hits        detected photons and noise hits of every event, ragged with offsets     optional (level)
    data.bursts      stochastic losses of every event, ragged with offsets                   optional
    data.paths       photon polylines of the first events                                    optional (save_paths)
    data.detectors   one Detector (geometry, medium, readout) per geometry
    data.config      the SimConfig of the run
    data.provenance  seed, device, chunking, versions, creation time, ...

Ragged arrays: the entries of event ``i`` are ``offset[i]:offset[i + 1]`` (``offset`` has ``n_events + 1``
entries). Positions are (x, z) in metres, times in nanoseconds, angles in radians, energies in GeV.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import IntEnum
from typing import Any, ClassVar, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from .config import EVENT_TYPES, SOURCES, Medium, Readout, SimConfig, medium_name
from .geometry import Geometry

__all__ = [
    "EVENT_TYPES",
    "POINT_TYPES",
    "SOURCES",
    "Bursts",
    "Detector",
    "Event",
    "Hits",
    "Paths",
    "PhotonCounts",
    "PointType",
    "Pulses",
    "SimData",
    "Truth",
    "ragged_select",
]


class PointType(IntEnum):
    """Type of a point of a photon path; the type of the last point is the photon's fate."""

    EMIT = 0  #: emission
    SCATTER = 1  #: scattering in the medium
    DETECT = 2  #: detected by a module
    ABSORB_MODULE = 3  #: reached a module but not detected (efficiency < 1)
    ABSORB = 4  #: absorbed in the medium
    ESCAPE = 5  #: left the world box
    LOST = 6  #: still in flight after max_steps steps


#: Codes of the point types of :class:`Paths` (``{"EMIT": 0, ...}``), also stored in the data files.
POINT_TYPES: dict[str, int] = {member.name: int(member) for member in PointType}

_T = TypeVar("_T", bound="_Arrays")


class _Arrays:
    """Shared helpers of the array containers."""

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the arrays of this container by name."""
        return {f.name: getattr(self, f.name) for f in fields(self)}  # type: ignore[arg-type]

    @classmethod
    def from_arrays(cls: type[_T], arrays: Mapping[str, ArrayLike]) -> _T:
        """Build the container from arrays by name (converted to the dtypes of :attr:`DTYPES`)."""
        dtypes: dict[str, Any] = cls.DTYPES  # type: ignore[attr-defined]
        missing = set(dtypes) - set(arrays)
        if missing:
            raise ValueError(f"{cls.__name__}: missing arrays {sorted(missing)}")
        return cls(**{name: np.ascontiguousarray(arrays[name], dtype=dt) for name, dt in dtypes.items()})


def _dtypes(cls: type[Any]) -> dict[str, Any]:
    return {f.name: f.metadata["dtype"] for f in fields(cls)}


def _a(dtype: Any, doc: str) -> Any:
    """Array field with its dtype and a description (unit in brackets)."""
    return field(metadata={"dtype": np.dtype(dtype), "doc": doc})


# ----------------------------------------------------------------------------------------------------------------------
# Per-event tables


@dataclass(frozen=True)
class Truth(_Arrays):
    """True parameters of every event (arrays of length ``n_events``).

    ``x0, z0`` is the anchor point of a through-going track (the point of the line closest to the origin) or the
    vertex of a starting track or cascade; ``t0`` is the time at that point. ``energy`` is the muon energy for
    ``track`` and ``atmospheric``, the neutrino energy for ``starting`` and the cascade energy for ``cascade``.
    The muon emits light on the segment ``x0 + s d`` with ``s_start <= s <= s_end`` (``track_length = s_end -
    s_start``): from its entry into the generation box (``s_start = 0`` at the vertex of a starting track) to its exit
    or to the point where it stops; ``s_start`` and ``s_end`` are NaN where no muon light is emitted (cascades, lines
    that miss the generation box).
    """

    event_id: np.ndarray = _a(np.int64, "event number, unique within a data set")
    geometry_index: np.ndarray = _a(np.int32, "index of the event's detector in data.detectors")
    type: np.ndarray = _a(np.int8, "event type code: 0 track, 1 starting, 2 cascade, 3 atmospheric")
    dir_x: np.ndarray = _a(np.float32, "direction of travel, x component")
    dir_z: np.ndarray = _a(np.float32, "direction of travel, z component (z up)")
    zenith: np.ndarray = _a(np.float32, "zenith = arccos(-dir_z) [rad]: 0 down-going, pi up-going")
    azimuth: np.ndarray = _a(np.float32, "azimuth [rad]: 0 if coming from +x (dir_x < 0), pi if from -x")
    angle: np.ndarray = _a(np.float32, "direction angle atan2(dir_z, dir_x) [rad] in (-pi, pi]")
    x0: np.ndarray = _a(np.float32, "anchor point or vertex, x [m]")
    z0: np.ndarray = _a(np.float32, "anchor point or vertex, z [m]")
    t0: np.ndarray = _a(np.float32, "time at the anchor point or vertex [ns]")
    energy: np.ndarray = _a(np.float32, "event energy [GeV] (muon, neutrino or cascade energy, see above)")
    energy_track: np.ndarray = _a(np.float32, "muon energy at the start of its light emission [GeV] (0 for cascade)")
    energy_cascade: np.ndarray = _a(np.float32, "energy of the vertex cascade [GeV] (0 for track, atmospheric)")
    cascade_fraction: np.ndarray = _a(np.float32, "inelasticity y of starting tracks (NaN for the other types)")
    track_length: np.ndarray = _a(np.float32, "length of the light-emitting muon segment [m] (0 for cascade)")
    s_start: np.ndarray = _a(
        np.float32, "start of the light-emitting muon segment along x0 + s d [m] (NaN if the muon emits no light)"
    )
    s_end: np.ndarray = _a(
        np.float32, "end of the light-emitting segment along x0 + s d [m] (the stop point of a stopped muon; or NaN)"
    )
    length_in_area: np.ndarray = _a(np.float32, "length of the emitting segment inside the instrumented area [m]")
    n_bursts: np.ndarray = _a(np.int32, "number of stochastic-loss bursts")
    energy_lost: np.ndarray = _a(np.float32, "sum of the burst energies [GeV]")
    stopped: np.ndarray = _a(np.bool_, "True if the muon stopped inside the generation box (ran out of energy)")

    DTYPES: ClassVar[dict[str, Any]] = {}  # dtypes by name, set below

    def __len__(self) -> int:
        return len(self.event_id)

    @property
    def type_name(self) -> np.ndarray:
        """Event type names (array of str)."""
        return np.asarray(EVENT_TYPES)[self.type.astype(np.int64)]

    def row(self, i: int) -> dict[str, Any]:
        """All truth values of event ``i`` as Python scalars."""
        return {name: value[i].item() for name, value in self.arrays().items()}


@dataclass(frozen=True)
class PhotonCounts(_Arrays):
    """Photon bookkeeping of every event (int32 arrays of length ``n_events``).

    Every emitted photon ends in exactly one way, and every hit is a detected photon inside the window or noise::

        n_emitted = n_emitted_track + n_emitted_burst + n_emitted_cascade
        n_emitted = n_detected + n_absorbed_module + n_absorbed + n_escaped + n_lost
        n_detected = n_detected_direct + n_detected_scattered
        n_hits = n_detected - n_out_of_window + n_noise
    """

    n_emitted: np.ndarray = _a(np.int32, "photons emitted")
    n_emitted_track: np.ndarray = _a(np.int32, "photons emitted by the continuous muon light")
    n_emitted_burst: np.ndarray = _a(np.int32, "photons emitted by stochastic-loss bursts")
    n_emitted_cascade: np.ndarray = _a(np.int32, "photons emitted by the vertex cascade")
    n_detected: np.ndarray = _a(np.int32, "photons detected by a module (before the window cut)")
    n_detected_direct: np.ndarray = _a(np.int32, "detected photons that never scattered")
    n_detected_scattered: np.ndarray = _a(np.int32, "detected photons that scattered at least once")
    n_absorbed_module: np.ndarray = _a(np.int32, "photons that reached a module but were not detected")
    n_absorbed: np.ndarray = _a(np.int32, "photons absorbed in the medium")
    n_escaped: np.ndarray = _a(np.int32, "photons that left the world box")
    n_lost: np.ndarray = _a(np.int32, "photons still in flight after max_steps steps")
    n_noise: np.ndarray = _a(np.int32, "noise hits")
    n_out_of_window: np.ndarray = _a(np.int32, "detected photons outside the readout window (dropped)")
    n_hits: np.ndarray = _a(np.int32, "hits: detected photons in the window plus noise hits")
    n_pulses: np.ndarray = _a(np.int32, "pulses")

    DTYPES: ClassVar[dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.n_emitted)


# ----------------------------------------------------------------------------------------------------------------------
# Ragged tables


def ragged_select(offset: np.ndarray, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Entries of the selected events of a ragged table.

    Parameters
    ----------
    offset
        Offsets of the table, ``[n_events + 1]``.
    index
        Selected events (integer array, any order).

    Returns
    -------
    entries : numpy.ndarray
        Rows of the table to take, event by event in the order of ``index`` (int64).
    new_offset : numpy.ndarray
        Offsets of the selection, ``[len(index) + 1]`` (int64).
    """
    offset = np.asarray(offset, dtype=np.int64)
    index = np.asarray(index, dtype=np.int64)
    counts = offset[index + 1] - offset[index]
    new_offset = np.zeros(len(index) + 1, dtype=np.int64)
    np.cumsum(counts, out=new_offset[1:])
    start = np.repeat(offset[index] - new_offset[:-1], counts)
    entries = np.arange(new_offset[-1], dtype=np.int64) + start
    return entries, new_offset


class _Ragged(_Arrays):
    """Ragged per-event table: the entries of event ``i`` are ``offset[i]:offset[i + 1]``."""

    offset: np.ndarray

    def __len__(self) -> int:
        return int(self.offset[-1])

    @property
    def n_events(self) -> int:
        """Number of events."""
        return len(self.offset) - 1

    def counts(self) -> np.ndarray:
        """Return the number of entries of every event (int64)."""
        return np.diff(self.offset)

    def event_index(self) -> np.ndarray:
        """Index of the event of every entry (int64)."""
        return np.repeat(np.arange(self.n_events, dtype=np.int64), self.counts())

    def select_events(self: Any, index: ArrayLike) -> Any:
        """Table of the selected events (integer indices, any order)."""
        entries, new_offset = ragged_select(self.offset, np.asarray(index))
        values = {name: value[entries] for name, value in self.arrays().items() if name != "offset"}
        return type(self)(offset=new_offset, **values)

    def of_event(self: Any, i: int) -> Any:
        """Table of event ``i`` alone (``offset = [0, n]``)."""
        return self.select_events(np.array([i]))


@dataclass(frozen=True)
class Hits(_Ragged):
    """Every hit (detected photon or noise hit) of every event, sorted by (event, t).

    ``source``: 0 track, 1 burst, 2 cascade, 3 noise. Noise hits have ``n_scatter = -1`` and ``path_length = NaN``.
    """

    offset: np.ndarray = _a(np.int64, "offsets [n_events + 1]")
    string: np.ndarray = _a(np.int32, "string number")
    module: np.ndarray = _a(np.int32, "module number on the string (0 = deepest)")
    x: np.ndarray = _a(np.float32, "module position x [m]")
    z: np.ndarray = _a(np.float32, "module position z [m]")
    t: np.ndarray = _a(np.float32, "hit time [ns]")
    source: np.ndarray = _a(np.int8, "light source: 0 track, 1 burst, 2 cascade, 3 noise")
    n_scatter: np.ndarray = _a(np.int32, "number of scatterings of the photon (-1 for noise)")
    path_length: np.ndarray = _a(np.float32, "path length of the photon [m] (NaN for noise)")
    is_noise: np.ndarray = _a(np.bool_, "True for noise hits")

    DTYPES: ClassVar[dict[str, Any]] = {}


@dataclass(frozen=True)
class Pulses(_Ragged):
    """Pulses of every event, sorted by (event, t): hits of a module merged within the merge window.

    Per module, a pulse starts at the earliest unassigned hit and takes every hit with ``t <= t_start + tau``; its
    time is the time of its first hit (or the mean time of its hits with ``readout.pulse_time = "mean"``), its charge
    the number of hits.
    """

    offset: np.ndarray = _a(np.int64, "offsets [n_events + 1]")
    string: np.ndarray = _a(np.int32, "string number")
    module: np.ndarray = _a(np.int32, "module number on the string (0 = deepest)")
    x: np.ndarray = _a(np.float32, "module position x [m]")
    z: np.ndarray = _a(np.float32, "module position z [m]")
    t: np.ndarray = _a(np.float32, "pulse time: first or mean hit time (readout.pulse_time) [ns]")
    charge: np.ndarray = _a(np.int32, "number of merged hits")
    n_noise: np.ndarray = _a(np.int32, "number of noise hits among them")

    DTYPES: ClassVar[dict[str, Any]] = {}


@dataclass(frozen=True)
class Bursts(_Ragged):
    """Stochastic-loss bursts of every event (point-like cascades along the muon), ordered along the track."""

    offset: np.ndarray = _a(np.int64, "offsets [n_events + 1]")
    x: np.ndarray = _a(np.float32, "position x [m]")
    z: np.ndarray = _a(np.float32, "position z [m]")
    t: np.ndarray = _a(np.float32, "time [ns]")
    energy: np.ndarray = _a(np.float32, "energy of the burst [GeV]")

    DTYPES: ClassVar[dict[str, Any]] = {}


@dataclass(frozen=True)
class Paths(_Arrays):
    """Photon paths (polylines) of the first events, flattened.

    Path ``k`` has the points ``offset[k]:offset[k + 1]``; each point has a type code ``kind`` (:class:`PointType`).
    The first point of a path is its emission point, the last one its fate.

    Examples
    --------
    >>> for k in data.paths.of_event(0):  # doctest: +SKIP
    ...     x, z, t, kind = data.paths.polyline(k)
    ...     plt.plot(x, z)
    """

    event_id: np.ndarray = _a(np.int64, "event of each path, [n_paths]")
    photon: np.ndarray = _a(np.int32, "number of the photon within its event, [n_paths]")
    source: np.ndarray = _a(np.int8, "light source of the photon: 0 track, 1 burst, 2 cascade, [n_paths]")
    offset: np.ndarray = _a(np.int64, "start of each path in the point arrays, [n_paths + 1]")
    x: np.ndarray = _a(np.float32, "point x [m], [n_points]")
    z: np.ndarray = _a(np.float32, "point z [m], [n_points]")
    t: np.ndarray = _a(np.float32, "time at the point [ns], [n_points]")
    kind: np.ndarray = _a(np.int8, "PointType code of the point, [n_points]")

    DTYPES: ClassVar[dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.event_id)

    def polyline(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Coordinates, times and point types ``(x, z, t, kind)`` of path ``k``."""
        part = slice(int(self.offset[k]), int(self.offset[k + 1]))
        return self.x[part], self.z[part], self.t[part], self.kind[part]

    def of_event(self, event_id: int) -> np.ndarray:
        """Return the indices of the paths of one event (by event id)."""
        return np.flatnonzero(self.event_id == event_id)

    @property
    def fate(self) -> np.ndarray:
        """PointType code of the last point of every path (int8)."""
        return self.kind[self.offset[1:] - 1]

    @property
    def n_scatter(self) -> np.ndarray:
        """Number of scatterings of every path (int64)."""
        is_scatter = (self.kind == PointType.SCATTER).astype(np.int64)
        cum = np.concatenate([[0], np.cumsum(is_scatter)])
        return cum[self.offset[1:]] - cum[self.offset[:-1]]

    def select_paths(self, keep: ArrayLike) -> Paths:
        """Paths with ``keep`` True (boolean mask over paths)."""
        index = np.flatnonzero(np.asarray(keep, dtype=bool))
        entries, new_offset = ragged_select(self.offset, index)
        return Paths(
            event_id=self.event_id[index],
            photon=self.photon[index],
            source=self.source[index],
            offset=new_offset,
            x=self.x[entries],
            z=self.z[entries],
            t=self.t[entries],
            kind=self.kind[entries],
        )


for _cls in (Truth, PhotonCounts, Hits, Pulses, Bursts, Paths):
    _cls.DTYPES = _dtypes(_cls)


# ----------------------------------------------------------------------------------------------------------------------
# Detector, event view, data set


@dataclass(frozen=True)
class Detector:
    """One detector: the module table, the medium and the readout.

    In a multi-detector data set every geometry has its own medium and readout (sampled from a
    :class:`~nutel.toy.family.DetectorFamily`); the physics switches are those of the run's configuration.
    """

    geometry: Geometry
    medium: Medium
    readout: Readout

    @property
    def geometry_id(self) -> str:
        """Identifier of the geometry."""
        return self.geometry.geometry_id

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python dictionary (JSON-serialisable)."""
        return {
            "geometry": self.geometry.to_dict(),
            "medium": self.medium.to_dict(),
            "readout": self.readout.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Detector:
        """Inverse of :meth:`to_dict`."""
        return cls(
            geometry=Geometry.from_dict(data["geometry"]),
            medium=Medium.from_dict(data["medium"]),
            readout=Readout.from_dict(data["readout"]),
        )

    def describe(self) -> str:
        """Short description: geometry, medium and readout."""
        m, r = self.medium, self.readout
        noise = f"{m.background_rate:.0f} Hz (from the medium)" if r.noise_rate is None else f"{r.noise_rate:.0f} Hz"
        return "\n".join(
            [
                self.geometry.describe(),
                f"  medium ({medium_name(m)}): n = {m.refractive_index:.3f}, lambda_a = {m.absorption_length:.1f} m, "
                f"lambda_eff = {m.scattering_length_eff:.1f} m, g = {m.g:.2f}",
                f"  readout: efficiency {r.efficiency:.2f}, jitter {r.time_jitter:.2f} ns, noise {noise}, "
                f"window [{r.window[0]:g}, {r.window[1]:g}) ns, merge {r.merge_window:g} ns, pulse time {r.pulse_time}",
            ]
        )


@dataclass(frozen=True)
class Event:
    """View of one event: its truth, detector, hits, pulses and bursts (tables with ``offset = [0, n]``)."""

    index: int
    truth: dict[str, Any]
    detector: Detector
    pulses: Pulses
    hits: Hits | None
    bursts: Bursts | None

    @property
    def geometry(self) -> Geometry:
        """Module table of the event's detector."""
        return self.detector.geometry

    @property
    def type_name(self) -> str:
        """Event type name."""
        return EVENT_TYPES[int(self.truth["type"])]


@dataclass
class SimData:
    """Simulated events: truth, photon bookkeeping, pulses, optional hits, bursts and photon paths, detectors.

    Attributes
    ----------
    truth : Truth
        True event parameters (one entry per event).
    photons : PhotonCounts
        Photon bookkeeping per event.
    pulses : Pulses
        Pulses of every event (ragged).
    detectors : list of Detector
        One detector (geometry, medium, readout) per geometry; ``truth.geometry_index`` points into this list.
    config : SimConfig
        Configuration of the run (in multi-detector runs the medium and readout of each detector come from
        ``detectors``).
    provenance : dict
        Seed, device, chunking, versions, creation time, wall time, command line, family.
    hits : Hits or None
        Every hit (level ``"hits"`` or ``"both"``).
    bursts : Bursts or None
        Stochastic-loss bursts of every event.
    paths : Paths or None
        Photon paths of the first events (``save_paths > 0``).
    """

    truth: Truth
    photons: PhotonCounts
    pulses: Pulses
    detectors: list[Detector]
    config: SimConfig = field(default_factory=SimConfig)
    provenance: dict[str, Any] = field(default_factory=dict)
    hits: Hits | None = None
    bursts: Bursts | None = None
    paths: Paths | None = None

    def __len__(self) -> int:
        return len(self.truth)

    @property
    def n_events(self) -> int:
        """Number of events."""
        return len(self)

    @property
    def n_geometries(self) -> int:
        """Number of geometries (detectors)."""
        return len(self.detectors)

    @property
    def geometries(self) -> list[Geometry]:
        """Module tables of the detectors, in the order of ``geometry_index``."""
        return [d.geometry for d in self.detectors]

    def geometry(self, i: int) -> Geometry:
        """Return the module table of the detector of event ``i``."""
        return self.detectors[int(self.truth.geometry_index[i])].geometry

    def detector(self, i: int) -> Detector:
        """Detector (geometry, medium, readout) of event ``i``."""
        return self.detectors[int(self.truth.geometry_index[i])]

    def event(self, i: int) -> Event:
        """View of event ``i`` (index into the data set, not the event id)."""
        n = len(self)
        if not -n <= i < n:
            raise IndexError(f"event index {i} out of range for {n} events")
        i = int(i) % n
        return Event(
            index=i,
            truth=self.truth.row(i),
            detector=self.detector(i),
            pulses=self.pulses.of_event(i),
            hits=None if self.hits is None else self.hits.of_event(i),
            bursts=None if self.bursts is None else self.bursts.of_event(i),
        )

    def select(self, mask: ArrayLike) -> SimData:
        """Subset of the events: a boolean mask over the events or an array of distinct event indices (any order).

        The detectors are kept (``geometry_index`` stays valid); photon paths are kept for the selected events.
        """
        sel = np.asarray(mask)
        index = np.flatnonzero(sel) if sel.dtype == bool else sel.astype(np.int64).ravel()
        if sel.dtype == bool and len(sel) != len(self):
            raise ValueError(f"mask has {len(sel)} entries for {len(self)} events")
        if len(np.unique(index)) != len(index):
            raise ValueError("repeated event indices: every event can be selected only once (event ids stay unique)")
        truth = Truth(**{k: v[index] for k, v in self.truth.arrays().items()})
        photons = PhotonCounts(**{k: v[index] for k, v in self.photons.arrays().items()})
        paths = None
        if self.paths is not None:
            paths = self.paths.select_paths(np.isin(self.paths.event_id, truth.event_id))
        return dataclasses.replace(
            self,
            truth=truth,
            photons=photons,
            pulses=self.pulses.select_events(index),
            hits=None if self.hits is None else self.hits.select_events(index),
            bursts=None if self.bursts is None else self.bursts.select_events(index),
            paths=paths,
            provenance=dict(self.provenance),
        )

    def __repr__(self) -> str:
        extras = [name for name in ("hits", "bursts", "paths") if getattr(self, name) is not None]
        return (
            f"SimData({len(self)} events, {self.n_geometries} geometr{'y' if self.n_geometries == 1 else 'ies'}, "
            f"{len(self.pulses)} pulses" + (f", with {', '.join(extras)}" if extras else "") + ")"
        )
