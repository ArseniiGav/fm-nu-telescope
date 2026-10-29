"""Reading, writing and merging simulated data: one HDF5 file per run, one or several detectors per file.

File layout (format version 1; ``N`` events, ``K`` detectors)::

    /             attrs: format_version ("1"), n_events, n_geometries, config (JSON), provenance (JSON)
    /geometries   offset [K + 1] (int64) into the concatenated module tables string, module (int32), x, z [m]
                  (float32); per detector: geometry_id (12 ascii characters), radius [m] (float64), layout, medium,
                  readout (JSON strings)                                                               shape [K]
    /truth        event_id (int64), geometry_index (int32), type (int8) and the true parameters      shape [N]
    /photons      photon bookkeeping per event (int32)                                                 shape [N]
    /pulses       offset [N + 1] (int64); string, module, x, z, t, charge, n_noise    (always)
    /hits         offset [N + 1] (int64); string, module, x, z, t, source, n_scatter, path_length, is_noise
                  (optional: level "hits" or "both")
    /bursts       offset [N + 1] (int64); x, z, t, energy                                (optional)
    /paths        event_id, photon, source [n_paths]; offset [n_paths + 1]; x, z, t, kind [n_points]
                  attrs: point_types (JSON, name -> code)                                (optional: save_paths K)

Every dataset carries ``units`` (empty for counts and codes) and ``description`` attributes. Files are written to
``<name>.part`` first and renamed when complete, so an interrupted job never leaves a truncated file under the final
name.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from .config import Medium, Readout, SimConfig, matching_preset, medium_name
from .data import POINT_TYPES, Bursts, Detector, Hits, Paths, PhotonCounts, Pulses, SimData, Truth
from .geometry import Geometry

__all__ = ["FORMAT_VERSION", "describe", "load", "merge", "save"]

#: Version of the file layout written by :func:`save` (stored as the string attribute ``format_version``).
FORMAT_VERSION = "1"

Level = Literal["pulses", "hits", "both"]

_GROUPS: dict[str, type[Any]] = {
    "truth": Truth,
    "photons": PhotonCounts,
    "pulses": Pulses,
    "hits": Hits,
    "bursts": Bursts,
    "paths": Paths,
}
_OPTIONAL = ("hits", "bursts", "paths")
_RAGGED = ("pulses", "hits", "bursts")  # groups with event offsets
_UNITS = {
    "x": "m",
    "z": "m",
    "x0": "m",
    "z0": "m",
    "t": "ns",
    "t0": "ns",
    "path_length": "m",
    "track_length": "m",
    "length_in_area": "m",
    "zenith": "rad",
    "azimuth": "rad",
    "angle": "rad",
    "energy": "GeV",
    "energy_track": "GeV",
    "energy_cascade": "GeV",
    "energy_lost": "GeV",
    "radius": "m",
}
_COPY_BLOCK = 1 << 22  # elements copied at a time when merging
_COMPRESS_ABOVE = 1024  # datasets with more elements are gzip-compressed


# ----------------------------------------------------------------------------------------------------------------------
# Public API


def save(data: SimData, path: str | os.PathLike[str], *, overwrite: bool = True) -> Path:
    """Write a data set to an HDF5 file.

    Parameters
    ----------
    data
        Output of :func:`~nutel.toy.generate` or :func:`load`.
    path
        Output file; missing parent directories are created.
    overwrite
        Replace an existing file (otherwise raise :class:`FileExistsError`).

    Returns
    -------
    pathlib.Path
        The path written.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        with h5py.File(part, "w") as f:
            _write_header(f, data.config.to_json(), data.provenance, len(data), len(data.detectors))
            _write_geometries(f, data.detectors)
            for name, cls in _GROUPS.items():
                container = getattr(data, name)
                if container is None:
                    continue
                group = f.create_group(name)
                docs = {fld.name: fld.metadata.get("doc", "") for fld in fields(cls)}
                for key, values in container.arrays().items():
                    _create(group, key, values, docs.get(key, ""))
            if data.paths is not None:
                f["paths"].attrs["point_types"] = json.dumps(POINT_TYPES)
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)
    return path


def load(path: str | os.PathLike[str], *, level: Level = "both", bursts: bool = True, paths: bool = True) -> SimData:
    """Read a data file written by :func:`save`, :func:`merge` or ``nutelsim``.

    Parameters
    ----------
    path
        HDF5 file.
    level
        ``"pulses"``: read the pulses only (the hits are skipped, saving memory); ``"both"`` (default): also the hits
        if the file has them; ``"hits"``: also the hits, which must be present.
    bursts, paths
        Read the optional ``/bursts`` and ``/paths`` groups if present.

    Returns
    -------
    SimData
        The stored events with their detectors, configuration and provenance.
    """
    if level not in ("pulses", "hits", "both"):
        raise ValueError(f"level must be 'pulses', 'hits' or 'both', got {level!r}")
    with _open(path) as f:
        _check_format(f, path)
        if level == "hits" and "hits" not in f:
            raise ValueError(f"{path} has no hit level (generate with --level hits or both)")
        wanted = {"hits": level != "pulses", "bursts": bursts, "paths": paths}
        groups: dict[str, Any] = {}
        for name, cls in _GROUPS.items():
            if name in f and wanted.get(name, True):
                groups[name] = cls.from_arrays({fld.name: f[name][fld.name][()] for fld in fields(cls)})
        return SimData(
            detectors=_read_geometries(f),
            config=SimConfig.from_json(_text(f.attrs["config"])),
            provenance=json.loads(_text(f.attrs["provenance"])),
            **groups,
        )


def merge(
    paths: Sequence[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    overwrite: bool = True,
    renumber: bool = False,
) -> Path:
    """Concatenate data files (e.g. cluster shards, or runs with different detectors).

    The configurations must agree except for the detectors (layout, custom table, medium and readout, which are
    stored per detector) and the seed. The detectors of all inputs are concatenated and de-duplicated (two detectors
    are the same if their geometry id, medium and readout agree), and ``geometry_index`` is remapped. The inputs
    must have the same optional groups and distinct seeds; event ids must be distinct (use ``first_event_id`` /
    ``--first-event-id`` to give shards disjoint ranges) unless ``renumber=True``, which numbers the events
    ``0 ... N - 1`` in the order of the inputs. Data are copied block by block, so the inputs need not fit in memory.

    Parameters
    ----------
    paths
        Input files, in the order in which their events are concatenated.
    out
        Output file.
    overwrite
        Replace an existing output file.
    renumber
        Renumber the event ids instead of requiring them to be distinct.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    sources, target = [Path(p) for p in paths], Path(out)
    if not sources:
        raise ValueError("no input files")
    if target.resolve() in {p.resolve() for p in sources}:
        raise ValueError(f"the output {target} is also an input")
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists (pass overwrite=True to replace it)")
    inputs = [_inspect(p) for p in sources]
    _check_mergeable(inputs, renumber)
    detectors: list[Detector] = []
    keys: dict[str, int] = {}
    remaps = []
    for item in inputs:
        remap = []
        for det in item["detectors"]:
            key = _detector_key(det)
            if key not in keys:
                keys[key] = len(detectors)
                detectors.append(det)
            remap.append(keys[key])
        remaps.append(np.asarray(remap, dtype=np.int32))
    from .. import __version__

    provenance = {
        "merged_from": [{"file": item["path"].name, **item["provenance"]} for item in inputs],
        "renumbered": renumber,
        "package_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    n_events = sum(item["n_events"] for item in inputs)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        with h5py.File(part, "w") as dst:
            _write_header(dst, inputs[0]["config"], provenance, n_events, len(detectors))
            _write_geometries(dst, detectors)
            sources_h5 = [h5py.File(item["path"], "r") for item in inputs]
            try:
                id_maps = _event_id_maps(inputs, renumber)
                for name in inputs[0]["groups"]:
                    _merge_group(dst, name, sources_h5, remaps, id_maps)
            finally:
                for src in sources_h5:
                    src.close()
            if "paths" in dst:
                dst["paths"].attrs["point_types"] = json.dumps(POINT_TYPES)
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    return target


def describe(path: str | os.PathLike[str]) -> str:
    """Human-readable summary of a data file: detectors, configuration, provenance, datasets and a few numbers."""
    path = Path(path)
    with _open(path) as f:
        _check_format(f, path)
        config = SimConfig.from_json(_text(f.attrs["config"]))
        provenance = json.loads(_text(f.attrs["provenance"]))
        n = int(f.attrs["n_events"])
        detectors = _read_geometries(f)
        lines = [f"file: {path}  ({path.stat().st_size / 1e6:.2f} MB)"]
        levels = [name for name in ("pulses", "hits") if name in f]
        extras = [name for name in ("bursts", "paths") if name in f]
        lines.append(
            f"format version {_text(f.attrs['format_version'])}, {n} events, {len(detectors)} detector(s), "
            f"levels: {' + '.join(levels)}" + (f"; also {', '.join(extras)}" if extras else "")
        )
        truth = f["truth"]
        gi = truth["geometry_index"][()]
        types = truth["type"][()]
        lines.append("")
        lines.append("detectors (events per detector):")
        counts = np.bincount(gi.astype(np.int64), minlength=len(detectors))
        sampled = provenance.get("family") is not None or any(
            item.get("family") is not None for item in provenance.get("merged_from", [])
        )
        configured = -1.0 if sampled else config.readout.noise_rate  # sampled detectors: never "from the medium"
        for k, det in enumerate(detectors):
            g, m, r = det.geometry, det.medium, det.readout
            lines.append(
                f"  [{k}] {g.geometry_id} {g.layout:<9s} {g.n_strings:3d} strings {g.n_modules:4d} modules  "
                f"{medium_name(m)}: n = {m.refractive_index:.3f}, lambda_a = {m.absorption_length:.0f} m, "
                f"lambda_eff = {m.scattering_length_eff:.0f} m; noise {_noise_text(r, m, configured)}, "
                f"efficiency {r.efficiency:.2f}, jitter {r.time_jitter:.2f} ns   ({counts[k]} events)"
            )
        from .config import EVENT_TYPES

        type_counts = np.bincount(types.astype(np.int64), minlength=len(EVENT_TYPES))
        lines.append("")
        lines.append(
            "event types: " + ", ".join(f"{name} {c}" for name, c in zip(EVENT_TYPES, type_counts, strict=False))
        )
        lines.append("")
        lines.append(f"configuration (preset: {matching_preset(config) or 'modified'}):")
        lines += ["  " + line for line in config.describe().splitlines()]
        lines.append("")
        lines.append("provenance:")
        for key, value in provenance.items():
            if key == "merged_from":
                lines.append(f"  merged from {len(value)} files: {', '.join(s['file'] for s in value)}")
                lines.append(f"  seeds: {list(_seeds(provenance))}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")
        lines.append("datasets:")

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                unit = obj.attrs.get("units", "")
                lines.append(f"  /{name:<30s} {obj.dtype!s:<8s} {obj.shape!s:<16s} {_text(unit)}".rstrip())

        f.visititems(visit)
        photons = f["photons"]
        emitted = int(photons["n_emitted"][()].sum())
        lines.append("")
        lines.append("per event (mean):")
        for key in ("n_emitted", "n_detected", "n_hits", "n_noise", "n_pulses"):
            lines.append(f"  {key[2:]:<18s} {photons[key][()].mean() if n else 0.0:10.1f}")
        if emitted > 0:
            lines.append("photon fates (fraction of the emitted photons):")
            for key in ("n_detected", "n_absorbed_module", "n_absorbed", "n_escaped", "n_lost"):
                lines.append(f"  {key[2:]:<18s} {photons[key][()].sum() / emitted:10.4f}")
    return "\n".join(lines)


def _noise_text(readout: Readout, medium: Medium, configured: float | None) -> str:
    """Format the resolved noise rate of a detector, marked ``(from the medium)`` where the medium sets it."""
    rate = medium.background_rate if readout.noise_rate is None else readout.noise_rate
    from_medium = readout.noise_rate is None or (configured is None and rate == medium.background_rate)
    return f"{rate:.0f} Hz" + (" (from the medium)" if from_medium else "")


# ----------------------------------------------------------------------------------------------------------------------
# Writing helpers


def _write_header(f: h5py.File, config_json: str, provenance: dict[str, Any], n_events: int, n_geometries: int) -> None:
    f.attrs["format_version"] = FORMAT_VERSION
    f.attrs["n_events"] = n_events
    f.attrs["n_geometries"] = n_geometries
    f.attrs["config"] = config_json
    f.attrs["provenance"] = json.dumps(provenance, default=str)


def _write_geometries(f: h5py.File, detectors: Sequence[Detector]) -> None:
    if not detectors:
        raise ValueError("a data set needs at least one detector")
    group = f.create_group("geometries")
    geometries = [d.geometry for d in detectors]
    sizes = [g.n_modules for g in geometries]
    _create(group, "offset", np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64), "start of each module table")
    for key, dtype, doc in (
        ("string", np.int32, "string number"),
        ("module", np.int32, "module number on the string (0 = deepest)"),
        ("x", np.float32, "module centre x"),
        ("z", np.float32, "module centre z"),
    ):
        _create(group, key, np.concatenate([getattr(g, key) for g in geometries]).astype(dtype), doc)
    ids = np.array([g.geometry_id for g in geometries], dtype="S12")
    _create(group, "geometry_id", ids, "first 12 hex characters of the SHA-1 of the module table")
    _create(group, "radius", np.array([g.radius for g in geometries], dtype=np.float64), "module radius")
    text = h5py.string_dtype()
    for key, values, doc in (
        ("layout", [json.dumps({"layout": g.layout, "params": g.params}) for g in geometries], "layout (JSON)"),
        ("medium", [json.dumps(d.medium.to_dict()) for d in detectors], "medium (JSON)"),
        ("readout", [json.dumps(d.readout.to_dict()) for d in detectors], "readout (JSON)"),
    ):
        dataset = group.create_dataset(key, data=np.array(values, dtype=object), dtype=text)
        dataset.attrs["units"] = ""
        dataset.attrs["description"] = doc


def _create(group: h5py.Group, name: str, values: np.ndarray, doc: str = "") -> h5py.Dataset:
    values = np.asarray(values)
    options: dict[str, Any] = {}
    if values.size > _COMPRESS_ABOVE:
        options = {"compression": "gzip", "compression_opts": 4, "shuffle": True, "chunks": True}
    dataset = group.create_dataset(name, data=values, **options)
    dataset.attrs["units"] = _UNITS.get(name, "")
    dataset.attrs["description"] = doc
    return dataset


# ----------------------------------------------------------------------------------------------------------------------
# Reading helpers


def _open(path: str | os.PathLike[str]) -> h5py.File:
    """Open a file for reading; a file that is not HDF5 (or a directory) gives a clear ``ValueError``."""
    try:
        return h5py.File(path, "r")
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError(f"{path} is not a readable HDF5 file ({error})") from error


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _check_format(f: h5py.File, path: str | os.PathLike[str]) -> None:
    version = f.attrs.get("format_version")
    if version is None or "geometries" not in f:
        raise ValueError(f"{path} is not a nutel toy data file (no format_version attribute or /geometries group)")
    if int(_text(version)) > int(FORMAT_VERSION):
        raise ValueError(f"{path} has format version {_text(version)}; this package reads up to {FORMAT_VERSION}")


def _read_geometries(f: h5py.File) -> list[Detector]:
    group = f["geometries"]
    offset = group["offset"][()]
    table = {key: group[key][()] for key in ("string", "module", "x", "z")}
    ids = [_text(v) for v in group["geometry_id"][()]]
    radius = group["radius"][()]
    layouts, media, readouts = (
        [json.loads(_text(v)) for v in group[key][()]] for key in ("layout", "medium", "readout")
    )
    detectors = []
    for k in range(len(offset) - 1):
        part = slice(int(offset[k]), int(offset[k + 1]))
        geometry = Geometry(
            string=table["string"][part],
            module=table["module"][part],
            x=table["x"][part],
            z=table["z"][part],
            radius=float(radius[k]),
            layout=layouts[k].get("layout", "custom"),
            params=layouts[k].get("params", {}),
        )
        if geometry.geometry_id != ids[k]:
            raise ValueError(
                f"geometry {k}: the stored id {ids[k]} does not match the module table ({geometry.geometry_id})"
            )
        detectors.append(
            Detector(geometry=geometry, medium=Medium.from_dict(media[k]), readout=Readout.from_dict(readouts[k]))
        )
    return detectors


# ----------------------------------------------------------------------------------------------------------------------
# Merging helpers


def _detector_key(det: Detector) -> str:
    return json.dumps([det.geometry.geometry_id, det.medium.to_dict(), det.readout.to_dict()], sort_keys=True)


def _comparable_config(config_json: str) -> dict[str, Any]:
    """Return the configuration without the per-detector parts (layout, custom table, medium, readout)."""
    data = SimConfig.from_json(config_json).to_dict()
    for key in ("layout", "custom_geometry", "medium", "readout"):
        data.pop(key, None)
    return data


def _inspect(path: Path) -> dict[str, Any]:
    with _open(path) as f:
        _check_format(f, path)
        return {
            "path": path,
            "config": _text(f.attrs["config"]),
            "provenance": json.loads(_text(f.attrs["provenance"])),
            "n_events": int(f.attrs["n_events"]),
            "groups": [name for name in _GROUPS if name in f],
            "event_id": f["truth/event_id"][()],
            "detectors": _read_geometries(f),
        }


def _seeds(provenance: dict[str, Any]) -> Iterable[Any]:
    if "merged_from" in provenance:
        for source in provenance["merged_from"]:
            yield from _seeds(source)
    else:
        yield provenance.get("seed")


def _check_mergeable(inputs: list[dict[str, Any]], renumber: bool) -> None:
    first = inputs[0]
    reference = _comparable_config(first["config"])
    for item in inputs[1:]:
        config = _comparable_config(item["config"])
        if config != reference:
            differ = sorted(
                f"{section}.{key}"
                for section in reference
                for key in reference[section]
                if config.get(section, {}).get(key) != reference[section][key]
            )
            raise ValueError(
                f"{item['path']} has a different configuration than {first['path']} "
                f"(apart from the detectors): {', '.join(differ)}"
            )
        if item["groups"] != first["groups"]:
            raise ValueError(
                f"{item['path']} has the groups {item['groups']}, {first['path']} has {first['groups']}; "
                "generate all inputs with the same --level, --save-paths and stochastic options"
            )
    seeds = [seed for item in inputs for seed in _seeds(item["provenance"])]
    repeated = sorted({s for s in seeds if seeds.count(s) > 1 and s is not None})
    if repeated:
        raise ValueError(
            f"seed(s) {repeated} occur in more than one input: those events would repeat the same random numbers"
        )
    if not renumber:
        ids = np.concatenate([item["event_id"] for item in inputs])
        if len(np.unique(ids)) != len(ids):
            raise ValueError(
                "the inputs share event ids; give each shard its own range with first_event_id / --first-event-id, "
                "or renumber the events (renumber=True / --renumber)"
            )


def _event_id_maps(inputs: list[dict[str, Any]], renumber: bool) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Per input: (old ids sorted, new ids in the same order) when renumbering, else None."""
    if not renumber:
        return [None] * len(inputs)
    maps: list[tuple[np.ndarray, np.ndarray] | None] = []
    start = 0
    for item in inputs:
        old = np.asarray(item["event_id"], dtype=np.int64)
        new = np.arange(start, start + len(old), dtype=np.int64)
        order = np.argsort(old)
        maps.append((old[order], new[order]))
        start += len(old)
    return maps


def _map_ids(values: np.ndarray, id_map: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if id_map is None:
        return values
    old, new = id_map
    return new[np.searchsorted(old, values)]


def _merge_group(
    dst: h5py.File,
    name: str,
    sources: list[h5py.File],
    remaps: list[np.ndarray],
    id_maps: list[tuple[np.ndarray, np.ndarray] | None],
) -> None:
    """Concatenate one group of all inputs into ``dst`` (offsets shifted, detector indices and event ids mapped)."""
    group = dst.create_group(name)
    for key in sources[0][name]:
        parts = [src[name][key] for src in sources]
        doc = _text(parts[0].attrs.get("description", ""))
        if key == "offset":
            pieces, shift = [], 0
            for part in parts:
                offsets = part[()].astype(np.int64)
                pieces.append(offsets[:-1] + shift)
                shift += int(offsets[-1])
            pieces.append(np.array([shift], dtype=np.int64))
            _create(group, key, np.concatenate(pieces), doc)
            continue
        total = sum(part.shape[0] for part in parts)
        options: dict[str, Any] = {}
        if total > _COMPRESS_ABOVE:
            options = {"compression": "gzip", "compression_opts": 4, "shuffle": True, "chunks": True}
        out = group.create_dataset(key, shape=(total, *parts[0].shape[1:]), dtype=parts[0].dtype, **options)
        out.attrs["units"] = _UNITS.get(key, "")
        out.attrs["description"] = doc
        start = 0
        for k, part in enumerate(parts):
            n = part.shape[0]
            for a in range(0, n, _COPY_BLOCK):
                b = min(n, a + _COPY_BLOCK)
                block = part[a:b]
                if name == "truth" and key == "geometry_index":
                    block = remaps[k][block.astype(np.int64)]
                elif key == "event_id" and name in ("truth", "paths"):
                    block = _map_ids(block.astype(np.int64), id_maps[k])
                out[start + a : start + b] = block
            start += n
