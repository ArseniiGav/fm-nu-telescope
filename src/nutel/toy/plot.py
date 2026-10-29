"""Detector and event displays in the (x, z) plane (matplotlib, imported only when a function is called).

* :func:`draw_geometry`: the modules as circles of their true radius and the strings as lines;
* :func:`event_display`: one event, hit modules filled (marker area proportional to the charge, colour = first pulse
  time), the true track or cascade and optionally the LineFit overlaid;
* :func:`photon_paths`: the stored photon paths of an event, coloured by their fate (direct, scattered, absorbed);
* :func:`animate_event`: a GIF of the Cherenkov front sweeping over the strings;
* :func:`gallery`: event displays in a grid of panels; :func:`draw_family`: sampled detector layouts in a grid.

Colours come from :data:`palette` (the Okabe-Ito colours, distinguishable with colour-vision deficiencies) and the
``viridis`` colour map for times. Plain matplotlib otherwise: the figures follow the active style.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from .config import C_VACUUM, EVENT_TYPES
from .reco import linefit as _linefit
from .reco import unit_vector

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "TIME_CMAP",
    "EventView",
    "animate_event",
    "draw_family",
    "draw_geometry",
    "event_display",
    "gallery",
    "palette",
    "photon_paths",
]

#: Colour-vision-deficiency-safe colours (Okabe & Ito) and the roles they play in the displays.
palette: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
    # roles
    "truth": "#D55E00",
    "linefit": "#000000",
    "module": "#555555",
    "string": "#BBBBBB",
    "noise": "#CC79A7",
    "direct": "#0072B2",
    "scattered": "#E69F00",
    "absorbed": "#999999",
    "escaped": "#56B4E9",
    # event types
    "track": "#0072B2",
    "starting": "#009E73",
    "cascade": "#D55E00",
    "atmospheric": "#CC79A7",
}

#: Colour map of the pulse times (perceptually uniform and CVD-safe).
TIME_CMAP = "viridis"

#: Photon fate of a stored path, as drawn by :func:`photon_paths`: name -> (colour, legend label).
_FATES: dict[str, tuple[str, str]] = {
    "direct": (palette["direct"], "detected, direct"),
    "scattered": (palette["scattered"], "detected, scattered"),
    "absorbed": (palette["absorbed"], "absorbed"),
    "other": (palette["escaped"], "escaped or lost"),
}


# ----------------------------------------------------------------------------------------------------------------------
# One event as plain arrays


@dataclass(frozen=True)
class EventView:
    """One event as plain arrays, the input of the displays.

    Build it from a data set with :meth:`from_data`, or by hand for a made-up event.

    Attributes
    ----------
    module_x, module_z
        Positions of all modules of the detector [m].
    radius
        Module radius [m].
    x, z, t, charge
        Pulses (or hits) of the event: module position [m], time [ns], charge [p.e.].
    n_noise
        Noise hits per pulse (``None`` = unknown).
    module_string
        String id of every module (to draw the strings; ``None`` groups the modules by ``x``).
    truth
        True event parameters: ``event_id``, ``type``, ``energy``, ``x0``, ``z0``, ``dir_x``, ``dir_z``, ``t0``
        and, for a muon that stopped, its stop point ``x_stop``, ``z_stop`` (any may be missing).
    n, n_group
        Refractive index (Cherenkov angle) and group index (speed of light ``c / n_group``; ``None`` = ``n``) of
        the medium, for the animation.
    geometry_index
        Index of the detector in a multi-geometry file (``None`` for single-geometry files).
    level
        What the rows are, ``"pulses"`` or ``"hits"`` (used in the title).
    """

    module_x: np.ndarray
    module_z: np.ndarray
    radius: float
    x: np.ndarray
    z: np.ndarray
    t: np.ndarray
    charge: np.ndarray
    n_noise: np.ndarray | None = None
    module_string: np.ndarray | None = None
    truth: Mapping[str, float] = field(default_factory=dict)
    n: float = 1.35
    n_group: float | None = None
    geometry_index: int | None = None
    level: str = "pulses"

    @classmethod
    def from_data(cls, data: Any, i: int, level: Literal["pulses", "hits"] = "pulses") -> EventView:
        """View of event ``i`` of a :class:`~nutel.toy.data.SimData` (pulses, or hits with charge 1 each)."""
        n_events = len(data.truth.event_id)
        if not -n_events <= i < n_events:
            raise IndexError(f"event index {i} out of range for {n_events} events")
        i = i % n_events
        rows = getattr(data, level, None)
        if rows is None:
            raise ValueError(f"the data set has no {level} level")
        a, b = int(rows.offset[i]), int(rows.offset[i + 1])
        x = np.asarray(rows.x[a:b], dtype=np.float64)
        if level == "pulses":
            charge = np.asarray(rows.charge[a:b], dtype=np.float64)
            n_noise = np.asarray(rows.n_noise[a:b], dtype=np.float64) if hasattr(rows, "n_noise") else None
        else:
            charge = np.ones_like(x)
            n_noise = np.asarray(rows.is_noise[a:b], dtype=np.float64) if hasattr(rows, "is_noise") else None
        truth_names = ("event_id", "type", "energy", "x0", "z0", "dir_x", "dir_z", "t0")
        truth = {name: float(getattr(data.truth, name)[i]) for name in truth_names if hasattr(data.truth, name)}
        detector = data.detector(i)
        geometry = detector.geometry
        stop = _stop_point(data, i, geometry)
        if stop is not None:
            truth["x_stop"], truth["z_stop"] = stop
        n_geometries = len(data.geometries)
        gi = getattr(data.truth, "geometry_index", None)
        return cls(
            module_x=np.asarray(geometry.x, dtype=np.float64),
            module_z=np.asarray(geometry.z, dtype=np.float64),
            radius=float(geometry.radius),
            x=x,
            z=np.asarray(rows.z[a:b], dtype=np.float64),
            t=np.asarray(rows.t[a:b], dtype=np.float64),
            charge=charge,
            n_noise=n_noise,
            module_string=np.asarray(geometry.string),
            truth=truth,
            n=float(detector.medium.refractive_index),
            n_group=float(detector.medium.group_index),
            geometry_index=int(gi[i]) if gi is not None and n_geometries > 1 else None,
            level=level,
        )

    def modules_hit(self) -> dict[str, np.ndarray]:
        """Pulses summed per module: ``x``, ``z``, ``charge``, ``t_first``, ``noise_only``, sorted by first time."""
        if len(self.x) == 0:
            empty = np.zeros(0)
            return {"x": empty, "z": empty, "charge": empty, "t_first": empty, "noise_only": empty.astype(bool)}
        keys, inverse = np.unique(np.stack([self.x, self.z], axis=1), axis=0, return_inverse=True)
        inverse = inverse.ravel()
        charge = np.bincount(inverse, self.charge, minlength=len(keys))
        t_first = np.full(len(keys), np.inf)
        np.minimum.at(t_first, inverse, self.t)
        if self.n_noise is None:
            noise_only = np.zeros(len(keys), dtype=bool)
        else:
            noise = np.bincount(inverse, self.n_noise, minlength=len(keys))
            noise_only = noise >= charge
        order = np.argsort(t_first)
        return {
            "x": keys[order, 0],
            "z": keys[order, 1],
            "charge": charge[order],
            "t_first": t_first[order],
            "noise_only": noise_only[order],
        }

    def title(self) -> str:
        """Title line: event id, type, energy, number of pulses or hits (and the detector in multi-geometry files)."""
        parts = []
        if "event_id" in self.truth:
            parts.append(f"event {int(self.truth['event_id'])}")
        if "type" in self.truth:
            kind = int(self.truth["type"])
            parts.append(EVENT_TYPES[kind] if 0 <= kind < len(EVENT_TYPES) else f"type {kind}")
        if "energy" in self.truth:
            parts.append(f"E = {_format_energy(self.truth['energy'])}")
        parts.append(f"{len(self.x)} {self.level}")
        if self.geometry_index is not None:
            parts.append(f"detector {self.geometry_index}")
        return ", ".join(parts)


# ----------------------------------------------------------------------------------------------------------------------
# Geometry


def draw_geometry(
    ax: Axes,
    geometry: Any,
    *,
    strings: bool = True,
    color: str | None = None,
    label: str | None = "module",
    margin: float = 0.08,
    set_limits: bool = True,
    fill: bool = False,
) -> Any:
    """Draw a detector: modules as circles of their true radius, strings as thin vertical lines.

    Parameters
    ----------
    ax
        Matplotlib axes (the aspect ratio is set to equal).
    geometry
        An object with arrays ``x``, ``z`` [m], ``radius`` [m] and optionally ``string`` (e.g.
        :class:`~nutel.toy.geometry.Geometry`), or an :class:`EventView`.
    strings
        Draw a line through the modules of each string.
    color
        Edge colour of the modules (default ``palette["module"]``).
    label
        Legend label of the modules (``None``: no legend entry).
    margin
        Margin around the modules, as a fraction of the larger extent.
    set_limits
        Set the axis limits to the modules plus the margin.
    fill
        Fill the modules with their edge colour (visible at the true radius in small panels).

    Returns
    -------
    matplotlib.collections.EllipseCollection
        The module circles.
    """
    from matplotlib.collections import EllipseCollection

    x, z, radius, string = _geometry_arrays(geometry)
    edge = color or palette["module"]
    if strings and len(x):
        groups = string if string is not None else np.round(x, 3)
        for s in np.unique(groups):
            sel = groups == s
            zs = np.sort(z[sel])
            ax.plot(
                np.full(2, x[sel].mean()),
                [zs[0], zs[-1]],
                color=palette["string"],
                lw=0.6,
                zorder=0.5,
                solid_capstyle="butt",
            )
    circles = EllipseCollection(
        widths=np.full(len(x), 2 * radius),
        heights=np.full(len(x), 2 * radius),
        angles=np.zeros(len(x)),
        units="xy",
        offsets=np.stack([x, z], axis=1) if len(x) else np.zeros((0, 2)),
        offset_transform=ax.transData,
        facecolors=edge if fill else "none",
        edgecolors=edge,
        linewidths=1.0,
        zorder=1,
    )
    ax.add_collection(circles)
    if label is not None:
        ax.plot([], [], ls="none", marker="o", ms=3, mfc=edge if fill else "none", mec=edge, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    if set_limits and len(x):
        _set_limits(ax, x, z, radius, margin)
    return circles


# ----------------------------------------------------------------------------------------------------------------------
# Event display


def event_display(
    ax: Axes,
    data: Any,
    i: int | None = None,
    *,
    size: Literal["charge", "none"] = "charge",
    color: Literal["time", "charge"] = "time",
    truth: bool = True,
    linefit: bool = False,
    level: Literal["pulses", "hits"] = "pulses",
    mark_noise: bool = False,
    colorbar: bool = True,
    legend: bool = True,
    cmap: str = TIME_CMAP,
    max_marker_area: float = 300.0,
    vmin: float | None = None,
    vmax: float | None = None,
    mask: ArrayLike | None = None,
    noise: bool = False,
) -> Any:
    """Draw the standard display of one event in the (x, z) plane.

    All modules are drawn as circles of their true radius; a hit module gets a filled marker whose AREA is
    proportional to its charge (the largest module of the event has ``max_marker_area`` points^2) and whose colour is
    its first pulse time. The true track (a line with an arrow along the direction of travel, through the anchor
    point; from the vertex for a starting track) or the cascade (vertex marker and an arrow along its axis) is
    overlaid, and optionally the LineFit (dashed, through the charge-weighted centre).

    Parameters
    ----------
    ax
        Matplotlib axes.
    data
        A :class:`~nutel.toy.data.SimData` (with ``i``) or an :class:`EventView`.
    i
        Event index in ``data``.
    size
        ``"charge"``: marker area proportional to the module charge; ``"none"``: one size for all hit modules.
    color
        ``"time"``: colour = first pulse time of the module; ``"charge"``: colour = module charge.
    truth
        Overlay the true track or cascade.
    linefit
        Overlay the LineFit direction of the pulses (charge-weighted).
    level
        ``"pulses"`` or ``"hits"`` (each hit with charge 1).
    mark_noise
        Draw modules whose pulses are all noise with a ring in ``palette["noise"]``.
    colorbar, legend
        Add a labelled colour bar and a legend.
    cmap
        Colour map of the markers.
    max_marker_area
        Marker area of the largest charge [points^2].
    vmin, vmax
        Colour range (default: the range of the drawn pulses).
    mask
        Pulses (or hits) to keep, e.g. from :func:`~nutel.toy.reco.causal_cleaning`: a boolean array over the rows
        of this event or over all rows of ``data``. The removed ones are drawn as grey crosses; the markers, the colour
        range and the LineFit use the kept ones only.
    noise
        Draw the pulses that are pure noise (from the truth, ``n_noise`` = charge) as crosses in
        ``palette["noise"]`` instead of markers (they are left out of the colour range and the LineFit).

    Returns
    -------
    matplotlib.collections.PathCollection
        The markers of the hit modules.
    """
    view = _view(data, i, level)
    draw_geometry(ax, view, label="module")
    keep = np.ones(len(view.x), dtype=bool)
    if mask is not None:
        keep = _event_mask(data, i, level, mask, len(view.x))
        _draw_crosses(ax, view.x[~keep], view.z[~keep], palette["absorbed"], "removed by the cleaning")
    if noise and view.n_noise is not None:
        is_noise = keep & (view.n_noise >= view.charge)
        _draw_crosses(ax, view.x[is_noise], view.z[is_noise], palette["noise"], "noise pulse")
        keep &= ~is_noise
    shown = _subset(view, keep)
    hit = shown.modules_hit()
    values = hit["t_first"] if color == "time" else hit["charge"]
    q_max = float(hit["charge"].max()) if len(hit["charge"]) else 1.0
    area = max_marker_area * hit["charge"] / q_max if size == "charge" else np.full(len(values), 0.3 * max_marker_area)
    area = np.maximum(area, 6.0)
    markers = ax.scatter(
        hit["x"],
        hit["z"],
        s=area,
        c=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )
    if mark_noise and hit["noise_only"].any():
        sel = hit["noise_only"]
        ax.scatter(
            hit["x"][sel],
            hit["z"][sel],
            s=area[sel] * 2.2 + 30,
            facecolors="none",
            edgecolors=palette["noise"],
            linewidths=1.2,
            zorder=3.5,
            label="noise only",
        )
    if truth:
        _draw_truth(ax, view.truth)
    if linefit and len(shown.x) >= 2:
        fit = _linefit(shown.x, shown.z, shown.t, shown.charge)
        if np.all(np.isfinite(fit.direction)):
            _draw_line(ax, fit.anchor, fit.direction, color=palette["linefit"], ls="--", label="LineFit")
    if colorbar and len(values):
        bar = ax.figure.colorbar(markers, ax=ax, fraction=0.046, pad=0.03)
        bar.set_label("first pulse time [ns]" if color == "time" else "module charge [p.e.]")
    if legend:
        _event_legend(ax, hit["charge"], max_marker_area if size == "charge" else None, q_max)
    ax.set_title(view.title())
    return markers


def gallery(
    data: Any,
    indices: Sequence[int],
    *,
    ncols: int = 3,
    panel_size: tuple[float, float] = (5.0, 3.8),
    titles: Sequence[str] | None = None,
    panel_options: Sequence[Mapping[str, Any] | None] | None = None,
    **display_options: Any,
) -> Figure:
    """Event displays of several events in a grid of panels.

    Parameters
    ----------
    data
        A :class:`~nutel.toy.data.SimData`.
    indices
        Event indices, one panel each.
    ncols
        Panels per row.
    panel_size
        Size of one panel [inches].
    titles
        Panel titles prepended to the standard title (e.g. the event type you selected).
    panel_options
        Per-panel options of :func:`event_display` on top of ``display_options`` (one mapping or ``None`` per
        panel), e.g. ``{"truth": False}`` for a noise-only readout; the key ``"data"`` takes that panel's event from
        another data set (e.g. a file made with ``--track-yield 0``).
    **display_options
        Passed to :func:`event_display` (the legend is drawn in the first panel only).

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _pyplot()
    if not indices:
        raise ValueError("no event index given")
    ncols = max(1, min(ncols, len(indices)))
    nrows = math.ceil(len(indices) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False, layout="constrained"
    )
    for k, ax in enumerate(axes.flat):
        if k >= len(indices):
            ax.set_axis_off()
            continue
        extra = panel_options[k] if panel_options is not None and k < len(panel_options) else None
        options: dict[str, Any] = {"legend": k == 0, **display_options, **(extra or {})}
        source = options.pop("data", data)
        event_display(ax, source, int(indices[k]), **options)
        if titles is not None:
            ax.set_title(f"{titles[k]}\n{ax.get_title()}", fontsize="medium")
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# Photon paths


def photon_paths(
    ax: Axes,
    data: Any,
    i: int,
    *,
    max_photons: int | None = 2000,
    show: Sequence[str] = ("direct", "scattered", "absorbed", "other"),
    alpha: float = 0.35,
    linewidth: float = 0.5,
    truth: bool = True,
    legend: bool = True,
    seed: int = 0,
) -> dict[str, int]:
    """Draw the stored photon paths of one event, coloured by the fate of the photon.

    Fates: ``direct`` (detected without scattering), ``scattered`` (detected after at least one scattering),
    ``absorbed`` (in the medium or by a module without detection) and ``other`` (escaped from the world or lost after
    the maximum number of steps). The paths are stored only for the first events of a file (``save_paths K``).

    Parameters
    ----------
    ax
        Matplotlib axes.
    data
        A :class:`~nutel.toy.data.SimData` with stored paths.
    i
        Event index (must be one of the events with stored paths).
    max_photons
        Draw at most this many paths (a random subset; detected photons are always drawn); ``None`` = all.
    show
        Fates to draw.
    alpha, linewidth
        Line style of the paths (detected photons are drawn twice as wide and more opaque).
    truth
        Overlay the true track or cascade.
    legend
        Add a legend with the number of photons of each fate.
    seed
        Seed of the random subset.

    Returns
    -------
    dict of str to int
        Number of photons of each fate in the event (all, not only those drawn).
    """
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    paths = _event_paths(data, i)
    view = EventView.from_data(data, i)
    draw_geometry(ax, view, label=None)
    fate = paths["fate"]
    counts = {name: int(np.sum(fate == name)) for name in _FATES}
    keep = np.isin(fate, list(show))
    if max_photons is not None and keep.sum() > max_photons:
        detected = keep & np.isin(fate, ["direct", "scattered"])
        others = np.flatnonzero(keep & ~detected)
        budget = max(0, max_photons - int(detected.sum()))
        chosen = np.random.default_rng(seed).choice(others, size=min(budget, len(others)), replace=False)
        keep = detected.copy()
        keep[chosen] = True
    handles = []
    for name in ("absorbed", "other", "scattered", "direct"):  # detected photons on top
        if name not in show:
            continue
        sel = np.flatnonzero(keep & (fate == name))
        segments = [paths["points"][k] for k in sel]
        colour, text = _FATES[name]
        bold = name in ("direct", "scattered")  # few and important: drawn bolder, on top
        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=colour,
                    linewidths=2 * linewidth if bold else linewidth,
                    alpha=min(1.0, 2.5 * alpha) if bold else alpha,
                    zorder=2.2 if bold else 2,
                )
            )
        handles.append(Line2D([], [], color=colour, lw=1.5, label=f"{text} ({counts[name]})"))
    if truth:
        _draw_truth(ax, view.truth)
    if legend:
        handles.reverse()
        truth_handles = [h for h in ax.get_legend_handles_labels()[0] if h.get_label() in {"true track", "true vertex"}]
        ax.legend(handles=handles + truth_handles, loc="upper right", fontsize="small", framealpha=0.85)
    ax.set_title(f"{view.title()}: photon paths")
    return counts


# ----------------------------------------------------------------------------------------------------------------------
# Animation


def animate_event(
    data: Any,
    i: int | None,
    path: str | Path,
    *,
    fps: int = 15,
    n_frames: int = 60,
    hold_frames: int = 10,
    dpi: int = 80,
    figsize: tuple[float, float] = (6.4, 4.8),
    t_range: tuple[float, float] | None = None,
    c: float = C_VACUUM,
) -> Path:
    """Write a GIF of the Cherenkov front of an event sweeping over the strings.

    Each frame at time ``t`` shows the muon position ``a + (t - t0) c d``, the two fronts of its Cherenkov light (in
    two dimensions the cone is two straight lines from the muon backwards, the envelope of the light emitted at
    ``theta_c`` since the muon entered the view or left the vertex), the circular front of the cascade light
    (radius ``(t - t0) c / n``) for cascades and starting tracks, and the modules whose first pulse has arrived,
    coloured by its time. Late (scattered) light and noise appear when they are recorded.

    Parameters
    ----------
    data
        A :class:`~nutel.toy.data.SimData` (with ``i``) or an :class:`EventView` (``i`` = None).
    i
        Event index.
    path
        Output GIF file (parent directories are created).
    fps
        Frames per second.
    n_frames
        Number of frames of the sweep; ``hold_frames`` more repeat the last one.
    hold_frames
        Frames that show the final state.
    dpi, figsize
        Resolution and size of the frames.
    t_range
        Time span ``(t_start, t_end)`` [ns]; default: from the moment the light source enters the view to the time
        the direct light has reached every module (or the median pulse time, if later).
    c
        Speed of light in vacuum [m/ns].

    Returns
    -------
    pathlib.Path
        The file written.
    """
    plt = _pyplot()
    from matplotlib import animation
    from matplotlib.colors import Normalize

    view = _view(data, i, "pulses")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="constrained")
    draw_geometry(ax, view, label=None)
    xlim, zlim = ax.get_xlim(), ax.get_ylim()
    hit = view.modules_hit()
    tr = view.truth
    kind = int(tr.get("type", 0))
    t0 = float(tr.get("t0", hit["t_first"].min() if len(hit["t_first"]) else 0.0))
    anchor = np.array([tr.get("x0", 0.0), tr.get("z0", 0.0)])
    d = np.array([tr.get("dir_x", 1.0), tr.get("dir_z", 0.0)])
    d = d / np.hypot(*d)
    n = view.n
    speed = c / (view.n_group or n)  # of the light in the medium
    has_track = kind != 2
    has_cascade = kind in (1, 2)
    s_enter, s_exit = _chord(anchor, d, xlim, zlim)
    s_start = 0.0 if kind == 1 else s_enter
    if t_range is None:
        corners = np.array([[xlim[0], zlim[0]], [xlim[0], zlim[1]], [xlim[1], zlim[0]], [xlim[1], zlim[1]]])
        reach = np.max(np.hypot(*(corners - anchor).T)) / speed
        t_start = t0 + (s_start / c if has_track else 0.0)
        t_end = t0 + (max(s_exit, 0.0) / c if has_track else 0.0) + 0.5 * reach
        if len(hit["t_first"]):
            t_start = min(t_start, float(hit["t_first"].min()))
            t_end = max(t_end, float(np.median(hit["t_first"])))
    else:
        t_start, t_end = t_range
    norm = Normalize(t_start, t_end)
    q_max = float(hit["charge"].max()) if len(hit["charge"]) else 1.0
    markers = ax.scatter([], [], s=[], c=[], cmap=TIME_CMAP, norm=norm, edgecolors="black", linewidths=0.4, zorder=3)
    bar = fig.colorbar(markers, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("first pulse time [ns]")
    theta_c = math.acos(1.0 / n) if n > 1 else 0.0
    rays = [unit_vector(math.atan2(d[1], d[0]) + sign * theta_c) for sign in (1.0, -1.0)]
    fronts = [ax.plot([], [], color=palette["sky_blue"], lw=1.5, zorder=2)[0] for _ in rays]
    (track_line,) = ax.plot([], [], color=palette["truth"], lw=1.2, zorder=2.5)
    (muon,) = ax.plot([], [], "o", color=palette["truth"], ms=5, zorder=4)
    (circle,) = ax.plot([], [], color=palette["sky_blue"], lw=1.5, zorder=2)
    if has_cascade:
        ax.plot(*anchor, marker="*", color=palette["truth"], mec="black", mew=0.4, ms=10, zorder=4, ls="none")
    proxies = [ax.plot([], [], color=palette["sky_blue"], lw=1.5, label="light front (direct)")[0]]
    if has_track:
        proxies.append(ax.plot([], [], "o-", color=palette["truth"], ms=4, lw=1.2, label="muon (true)")[0])
    if has_cascade:
        proxies.append(ax.plot([], [], "*", color=palette["truth"], mec="black", mew=0.4, ms=8, label="vertex")[0])
    ax.legend(handles=proxies, loc="upper right", fontsize="x-small", framealpha=0.85)
    clock = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize="small")
    ax.set_title(view.title(), fontsize="medium")
    times = np.concatenate([np.linspace(t_start, t_end, n_frames), np.full(hold_frames, t_end)])
    phi = np.linspace(0, 2 * np.pi, 181)

    def update(frame: int) -> list[Any]:
        t = float(times[frame])
        arrived = hit["t_first"] <= t
        markers.set_offsets(np.stack([hit["x"][arrived], hit["z"][arrived]], axis=1))
        markers.set_sizes(np.maximum(250.0 * hit["charge"][arrived] / q_max, 6.0))
        markers.set_array(hit["t_first"][arrived])
        clock.set_text(f"t = {t:.0f} ns")
        if has_track:
            s_mu = (t - t0) * c
            if s_mu > s_start:
                tip = anchor + s_mu * d
                track_line.set_data(*np.stack([anchor + s_start * d, tip], axis=1))
                muon.set_data([tip[0]], [tip[1]])
                for line, u in zip(fronts, rays, strict=True):
                    tail = anchor + s_start * d + (t - t0 - s_start / c) * speed * u
                    line.set_data([tip[0], tail[0]], [tip[1], tail[1]])
        if has_cascade and t > t0:
            r = (t - t0) * speed
            circle.set_data(anchor[0] + r * np.cos(phi), anchor[1] + r * np.sin(phi))
        return [markers, clock, track_line, muon, circle, *fronts]

    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    fig.canvas.draw()
    fig.set_layout_engine("none")  # the layout of the first frame is kept: much faster
    anim = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------------------------------------
# Detector family


def draw_family(
    detectors: Sequence[Any],
    *,
    ncols: int = 4,
    panel_size: tuple[float, float] = (4.0, 3.2),
    share_limits: bool = True,
    fill: bool = True,
) -> Figure:
    """Draw sampled detectors (geometries) in a grid of panels.

    Parameters
    ----------
    detectors
        :class:`~nutel.toy.family.Detector` objects (geometry + medium + readout) or bare geometries.
    ncols
        Panels per row.
    panel_size
        Size of one panel [inches].
    share_limits
        The same axis ranges in every panel, so that the detector sizes compare at a glance.
    fill
        Draw the modules as filled discs (see :func:`draw_geometry`).

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _pyplot()
    if not detectors:
        raise ValueError("no detector given")
    ncols = max(1, min(ncols, len(detectors)))
    nrows = math.ceil(len(detectors) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
        sharex=share_limits,
        sharey=share_limits,
        layout="constrained",
    )
    arrays = [_geometry_arrays(getattr(det, "geometry", det)) for det in detectors]
    for k, ax in enumerate(axes.flat):
        if k >= len(detectors):
            ax.set_axis_off()
            continue
        det = detectors[k]
        geometry = getattr(det, "geometry", det)
        draw_geometry(ax, geometry, label=None, set_limits=not share_limits, fill=fill)
        ax.set_title(_detector_title(k, det), fontsize="small")
        if share_limits and k % ncols:
            ax.set_ylabel("")
        if share_limits and k < len(detectors) - ncols:
            ax.set_xlabel("")
        elif share_limits:
            ax.xaxis.set_tick_params(labelbottom=True)
    if share_limits:
        x = np.concatenate([a[0] for a in arrays])
        z = np.concatenate([a[1] for a in arrays])
        _set_limits(axes.flat[0], x, z, max(a[2] for a in arrays), 0.05)
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# Helpers


def _subset(view: EventView, keep: np.ndarray) -> EventView:
    """Return the view with only the rows ``keep``."""
    if keep.all():
        return view
    import dataclasses

    return dataclasses.replace(
        view,
        x=view.x[keep],
        z=view.z[keep],
        t=view.t[keep],
        charge=view.charge[keep],
        n_noise=None if view.n_noise is None else view.n_noise[keep],
    )


def _event_mask(data: Any, i: int | None, level: str, mask: ArrayLike, n_rows: int) -> np.ndarray:
    """Boolean mask over the rows of one event, from a mask over that event or over all rows of ``data``."""
    m = np.asarray(mask, dtype=bool)
    if len(m) == n_rows:
        return m
    rows = getattr(data, level, None)
    if i is not None and rows is not None and len(m) == len(rows.t):
        k = i % len(rows.offset[:-1])
        return m[int(rows.offset[k]) : int(rows.offset[k + 1])]
    raise ValueError(f"mask has {len(m)} entries; expected {n_rows} (this event) or the rows of the whole data set")


def _draw_crosses(ax: Axes, x: np.ndarray, z: np.ndarray, colour: str, label: str) -> None:
    if len(x):
        ax.scatter(x, z, marker="x", s=28, c=colour, linewidths=1.1, zorder=2.8, label=label)


def _format_energy(energy: float) -> str:
    if energy >= 1e3:
        return f"{energy / 1e3:.3g} TeV"
    if energy >= 10:
        return f"{energy:.0f} GeV"
    return f"{energy:.2g} GeV"


def _pyplot() -> Any:
    import matplotlib.pyplot as plt

    return plt


def _view(data: Any, i: int | None, level: Literal["pulses", "hits"]) -> EventView:
    if isinstance(data, EventView):
        return data
    if i is None:
        raise TypeError("an event index i is required unless data is an EventView")
    return EventView.from_data(data, i, level)


def _geometry_arrays(geometry: Any) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    if isinstance(geometry, EventView):
        return geometry.module_x, geometry.module_z, geometry.radius, geometry.module_string
    string = getattr(geometry, "string", None)
    return (
        np.asarray(geometry.x, dtype=np.float64),
        np.asarray(geometry.z, dtype=np.float64),
        float(geometry.radius),
        None if string is None else np.asarray(string),
    )


def _set_limits(ax: Axes, x: np.ndarray, z: np.ndarray, radius: float, margin: float) -> None:
    span = max(float(np.ptp(x)), float(np.ptp(z)), 4 * radius, 1.0)
    pad = margin * span + radius
    ax.set_xlim(float(x.min()) - pad, float(x.max()) + pad)
    ax.set_ylim(float(z.min()) - pad, float(z.max()) + pad)


def _chord(anchor: np.ndarray, d: np.ndarray, xlim: Sequence[float], zlim: Sequence[float]) -> tuple[float, float]:
    """Parameters ``s`` at which the line ``anchor + s d`` enters and leaves the box (``(0, 0)`` if it misses)."""
    lo, hi = -math.inf, math.inf
    for k, (a, b) in enumerate((xlim, zlim)):
        if abs(d[k]) < 1e-12:
            if not min(a, b) <= anchor[k] <= max(a, b):
                return 0.0, 0.0
            continue
        s1, s2 = (a - anchor[k]) / d[k], (b - anchor[k]) / d[k]
        lo, hi = max(lo, min(s1, s2)), min(hi, max(s1, s2))
    return (lo, hi) if lo <= hi else (0.0, 0.0)


def _stop_point(data: Any, i: int, geometry: Any) -> tuple[float, float] | None:
    """Point where the muon of event ``i`` stopped (end of its light-emitting segment), or ``None``."""
    truth = data.truth
    if not bool(np.asarray(getattr(truth, "stopped", np.zeros(len(truth.event_id), bool)))[i]):
        return None
    kind = int(truth.type[i])
    p = np.array([float(truth.x0[i]), float(truth.z0[i])])
    d = np.array([float(truth.dir_x[i]), float(truth.dir_z[i])])
    s_start = 0.0
    if kind in (0, 3):  # through-going: the segment starts where the line enters the generation box
        config = getattr(data, "config", None)
        margin = float(config.prior.track_margin) if config is not None else 100.0
        x_min, x_max, z_min, z_max = geometry.bounds
        s_start, _ = _chord(p, d, (x_min - margin, x_max + margin), (z_min - margin, z_max + margin))
    end = p + (s_start + float(truth.track_length[i])) * d
    return float(end[0]), float(end[1])


def _draw_line(
    ax: Axes,
    point: ArrayLike,
    direction: ArrayLike,
    *,
    color: str,
    ls: str,
    label: str,
    start_at_point: bool = False,
    end: float | None = None,
) -> None:
    """Line through ``point`` along ``direction`` across the view (or up to ``point + end d``), with an arrow."""
    p = np.asarray(point, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.hypot(*d)
    xlim, zlim = ax.get_xlim(), ax.get_ylim()
    s_in, s_out = _chord(p, d, xlim, zlim)
    if start_at_point:
        s_in = 0.0
    if end is not None:
        s_out = min(s_out, end)
    if s_out <= s_in:
        s_in, s_out = 0.0, 0.0
    a, b = p + s_in * d, p + s_out * d
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, ls=ls, lw=1.4, zorder=2.5, label=label)
    span = math.hypot(xlim[1] - xlim[0], zlim[1] - zlim[0])
    centre = p if start_at_point or s_in <= 0 <= s_out else 0.5 * (a + b)
    tip = centre + 0.08 * span * d
    ax.annotate(
        "",
        xy=(float(tip[0]), float(tip[1])),
        xytext=(float(centre[0]), float(centre[1])),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4, "mutation_scale": 14},
        zorder=4,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)


def _draw_truth(ax: Axes, truth: Mapping[str, float]) -> None:
    if not {"x0", "z0", "dir_x", "dir_z"} <= set(truth):
        return
    kind = int(truth.get("type", 0))
    point = (truth["x0"], truth["z0"])
    direction = (truth["dir_x"], truth["dir_z"])
    colour = palette["truth"]
    if kind == 2:
        xlim, zlim = ax.get_xlim(), ax.get_ylim()
        span = math.hypot(xlim[1] - xlim[0], zlim[1] - zlim[0])
        d = np.asarray(direction) / np.hypot(*direction)
        tip = np.asarray(point) + 0.08 * span * d
        ax.annotate(
            "",
            xy=(float(tip[0]), float(tip[1])),
            xytext=point,
            arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.4, "mutation_scale": 14},
            zorder=4,
        )
    else:
        end = None
        if "x_stop" in truth and "z_stop" in truth:
            d = np.asarray(direction, dtype=np.float64) / np.hypot(*direction)
            end = float(np.dot(np.array([truth["x_stop"], truth["z_stop"]]) - np.asarray(point), d))
        _draw_line(ax, point, direction, color=colour, ls="-", label="true track", start_at_point=kind == 1, end=end)
        if end is not None:
            xlim, zlim = ax.get_xlim(), ax.get_ylim()
            inside = xlim[0] <= truth["x_stop"] <= xlim[1] and zlim[0] <= truth["z_stop"] <= zlim[1]
            ax.plot(
                truth["x_stop"],
                truth["z_stop"],
                marker="X",
                ms=9,
                color=colour,
                mec="black",
                mew=0.4,
                ls="none",
                zorder=4,
                label="muon stops" if inside else "muon stops (outside the view)",
            )
    if kind in (1, 2):
        ax.plot(*point, marker="*", ms=11, color=colour, mec="black", mew=0.4, ls="none", zorder=4, label="true vertex")


def _event_legend(ax: Axes, charges: np.ndarray, max_area: float | None, q_max: float) -> None:
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    if max_area is not None and len(charges):
        for q in sorted({1.0, float(np.round(q_max / 2)), q_max}):
            if q < 1:
                continue
            area = max(max_area * q / q_max, 6.0)
            handles.append(Line2D([], [], ls="none", marker="o", ms=math.sqrt(area), mfc="white", mec="black", mew=0.5))
            labels.append(f"{q:g} p.e.")
    ax.legend(handles, labels, loc="upper right", fontsize="x-small", framealpha=0.85, labelspacing=0.9, borderpad=0.8)


def _event_paths(data: Any, i: int) -> dict[str, Any]:
    """Polylines of the photons of event ``i`` and their fates, from ``data.paths``."""
    from .data import PointType

    paths = getattr(data, "paths", None)
    if paths is None:
        raise ValueError("the data set has no stored photon paths (generate with save_paths K / --save-paths K)")
    event_id = int(np.asarray(data.truth.event_id)[i])
    photons = paths.of_event(event_id)
    if len(photons) == 0:
        raise ValueError(f"no stored photon paths for event {i} (paths are stored for the first events only)")
    last = np.asarray(paths.fate)[photons]
    scattered = np.asarray(paths.n_scatter)[photons] > 0
    fate = np.full(len(photons), "other", dtype=object)
    fate[np.isin(last, [PointType.ABSORB, PointType.ABSORB_MODULE])] = "absorbed"
    detected = last == PointType.DETECT
    fate[detected & scattered] = "scattered"
    fate[detected & ~scattered] = "direct"
    points = [np.stack(paths.polyline(int(k))[:2], axis=1) for k in photons]
    return {"points": points, "fate": fate.astype(str)}


def _detector_title(k: int, det: Any) -> str:
    geometry = getattr(det, "geometry", det)
    n_strings = len(np.unique(np.asarray(geometry.string))) if hasattr(geometry, "string") else 0
    lines = [f"detector {k}: {n_strings} strings, {len(np.asarray(geometry.x))} modules"]
    medium = getattr(det, "medium", None)
    if medium is not None:
        lines.append(
            f"$n$ = {medium.n:.3g}, $\\lambda_a$ = {medium.absorption_length:.3g} m, "
            f"$\\lambda_\\mathrm{{eff}}$ = {medium.scattering_length_eff:.3g} m"
        )
    readout = getattr(det, "readout", None)
    if readout is not None:
        rate = readout.noise_rate if readout.noise_rate is not None else getattr(medium, "background_rate", None)
        if rate is not None:
            lines[-1] += f", noise {rate / 1e3:.3g} kHz"
    return "\n".join(lines)
