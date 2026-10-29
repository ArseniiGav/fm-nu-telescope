"""Command line interface ``nutelsim`` (also available as ``python -m nutel.toy``).

::

    nutelsim generate [--preset NAME] [--medium water|ice] [-n N] [--seed S] [-o FILE] [options]
    nutelsim info FILE
    nutelsim merge -o OUT FILE [FILE ...]
    nutelsim presets
    nutelsim family [--family NAME] [--n K] [--seed S] [-o FIG]
    nutelsim noise FILE --rate R [--seed S] [-o OUT]
    nutelsim display FILE [-i I] [-o FIG]
    nutelsim animate FILE [-i I] [-o GIF]

Every configuration parameter has an option of the same name (``absorption_length`` -> ``--absorption-length``);
options given on the command line change the chosen preset and medium, e.g.
``nutelsim generate --preset scatter --medium ice --noise-rate 20e3``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import MEDIA, PRESETS, SimConfig, config_fields, medium_name, preset
from .family import FAMILIES

__all__ = ["build_parser", "default_output", "main"]

log = logging.getLogger("nutel")

_PART_TITLES = {
    "layout": "detector layout (single-detector runs)",
    "medium": "medium (changes the --medium preset)",
    "readout": "readout",
    "physics": "physics switches and photon transport (change the --preset)",
    "emission": "light emission",
    "prior": "prior of the event parameters",
}
_FIXED_TITLE = "fixed event parameters (the same value for every event)"
_FIXED = {"event_type", "fixed_angle", "fixed_vertex", "fixed_energy", "fixed_t0", "fixed_inelasticity"}


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show the default of an option, unless it is empty (``None`` or ``False``)."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None or action.default is False or action.default is argparse.SUPPRESS:
            return action.help or ""
        return super()._get_help_string(action) or ""


# ----------------------------------------------------------------------------------------------------------------------
# Parser


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of ``nutelsim``."""
    parser = argparse.ArgumentParser(
        prog="nutelsim",
        description='Simulation of the two-dimensional neutrino-telescope toy "Light in the Deep".',
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    _add_generate(commands)

    info = commands.add_parser("info", help="describe a data file", description="Describe a data file.")
    info.add_argument("file", type=Path, help="HDF5 file written by nutelsim")
    _add_logging_options(info)

    mrg = commands.add_parser(
        "merge",
        help="concatenate data files (cluster shards, several detectors)",
        description="Concatenate data files of the same configuration (apart from the detectors), e.g. shards "
        "generated with different seeds and disjoint --first-event-id ranges. Detectors are de-duplicated.",
        formatter_class=_HelpFormatter,
    )
    mrg.add_argument("-o", "--output", type=Path, required=True, metavar="OUT", help="output file")
    mrg.add_argument("files", type=Path, nargs="+", metavar="FILE", help="input files")
    mrg.add_argument(
        "--renumber", action="store_true", help="renumber the event ids 0 ... N-1 instead of checking them"
    )
    mrg.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    _add_logging_options(mrg)

    pre = commands.add_parser(
        "presets", help="list the physics presets and media", description="List the physics presets and media."
    )
    _add_logging_options(pre)

    fam = commands.add_parser(
        "family",
        help="sample and draw detectors of a detector family",
        description="Sample K detectors (geometry, medium, readout) of a detector family, list them and optionally "
        "draw their layouts.",
        formatter_class=_HelpFormatter,
    )
    fam.add_argument("--family", choices=list(FAMILIES), default="default", help="family preset")
    fam.add_argument("--n", type=int, default=8, metavar="K", help="number of detectors")
    fam.add_argument("--seed", type=int, default=0, help="seed of the sampling (the --geometry-seed of generate)")
    fam.add_argument("-o", "--output", type=Path, default=None, metavar="FIG", help="draw the layouts into this file")
    fam.add_argument("--ncols", type=int, default=4, help="panels per row of the figure")
    fam.add_argument("--dpi", type=int, default=120, help="resolution of a raster figure")
    _add_logging_options(fam)

    noise = commands.add_parser(
        "noise",
        help="add noise hits to a stored file (augmentation)",
        description="Add noise hits at RATE per module to every event of a file with the hit level and rebuild the "
        "pulses (train with noise, test without: the NuBench protocol).",
        formatter_class=_HelpFormatter,
    )
    noise.add_argument("file", type=Path, help="input file (needs the hit level)")
    noise.add_argument("--rate", type=float, required=True, metavar="R", help="added noise rate per module [Hz]")
    noise.add_argument("--seed", type=int, default=0, help="random seed of the noise")
    noise.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="OUT",
        help="output file (default: <input>_noise<R>Hz_seed<S>.h5)",
    )
    noise.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    _add_logging_options(noise)

    disp = commands.add_parser(
        "display",
        help="draw the event display of one event",
        description="Draw one event: modules, hit modules (marker area = charge, colour = first pulse time), the true "
        "track or cascade, optionally the LineFit and the stored photon paths.",
        formatter_class=_HelpFormatter,
    )
    disp.add_argument("file", type=Path, help="data file")
    disp.add_argument("-i", "--index", type=int, default=0, metavar="I", help="event index in the file")
    disp.add_argument(
        "-o", "--output", type=Path, default=None, metavar="FIG", help="figure file (default: <input>_event<I>.png)"
    )
    disp.add_argument("--level", choices=("pulses", "hits"), default="pulses", help="draw pulses or hits")
    disp.add_argument("--linefit", action=argparse.BooleanOptionalAction, default=True, help="overlay the LineFit")
    disp.add_argument("--paths", action="store_true", help="add a panel with the stored photon paths")
    disp.add_argument("--dpi", type=int, default=150, help="resolution of a raster figure")
    _add_logging_options(disp)

    anim = commands.add_parser(
        "animate",
        help="write a GIF of the light front of one event",
        description="Write a GIF of the Cherenkov front of one event sweeping over the strings.",
        formatter_class=_HelpFormatter,
    )
    anim.add_argument("file", type=Path, help="data file")
    anim.add_argument("-i", "--index", type=int, default=0, metavar="I", help="event index in the file")
    anim.add_argument(
        "-o", "--output", type=Path, default=None, metavar="GIF", help="GIF file (default: <input>_event<I>.gif)"
    )
    anim.add_argument("--fps", type=int, default=15, help="frames per second")
    anim.add_argument("--frames", type=int, default=60, help="number of frames of the sweep")
    anim.add_argument("--dpi", type=int, default=80, help="resolution of the frames")
    _add_logging_options(anim)
    return parser


def _add_generate(commands: Any) -> None:
    gen = commands.add_parser(
        "generate",
        help="simulate events and write them to an HDF5 file",
        description="Simulate events with parameters drawn from the prior and write them to an HDF5 file. Every "
        "configuration parameter has an option; the options change the chosen preset and medium.",
        formatter_class=_HelpFormatter,
    )
    gen.add_argument("--preset", choices=list(PRESETS), default="full", help="physics preset (cumulative ladder)")
    gen.add_argument("--medium", choices=list(MEDIA), default=None, help="medium preset (default: water)")
    gen.add_argument("-n", "--n-events", type=int, default=None, metavar="N", help="number of events (default 10000)")
    gen.add_argument("--seed", type=int, default=0, help="random seed (non-negative integer)")
    out = gen.add_argument_group("output")
    out.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="output file (default: data/toy_<preset>_<medium>_n<N>_seed<S>.h5)",
    )
    out.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    out.add_argument(
        "--level",
        choices=("hits", "pulses", "both"),
        default="both",
        help="data levels to store: pulses are always stored; 'both' also keeps every hit ('hits': alias of 'both')",
    )
    out.add_argument(
        "--save-paths", type=int, default=0, metavar="K", help="store the photon paths of the first K events"
    )
    multi = gen.add_argument_group("multi-detector data sets")
    multi.add_argument(
        "--family",
        choices=list(FAMILIES),
        default=None,
        help="sample the detectors (layout, medium, readout efficiency, jitter and noise) from this family; the "
        "layout, medium and these readout options cannot be given with it",
    )
    multi.add_argument("--n-geometries", type=int, default=1, metavar="K", help="number of sampled detectors")
    multi.add_argument("--geometry-seed", type=int, default=0, metavar="S", help="seed of the detector sampling")
    multi.add_argument(
        "--events-per-geometry",
        type=int,
        default=None,
        metavar="M",
        help="M events in every detector (instead of N events split over the detectors)",
    )
    groups = {name: gen.add_argument_group(title) for name, title in _PART_TITLES.items()}
    groups["fixed"] = gen.add_argument_group(_FIXED_TITLE)
    groups["fixed"].add_argument(
        "--fixed-direction",
        type=float,
        nargs=2,
        default=None,
        metavar=("ZENITH", "AZIMUTH"),
        help="direction of every event as zenith and azimuth [deg] (the alternative to --fixed-angle)",
    )
    _add_config_options(groups)
    gen.set_defaults(subparser=gen)
    adv = gen.add_argument_group("advanced options")
    adv.add_argument("--device", default="auto", help="'auto' (CUDA if available), 'cpu', 'cuda' or 'cuda:N'")
    adv.add_argument("--threads", type=int, default=None, help="CPU threads used by torch (default: torch's choice)")
    adv.add_argument(
        "--photons-per-chunk",
        type=int,
        default=None,
        metavar="P",
        help="photons transported at once (memory control; default depends on the device)",
    )
    adv.add_argument("--first-event-id", type=int, default=0, metavar="ID", help="event id of the first event")
    adv.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True, help="show a progress indicator"
    )
    _add_logging_options(gen)


def _add_logging_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("logging").add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", help="more log messages")
    group.add_argument("-q", "--quiet", action="store_true", help="only warnings and errors")


def _add_config_options(groups: dict[str, Any]) -> None:
    """One option per configuration parameter, derived from the field metadata; ``None`` keeps the preset's value."""
    defaults = SimConfig()
    for part, f in config_fields():
        flag = "--" + f.name.replace("_", "-")
        meta = f.metadata
        unit = f" [{meta['unit']}]" if meta.get("unit") else ""
        value = getattr(getattr(defaults, part), f.name)
        kwargs: dict[str, Any] = {"default": None, "dest": f.name}
        if isinstance(f.default, bool):
            on = [name for name, cfg in PRESETS.items() if getattr(getattr(cfg, part), f.name)]
            kwargs["action"] = argparse.BooleanOptionalAction
            kwargs["help"] = f"{meta['help']} (on in: {', '.join(on) or 'no preset'})"
        else:
            shown = list(value) if isinstance(value, tuple) else value
            kwargs["help"] = f"{meta['help']}{unit}" + ("" if value is None else f" (full, water: {_fmt(shown)})")
            number = (
                int
                if meta.get("integer") or (isinstance(f.default, int) and not isinstance(f.default, bool))
                else float
            )
            if "choices" in meta:
                kwargs["choices"] = list(meta["choices"])
            elif meta.get("nargs"):
                nargs = int(meta["nargs"])
                kwargs |= {"nargs": nargs, "type": number}
                if "metavar" in meta:
                    kwargs["metavar"] = tuple(meta["metavar"])
            else:
                kwargs |= {"type": number, "metavar": meta.get("metavar", "N" if number is int else "VALUE")}
        group = groups["fixed"] if f.name in _FIXED else groups[part]
        group.add_argument(flag, **kwargs)


def _fmt(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_fmt(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# ----------------------------------------------------------------------------------------------------------------------
# Commands


def default_output(
    preset_name: str,
    config: SimConfig,
    n_events: int,
    seed: int,
    *,
    family: str | None = None,
    n_geometries: int = 1,
    medium: str = "water",
) -> Path:
    """Default file name ``data/toy_<preset>_<medium>_n<N>_seed<S>.h5``.

    A configuration that differs from the preset in the given medium gets the tag ``<preset>-custom-<hash>``; a
    multi-detector run replaces the medium by ``<family>-k<K>``.
    """
    tag = preset_name
    if config != preset(preset_name, medium=medium):
        tag += "-custom-" + hashlib.sha1(config.to_json().encode()).hexdigest()[:6]
    place = f"{family}-k{n_geometries}" if family is not None else medium_name(config.medium)
    return Path("data") / f"toy_{tag}_{place}_n{n_events}_seed{seed}.h5"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``nutelsim``; returns the exit status."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("nutel").setLevel(level)
    try:
        if args.command == "generate":
            return _generate(args, argv)
        if args.command == "info":
            from .io import describe

            print(describe(args.file))
        elif args.command == "merge":
            from .io import merge

            out = merge(args.files, args.output, overwrite=args.overwrite, renumber=args.renumber)
            log.info("merged %d files into %s", len(args.files), out)
        elif args.command == "presets":
            from .config import describe_presets

            print(describe_presets())
            print("\nexample: nutelsim generate --preset scatter --medium ice --noise-rate 20e3 -n 1000")
        elif args.command == "family":
            _family(args)
        elif args.command == "noise":
            _noise(args)
        elif args.command == "display":
            _display(args)
        elif args.command == "animate":
            _animate(args)
    except (ValueError, TypeError, OSError, RuntimeError, IndexError, KeyError, ImportError) as error:
        print(f"nutelsim {args.command}: error: {error}", file=sys.stderr)
        return 1
    return 0


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for _, f in config_fields():
        value = getattr(args, f.name)
        if value is not None:
            out[f.name] = tuple(value) if f.metadata.get("nargs") else value
    if args.fixed_direction is not None:
        if "fixed_angle" in out:
            raise ValueError("give either --fixed-angle or --fixed-direction, not both")
        from .events import angle_from_zenith_azimuth

        zenith, azimuth = (math.radians(v) for v in args.fixed_direction)
        out["fixed_angle"] = math.degrees(float(angle_from_zenith_azimuth(np.asarray(zenith), np.asarray(azimuth))))
    return out


#: Readout options that a detector family samples per detector.
_FAMILY_READOUT = ("efficiency", "time_jitter", "noise_rate")


def _family_flag_conflicts(args: argparse.Namespace) -> list[str]:
    """Options given explicitly that a detector family would replace (layout, medium, sampled readout values)."""
    flags = ["--medium"] if args.medium is not None else []
    for part, f in config_fields():
        sampled = part in ("layout", "medium") or (part == "readout" and f.name in _FAMILY_READOUT)
        if sampled and getattr(args, f.name) is not None:
            flags.append("--" + f.name.replace("_", "-"))
    return flags


def _generate(args: argparse.Namespace, argv: list[str]) -> int:
    if args.family is not None:
        conflicts = _family_flag_conflicts(args)
        if conflicts:
            args.subparser.error(
                f"--family {args.family} samples the layout, the medium and the readout efficiency, time jitter and "
                f"noise of every detector; remove {', '.join(conflicts)} (or drop --family)"
            )
    args.medium = args.medium or "water"
    if args.n_events is None and args.events_per_geometry is None:
        args.n_events = 10_000
    overrides = _overrides(args)
    config = preset(args.preset, medium=args.medium, **overrides)
    n_total = args.n_events if args.n_events is not None else args.events_per_geometry * args.n_geometries
    output = args.output or default_output(
        args.preset,
        config,
        n_total,
        args.seed,
        family=args.family,
        n_geometries=args.n_geometries,
        medium=args.medium,
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists (use --overwrite to replace it)")
    import torch

    from .io import save
    from .simulate import generate, resolve_device

    if args.threads is not None:
        if args.threads < 1:
            raise ValueError(f"--threads must be at least 1, got {args.threads}")
        torch.set_num_threads(args.threads)
    resolve_device(args.device)  # fail before the simulation, not after it
    output.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(output.parent, os.W_OK):
        raise PermissionError(f"cannot write {output}: no write permission in {output.parent.resolve()}")
    place = f"detectors of the family {args.family}" if args.family is not None else args.medium
    log.info("preset %s in %s%s", args.preset, place, f" with {overrides}" if overrides else "")
    data = generate(
        args.n_events,
        config=config,
        seed=args.seed,
        device=args.device,
        level=args.level,
        save_paths=args.save_paths,
        family=args.family,
        n_geometries=args.n_geometries,
        geometry_seed=args.geometry_seed,
        events_per_geometry=args.events_per_geometry,
        photons_per_chunk=args.photons_per_chunk,
        first_event_id=args.first_event_id,
        progress=args.progress and not args.quiet,
    )
    data.provenance |= {
        "preset": args.preset,
        "medium": args.medium if args.family is None else "sampled per detector",
        "overrides": {k: list(v) if isinstance(v, tuple) else v for k, v in overrides.items()},
        "command_line": "nutelsim " + shlex.join(argv),
    }
    save(data, output, overwrite=args.overwrite)
    seconds = float(data.provenance.get("wall_time_s", float("nan")))
    log.info(
        "wrote %s: %d events in %.1f s (%.1f ms/event)", output, len(data), seconds, 1e3 * seconds / max(len(data), 1)
    )
    return 0


def _family(args: argparse.Namespace) -> None:
    from .family import family, sample_detectors

    fam = family(args.family)
    detectors = sample_detectors(args.n, fam, seed=args.seed)
    print(f"family {args.family!r}:")
    print(fam.describe())
    print(f"\n{args.n} detectors (seed {args.seed}):")
    for k, det in enumerate(detectors):
        print(f"[{k}] " + det.describe())
    if args.output is not None:
        plt = _pyplot()
        from .plot import draw_family

        fig = draw_family(detectors, ncols=args.ncols)
        fig.suptitle(f"detector family '{args.family}', seed {args.seed}")
        _savefig(fig, args.output, args.dpi)
        plt.close(fig)
        log.info("wrote %s", args.output)


def _noise(args: argparse.Namespace) -> None:
    from .io import load, save
    from .readout import add_noise

    output = args.output or args.file.with_name(f"{args.file.stem}_noise{args.rate:g}Hz_seed{args.seed}.h5")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists (use --overwrite to replace it)")
    data = load(args.file, level="hits")
    noisy = add_noise(data, args.rate, args.seed)
    save(noisy, output, overwrite=args.overwrite)
    log.info("wrote %s: %d events with %.0f Hz of added noise per module", output, len(noisy), args.rate)


def _display(args: argparse.Namespace) -> None:
    from .io import load

    plt = _pyplot()
    from .plot import event_display, photon_paths

    data = load(args.file, level="both" if args.level == "hits" else "pulses", paths=args.paths)
    output = args.output or args.file.with_name(f"{args.file.stem}_event{args.index}.png")
    ncols = 2 if args.paths else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.0 * ncols, 5.0), squeeze=False, layout="constrained")
    event_display(axes[0, 0], data, args.index, level=args.level, linefit=args.linefit)
    if args.paths:
        photon_paths(axes[0, 1], data, args.index)
    _savefig(fig, output, args.dpi)
    plt.close(fig)
    log.info("wrote %s", output)


def _animate(args: argparse.Namespace) -> None:
    from .io import load

    _pyplot()
    from .plot import animate_event

    data = load(args.file, level="pulses", paths=False)
    output = args.output or args.file.with_name(f"{args.file.stem}_event{args.index}.gif")
    animate_event(data, args.index, output, fps=args.fps, n_frames=args.frames, dpi=args.dpi)
    log.info("wrote %s", output)


def _pyplot() -> Any:
    """Matplotlib with a file backend (the commands write figures, they never open windows)."""
    try:
        import matplotlib
    except ImportError as error:
        raise ImportError("drawing needs matplotlib: pip install 'nutel[analysis]'") from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _savefig(fig: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
