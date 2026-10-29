"""Quick start: simulate events of the toy telescope, save them, read them back and look at what was stored.

Run from the repository root; it takes seconds on a CPU::

    python examples/quickstart.py
    python examples/quickstart.py --events 5000 --preset scatter --medium ice --device cpu

It writes ``data/quickstart_<preset>_<medium>.h5`` and, if matplotlib is installed, a display of the first event
next to it (``.png``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nutel.toy import PRESETS, SimData, event_features, generate, load, preset, save


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Simulate, save and read back a small data set of the toy telescope.")
    parser.add_argument("--events", type=int, default=2000, help="number of events (default: 2000)")
    parser.add_argument("--preset", choices=list(PRESETS), default="full", help="physics preset (default: full)")
    parser.add_argument("--medium", choices=("water", "ice"), default="water", help="medium (default: water)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--device", default="auto", help="'auto' (GPU if available), 'cpu' or 'cuda' (default: auto)")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory (default: data)")
    return parser.parse_args()


def summarise(data: SimData) -> None:
    """Print the photon bookkeeping, the pulses per event by type and the first event."""
    photons = data.photons
    emitted = photons.n_emitted.sum()
    fates = {
        "detected": photons.n_detected,
        "absorbed by a module": photons.n_absorbed_module,
        "absorbed in the medium": photons.n_absorbed,
        "escaped": photons.n_escaped,
        "lost": photons.n_lost,
    }
    print(f"fate of the {emitted:.3g} emitted photons:")
    for name, count in fates.items():
        print(f"  {name:<24s} {count.sum() / emitted:8.3%}")
    ended = sum(fates.values())
    print(f"  every photon accounted for in every event: {bool(np.all(ended == photons.n_emitted))}")

    n_pulses = np.diff(data.pulses.offset)
    print("\npulses per event (median, 5th and 95th percentile):")
    for code, name in enumerate(("track", "starting", "cascade", "atmospheric")):
        mask = data.truth.type == code
        if mask.any():
            lo, mid, hi = np.percentile(n_pulses[mask], [5, 50, 95])
            print(f"  {name:<12s} {mid:6.0f} ({lo:.0f} to {hi:.0f}), {mask.sum()} events")

    event = data.event(0)
    t = event.truth
    print(
        f"\nfirst event: {event.type_name}, angle {np.degrees(t['angle']):.1f} deg, "
        f"zenith {np.degrees(t['zenith']):.1f} deg, E = {t['energy']:.0f} GeV, {len(event.pulses)} pulses on "
        f"{len(np.unique(event.pulses.string * 1000 + event.pulses.module))} modules"
    )


def plot_first_event(data: SimData, path: Path) -> None:
    """Save the standard display of the first event."""
    import matplotlib.pyplot as plt

    from nutel.toy.plot import event_display

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    event_display(ax, data, 0, linefit=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Simulate a small data set, save it, read it back, print a summary and draw the first event."""
    args = parse_args()
    config = preset(args.preset, medium=args.medium)
    print(f"preset {args.preset}, medium {args.medium}:\n{config.describe()}\n")

    # Simulate: the event parameters are drawn from the prior and every photon is tracked.
    data = generate(args.events, config=config, seed=args.seed, device=args.device)
    print(f"{data} simulated on {data.provenance['device']} in {data.provenance['wall_time_s']:.2f} s")

    # Save and read back: the file also holds the configuration, the detector and how the data were made.
    path = save(data, args.out / f"quickstart_{args.preset}_{args.medium}.h5")
    data = load(path)
    print(f"saved to {path} ({path.stat().st_size / 1e6:.2f} MB) and read back: {data}\n")
    summarise(data)

    # Hand-made features: one row per event.
    features = event_features(data)
    print(f"\nfeatures: {', '.join(features)}")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed: no figure (pip install -e '.[analysis]')")
        return
    figure = path.with_suffix(".png")
    plot_first_event(data, figure)
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
