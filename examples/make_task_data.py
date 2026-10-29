"""Make the data sets of TASKS.md ("Data for T1 to T5") on any operating system.

The commands are read from the bash block under the heading "Data for T1 to T5" of TASKS.md, so the script and the
task sheet cannot disagree, and run through the entry point of the command line interface, so they write the same
files as the commands typed by hand. Run it from anywhere inside the repository::

    python examples/make_task_data.py              # every file of the block; files that exist are kept
    python examples/make_task_data.py --overwrite  # make every file again
    python examples/make_task_data.py --dry-run    # only print the commands
    python examples/make_task_data.py --only t3_   # only the files whose name contains "t3_"

All 44 files take about 15 minutes on 8 cores of a server CPU and 0.5 GB in ``data/``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repository root: every path of TASKS.md is relative to it
TASKS = ROOT / "TASKS.md"
HEADING = "## Data for T1 to T5"


def task_commands(tasks: Path = TASKS) -> list[list[str]]:
    """Return the ``nutelsim`` commands of the data block of TASKS.md, in order, without the leading ``nutelsim``."""
    lines = tasks.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(HEADING)
    except ValueError:
        raise RuntimeError(f"{tasks} has no heading {HEADING!r}") from None
    fence = next(k for k in range(start, len(lines)) if lines[k].strip() == "```bash")
    end = next(k for k in range(fence + 1, len(lines)) if lines[k].strip() == "```")
    commands = []
    for line in lines[fence + 1 : end]:
        words = shlex.split(line, comments=True)
        if not words:
            continue
        if words[0] != "nutelsim":
            raise RuntimeError(f"unexpected line in the data block of {tasks}: {line!r}")
        commands.append(words[1:])
    return commands


def output_of(command: list[str]) -> Path:
    """Return the output file of a command (the value after ``-o``)."""
    return Path(command[command.index("-o") + 1])


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Make the data sets of TASKS.md (T1 to T5) on any operating system.")
    parser.add_argument("--overwrite", action="store_true", help="make every file again, also those that exist")
    parser.add_argument("--dry-run", action="store_true", help="only print the commands")
    parser.add_argument("--only", metavar="TEXT", help="only the commands whose output file name contains TEXT")
    return parser.parse_args()


def main() -> int:
    """Run the commands of the data block; return the exit status of the first command that fails, else 0."""
    args = parse_args()
    os.chdir(ROOT)
    commands = task_commands()
    if args.only:
        commands = [c for c in commands if args.only in output_of(c).name]
    print(f"{len(commands)} data sets for TASKS.md, written to {ROOT / 'data'}")
    if args.dry_run:
        for command in commands:
            print("nutelsim " + shlex.join(command))
        return 0

    from nutel.toy.cli import main as nutelsim

    start = time.perf_counter()
    for command in commands:
        line = "nutelsim " + shlex.join(command)
        target = output_of(command)
        if target.exists():
            if not args.overwrite:
                print(f"exists, kept: {target}")
                continue
            target.unlink()
        print(line, flush=True)
        quiet = ["--no-progress"] if command[0] == "generate" else []
        status = nutelsim([*command, *quiet])
        if status:
            print(f"failed with exit status {status}: {line}", file=sys.stderr)
            return int(status)
    print(f"done in {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
