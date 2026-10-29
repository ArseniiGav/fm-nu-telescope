"""The data commands of TASKS.md (read by examples/make_task_data.py) run, with fewer events."""

import importlib.util
from pathlib import Path

import pytest

from nutel.toy.cli import main
from nutel.toy.io import load

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "make_task_data.py"

pytestmark = pytest.mark.skipif(
    not (SCRIPT.exists() and (ROOT / "TASKS.md").exists()), reason="needs the repository checkout"
)


def _task_commands() -> list[list[str]]:
    spec = importlib.util.spec_from_file_location("make_task_data", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.task_commands()


def _small(command: list[str], data_dir: Path) -> list[str]:
    """The command with at most 8 events (one per detector at least) and every data path moved to ``data_dir``."""
    out = [str(data_dir / word[len("data/") :]) if word.startswith("data/") else word for word in command]
    if "-n" in out:
        k = out.index("-n") + 1
        n_geometries = int(out[out.index("--n-geometries") + 1]) if "--n-geometries" in out else 1
        out[k] = str(min(int(out[k]), max(8, n_geometries)))
    if out[0] == "generate":
        out += ["--device", "cpu", "--no-progress", "--quiet"]
    return out


def test_task_commands_run(tmp_path):
    commands = _task_commands()
    assert len(commands) >= 40
    assert all(c[0] in ("generate", "noise") for c in commands)
    outputs = [c[c.index("-o") + 1] for c in commands]
    assert len(set(outputs)) == len(outputs)
    assert all(o.startswith("data/") and o.endswith(".h5") for o in outputs)
    for command in commands:
        assert main(_small(command, tmp_path)) == 0, command
    for name in outputs:
        assert len(load(tmp_path / name[len("data/") :], level="pulses", paths=False)) > 0
