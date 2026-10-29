# Installation details

The short version is in the [README](../README.md#installation): a virtual environment, one `pip install`, the tests
and the Jupyter kernel. This page covers what the short version leaves out: Windows, the PyTorch build for your
machine (CPU, GPU, Intel Macs), `uv`, clusters and the common problems. The repository is called `fm-nu-telescope`;
the package it installs is `nutel`, with the command `nutelsim`.

## Windows

Create and activate the environment in the Command Prompt or in PowerShell; the other commands are the same as in the
README:

```bat
python -m venv .venv
.venv\Scripts\activate.bat        :: Command Prompt
.venv\Scripts\Activate.ps1        :: PowerShell
```

If `python` is not found, use `py` instead. Every `nutelsim` command of [TASKS.md](../TASKS.md) is a single line and
works in any shell; `python examples/make_task_data.py` makes all the data sets of the tasks in one go.
[WSL](https://learn.microsoft.com/windows/wsl/) gives a Linux shell on Windows if you prefer bash.

## Which PyTorch

A plain `pip install -e ".[analysis,dev]"` installs the PyTorch build that PyPI serves for your machine. It works
everywhere: the simulation runs on any CPU and uses an NVIDIA GPU automatically when PyTorch sees one
(`--device auto`, the default). The size of the download is the only reason to choose a build by hand:

| machine                              | what PyPI installs                              | smaller alternative                       |
|--------------------------------------|-------------------------------------------------|-------------------------------------------|
| Linux laptop or CPU node             | CUDA build with NVIDIA libraries, several GB    | CPU build, about 200 MB (below)           |
| Windows                              | CPU build                                       | none needed                               |
| Mac with Apple silicon               | CPU build                                       | none needed                               |
| Mac with an Intel processor          | nothing recent (see below)                      |                                           |
| machine with an NVIDIA GPU           | CUDA build                                      | the build matching your driver (below)    |

Install PyTorch **before** the package to choose a build; `pip` then keeps it:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU build, about 200 MB
python -m pip install -e ".[analysis,dev]"
```

**NVIDIA GPU.** `nvidia-smi` prints the newest CUDA version the driver supports in its header. Install the build for
that CUDA major version, e.g. `--index-url https://download.pytorch.org/whl/cu126` for CUDA 12 or `cu130` for
CUDA 13. The [PyTorch selector](https://pytorch.org/get-started/locally/) lists the current builds. A CUDA build takes
several GB: put the environment on a file system with enough space. The toy is two-dimensional and cheap, so a GPU is
never needed for stage A.

**Macs with an Intel processor.** PyTorch 2.2.2 is the last release for them (Python 3.10 to 3.12) and it needs
NumPy 1:

```bash
python -m pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[analysis,dev]" "numpy<2"
```

**Check what you got:** `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

## With uv

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[analysis,dev]"
python -m ipykernel install --user --name fm-nu-telescope --display-name "Python (fm-nu-telescope)"
```

An environment made by `uv venv` has no pip of its own: use `uv pip install ...` for every later install (or create it
with `uv venv --seed`), since `python -m pip` fails there and a bare `pip` would install into another environment.

## On a cluster

A laptop is enough for stage A; a cluster helps for large samples and for the multi-detector data sets of the later
stages (see [examples/cluster](../examples/cluster/README.md)). Load the cluster's Python module (if there is one)
before you create the environment. Put the environment (about 1.5 GB with the CPU build of PyTorch) and the data (0.5
GB for the tasks) on a work or scratch file system, not in a small home directory. Run the simulations in a job or an
interactive session, not on the login node.

## Notebooks and kernels

The README registers the environment as the Jupyter kernel `Python (fm-nu-telescope)`; the notebooks of this
repository ask for that kernel by name, so JupyterLab, VS Code and JupyterHub pick it up. `python -m jupyter lab`,
run in the activated environment, starts JupyterLab from this environment (a bare `jupyter` may belong to another
Python installation). If a notebook reports `ModuleNotFoundError: No module named 'nutel'` or `'torch'`, it runs on
another kernel: choose `Python (fm-nu-telescope)` in the kernel menu (JupyterLab: Kernel > Change Kernel; VS Code: the
kernel picker in the top right). In VS Code, open the repository folder itself, so that it finds `.venv`.

## Common problems

- `No module named 'nutel'` or `'torch'` in a notebook: the kernel is not the environment, see above.
- `nutelsim: command not found`: the environment is not activated; `python -m nutel.toy` is the same command.
- `No space left on device` during `pip install`: on Linux, PyPI serves a CUDA build of PyTorch of several GB; the
  CPU build (`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`, before the package)
  is about 200 MB. Free space (`python -m pip cache purge` empties the pip cache) or create the environment on another
  disk and link it: `python3 -m venv /path/with/space/fm-nu-telescope && ln -s /path/with/space/fm-nu-telescope .venv`.
- `pytest` not found: it comes with the `dev` extra, `pip install -e ".[analysis,dev]"`.
- In every new terminal, activate the environment again: `cd fm-nu-telescope && source .venv/bin/activate`.
