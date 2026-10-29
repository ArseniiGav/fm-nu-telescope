# Generating data on a cluster

Large data sets are generated as **shards**: one job per shard, each with its own seed and its own range of event ids,
combined at the end with `nutelsim merge`. The shards run in parallel, each is reproducible from its seed, and a failed
job is simply run again. The snippets below are generic: **adapt them to your cluster** (partitions, accounts,
modules, file systems and limits differ from one cluster to the next).

## Before the first job

1. Install the package in a virtual environment on the cluster, as in the [README](../../README.md#installation) and
   [docs/install.md](../../docs/install.md): the CPU build of PyTorch for CPU nodes, a CUDA build for GPU nodes. The
   jobs must be able to read the environment.
2. Put the data on a work or scratch file system with enough space, not in a small home directory.
3. Try one small shard in an interactive session (not on the login node), from the repository root:

   ```bash
   nutelsim generate -n 1000 --seed 1000 --first-event-id 0 -o data/shards/test_shard0000.h5
   nutelsim info data/shards/test_shard0000.h5
   ```

## One shard

Shard `i` of a data set gets the seed `BASE_SEED + i` and the event ids from `i * EVENTS` on:

```bash
#!/bin/bash
# run_shard.sh I [extra nutelsim options]: generate shard I (adapt to your cluster)
set -euo pipefail
I=$1; shift
EVENTS=${EVENTS:-100000}            # events per shard
BASE_SEED=${BASE_SEED:-1000}        # use separate seed ranges for independent data sets
OUT_DIR=${OUT_DIR:-data/shards}
OUT=$(printf "%s/shard%04d.h5" "$OUT_DIR" "$I")
[ -f "$OUT" ] && { echo "exists: $OUT"; exit 0; }   # resubmitting only makes the missing shards
mkdir -p "$OUT_DIR"
nutelsim generate -n "$EVENTS" --seed $((BASE_SEED + I)) --first-event-id $((I * EVENTS)) \
  --threads "${THREADS:-4}" --no-progress -o "$OUT" "$@"
```

Every option of `nutelsim generate` can follow the shard index, e.g. `bash run_shard.sh 3 --preset scatter --medium
ice` or, for a multi-detector data set, `bash run_shard.sh 3 --family default --n-geometries 16 --geometry-seed 7`
(the same `--geometry-seed` in every shard gives every shard the same detectors).

## SLURM array (adapt to your cluster)

```bash
#!/bin/bash
#SBATCH --job-name=nutel
#SBATCH --array=0-9                 # ten shards; --array=0-99%10 runs at most ten at a time
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
##SBATCH --partition=...            # your partition, account, ...
##SBATCH --gres=gpu:1               # for a GPU node, with DEVICE=cuda below
source .venv/bin/activate
THREADS=$SLURM_CPUS_PER_TASK bash run_shard.sh "$SLURM_ARRAY_TASK_ID" --device "${DEVICE:-cpu}"
```

Submit from the repository root with `mkdir -p logs && sbatch slurm_array.sh`.

## HTCondor (adapt to your cluster)

```text
# htcondor.sub
executable     = /bin/bash
arguments      = -c "source .venv/bin/activate && THREADS=4 bash run_shard.sh $(Process) --device cpu"
initialdir     = /path/to/fm-nu-telescope
request_cpus   = 4
request_memory = 4 GB
output         = logs/nutel_$(Cluster)_$(Process).out
error          = logs/nutel_$(Cluster)_$(Process).err
log            = logs/nutel_$(Cluster).log
queue 10
```

Submit with `mkdir -p logs && condor_submit htcondor.sub`; `max_materialize` limits the jobs in the queue if your
cluster asks for it.

## Combining the shards

```bash
ls data/shards/shard*.h5 | wc -l                          # all shards there?
nutelsim merge -o data/toy_n1M.h5 data/shards/shard*.h5
nutelsim info data/toy_n1M.h5
```

`nutelsim merge` refuses shards with different configurations (apart from the detectors), shards with the same seed
and shards whose event ids overlap (`--renumber` numbers the events 0 to N - 1 instead). Shards with different
detectors merge into one multi-detector file. It does not know how many shards you meant to make, so count them
first. The merged file keeps the provenance of every shard.

## Good practice

- Measure the time and memory of one shard first (task T0.3) and size the jobs from that.
- Do not run simulations on the login node; use jobs or an interactive session.
- Use separate seed ranges for data sets that must be independent (e.g. `BASE_SEED=1000` for training and
  `BASE_SEED=2000` for testing, with fewer than 1000 shards each). The same seed gives the same events.
- Keep the logs (`logs/` is ignored by git) until the data set is checked.
