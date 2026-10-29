# Data format

Every run of `nutelsim generate` (or `save()` in Python) writes one HDF5 file. This page describes format version 1,
written by `nutel` 0.1.0. Below, `N` is the number of events and `K` the number of detectors in the file (1 unless a
detector family was sampled). The physics behind the stored quantities is in [physics.md](physics.md).

```text
file.h5
├── attrs: format_version, n_events, n_geometries, config (JSON), provenance (JSON)
├── geometries/  module tables, media and readouts of the K detectors      [K + 1], [M_total], [K]
├── truth/       true event parameters                                     [N]
├── photons/     photon bookkeeping per event                              [N]
├── pulses/      pulses of every event (always)                            [N + 1], [n_pulses]
├── hits/        hits of every event (level "both", alias "hits")          [N + 1], [n_hits]
├── bursts/      stochastic losses of every event (when present)           [N + 1], [n_bursts]
└── paths/       photon paths of the first events (--save-paths K)         [n_paths], [n_points]
```

## Two levels: hits and pulses

**Hits** are the detected photons and the noise hits, one row each. **Pulses** are the hits of a module merged within
the merging window (time of the first hit, or their mean time with `--pulse-time mean`; charge = number of hits),
what a detector records. The two levels mirror
the merged photons and the pulses of NuBench ([nubench.md](nubench.md)). Pulses are always written; `--level` chooses
whether the hits are kept as well (`both`, the default; `hits` is an alias of `both`) or dropped (`pulses`).

## Reading the data

In Python, `load` returns the same `SimData` object as `generate`, with NumPy arrays:

```python
from nutel.toy import load

data = load("data/toy_full_water_n1000_seed1.h5")
data.truth.energy  # [N] true energies
data.pulses.t, data.pulses.charge  # all pulses of all events, concatenated
data.config  # the SimConfig used, e.g. data.config.medium.absorption_length
data.provenance["seed"]  # seed, versions, creation time, ...

light = load("data/toy_full_water_n1000_seed1.h5", level="pulses", paths=False)  # skip the large optional groups
```

`nutelsim info FILE` prints the detectors, the configuration, the provenance and the datasets. The files are plain
HDF5, so any HDF5 reader works:

```python
import h5py

with h5py.File("data/toy_full_water_n1000_seed1.h5", "r") as f:
    t = f["pulses/t"][()]  # numpy array [n_pulses]
    print(f["pulses/t"].attrs["units"])  # ns
```

## Ragged tables and offsets

Events have different numbers of pulses, so the pulses of all events are stored one after the other, and `offset`
(int64, `N + 1` entries, `offset[0] = 0`) says where each event starts: the pulses of event `i` are the rows
`offset[i]:offset[i + 1]`. The same holds for `hits` and `bursts`, and for the module tables of `geometries` (with `K`
instead of `N`).

```python
import numpy as np

p = data.pulses
i = 7
rows = slice(p.offset[i], p.offset[i + 1])
x, z, t, q = p.x[rows], p.z[rows], p.t[rows], p.charge[rows]  # the pulses of event 7, in time order

n_pulses = np.diff(p.offset)  # pulses per event, [N]
event_of_row = np.repeat(np.arange(len(data)), n_pulses)  # event index of every pulse (also p.event_index())
total_charge = np.bincount(event_of_row, weights=p.charge, minlength=len(data))
```

`data.event(i)` returns the same slices as a view (`.truth`, `.pulses`, `.hits`, `.bursts`, `.detector`), and
`data.select(mask)` a new `SimData` with the selected events (a boolean mask or event indices).

## Root attributes

| attribute        | type   | content                                                                         |
|------------------|--------|---------------------------------------------------------------------------------|
| `format_version` | string | `"1"`                                                                           |
| `n_events`       | int64  | number of events `N`                                                            |
| `n_geometries`   | int64  | number of detectors `K`                                                         |
| `config`         | string | JSON of the complete configuration (`SimConfig.to_json()`, read back by `SimConfig.from_json`) |
| `provenance`     | string | JSON: how the file was made (see [Provenance](#provenance))                     |

`config` has the sections `layout`, `medium`, `readout`, `physics`, `emission`, `prior` and `custom_geometry` with
every parameter of [physics.md](physics.md). In a multi-detector file the medium and the readout of each detector are
those of `/geometries`, not those of `config`.

## `/geometries`: the detectors

| dataset       | dtype   | shape          | units | content                                                       |
|---------------|---------|----------------|-------|---------------------------------------------------------------|
| `offset`      | int64   | [K + 1]        |       | detector `k` has the modules `offset[k]:offset[k + 1]`        |
| `string`      | int32   | [M_total]      |       | string number                                                 |
| `module`      | int32   | [M_total]      |       | module number on its string (0 = deepest)                     |
| `x`, `z`      | float32 | [M_total]      | m     | module centre                                                 |
| `geometry_id` | S12     | [K]            |       | first 12 hex characters of the SHA-1 of the module table (cm) and the radius (mm) |
| `radius`      | float64 | [K]            | m     | module radius                                                 |
| `layout`      | string  | [K]            |       | JSON: layout name and parameters                              |
| `medium`      | string  | [K]            |       | JSON: refractive index, group index, absorption length, effective scattering length, g, background rate |
| `readout`     | string  | [K]            |       | JSON: efficiency, time jitter, noise rate (resolved, in Hz), window, merge window, pulse time |

`load` turns them into `data.detectors` (a list of `Detector`: `geometry`, `medium`, `readout`) and checks every
`geometry_id` against its table. `data.geometry(i)` and `data.detector(i)` return the detector of event `i`.

## `/truth`: event parameters

| dataset            | dtype   | units | content                                                          |
|--------------------|---------|-------|------------------------------------------------------------------|
| `event_id`         | int64   |       | event number, unique within a file (`--first-event-id` sets the first one) |
| `geometry_index`   | int32   |       | index of the event's detector in `/geometries` (0 in single-detector files) |
| `type`             | int8    |       | 0 track, 1 starting, 2 cascade, 3 atmospheric                    |
| `dir_x`, `dir_z`   | float32 |       | unit vector of the direction of travel                           |
| `zenith`           | float32 | rad   | $\arccos(-d_z)$: 0 down-going, $\pi$ up-going                    |
| `azimuth`          | float32 | rad   | 0 if the particle comes from $+x$, $\pi$ if from $-x$            |
| `angle`            | float32 | rad   | $\mathrm{atan2}(d_z, d_x)$ in $(-\pi, \pi]$               |
| `x0`, `z0`         | float32 | m     | anchor point of a through-going track (closest point to the origin) or vertex |
| `t0`               | float32 | ns    | time at the anchor point or vertex                               |
| `energy`           | float32 | GeV   | muon energy (track, atmospheric), neutrino energy (starting), cascade energy (cascade) |
| `energy_track`     | float32 | GeV   | muon energy where its light emission starts (0 for cascades)     |
| `energy_cascade`   | float32 | GeV   | energy of the vertex cascade (0 for track, atmospheric)          |
| `cascade_fraction` | float32 |       | inelasticity $y$ of starting tracks, NaN for the other types     |
| `track_length`     | float32 | m     | length of the light-emitting muon segment (0 for cascades)       |
| `s_start`, `s_end` | float32 | m     | ends of that segment along the line $\mathbf{a} + s\,\mathbf{d}$ from the anchor or vertex: entry into the generation box (0 at the vertex of a starting track) and exit or stop point; `track_length` = `s_end` - `s_start`; NaN where no muon light is emitted (cascades, lines that miss the generation box) |
| `length_in_area`   | float32 | m     | part of that segment inside the instrumented area                |
| `n_bursts`         | int32   |       | number of stochastic-loss bursts                                 |
| `energy_lost`      | float32 | GeV   | sum of the burst energies                                        |
| `stopped`          | bool    |       | true if the muon ran out of energy inside the generation box     |

## `/photons`: photon bookkeeping per event

All datasets are int32 arrays of shape [N].

| dataset                | content                                                              |
|------------------------|----------------------------------------------------------------------|
| `n_emitted`            | photons emitted                                                      |
| `n_emitted_track`, `n_emitted_burst`, `n_emitted_cascade` | photons emitted by each source      |
| `n_detected`           | photons detected by a module (before the window cut)                 |
| `n_detected_direct`    | detected photons that never scattered                                |
| `n_detected_scattered` | detected photons that scattered at least once                        |
| `n_absorbed_module`    | photons that reached a module but were not detected (efficiency < 1) |
| `n_absorbed`           | photons absorbed in the medium                                       |
| `n_escaped`            | photons that left the world box                                      |
| `n_lost`               | photons still in flight after `max_steps` steps                      |
| `n_noise`              | noise hits                                                           |
| `n_out_of_window`      | detected photons outside the readout window (dropped)                |
| `n_hits`, `n_pulses`   | hits and pulses of the event                                         |

These identities hold exactly for every event:

```text
n_emitted  = n_emitted_track + n_emitted_burst + n_emitted_cascade
n_emitted  = n_detected + n_absorbed_module + n_absorbed + n_escaped + n_lost
n_detected = n_detected_direct + n_detected_scattered
n_hits     = n_detected - n_out_of_window + n_noise
```

## `/pulses` and `/hits`

Both are ragged (`offset` [N + 1]) and sorted by event and time. The positions are those of the module centres, so
a model can read the hits of any detector without a module table.

| dataset       | dtype   | units | pulses | hits | content                                                |
|---------------|---------|-------|--------|------|--------------------------------------------------------|
| `string`      | int32   |       | yes    | yes  | string number                                          |
| `module`      | int32   |       | yes    | yes  | module number on the string                            |
| `x`, `z`      | float32 | m     | yes    | yes  | module position                                        |
| `t`           | float32 | ns    | yes    | yes  | pulse time (first or mean hit time, `pulse_time`) / hit time |
| `charge`      | int32   |       | yes    |      | number of merged hits                                  |
| `n_noise`     | int32   |       | yes    |      | noise hits among them                                  |
| `source`      | int8    |       |        | yes  | 0 track, 1 burst, 2 cascade, 3 noise                   |
| `n_scatter`   | int32   |       |        | yes  | scatterings of the photon (-1 for noise)               |
| `path_length` | float32 | m     |        | yes  | path length of the photon (NaN for noise)              |
| `is_noise`    | bool    |       |        | yes  | true for noise hits                                    |

## `/bursts`: stochastic losses

Ragged (`offset` [N + 1]), ordered along the track: `x`, `z` (m), `t` (ns) and `energy` (GeV) of every burst.

## `/paths`: photon paths (optional)

Written with `--save-paths K`: the paths of the photons of the first `K` events, as polylines flattened into one list
of points.

| dataset    | dtype   | shape          | units | content                                                  |
|------------|---------|----------------|-------|----------------------------------------------------------|
| `event_id` | int64   | [n_paths]      |       | event of each path                                       |
| `photon`   | int32   | [n_paths]      |       | number of the photon within its event                    |
| `source`   | int8    | [n_paths]      |       | 0 track, 1 burst, 2 cascade                              |
| `offset`   | int64   | [n_paths + 1]  |       | path `k` has the points `offset[k]:offset[k + 1]`        |
| `x`, `z`   | float32 | [n_points]     | m     | point coordinates                                        |
| `t`        | float32 | [n_points]     | ns    | time at the point                                        |
| `kind`     | int8    | [n_points]     |       | type of the point (table below)                          |

The group attribute `point_types` holds the codes as JSON:

| code | name            | meaning                                         |
|------|-----------------|-------------------------------------------------|
| 0    | `EMIT`          | emission (the first point of every path)        |
| 1    | `SCATTER`       | scattering in the medium                        |
| 2    | `DETECT`        | detected by a module                            |
| 3    | `ABSORB_MODULE` | reached a module but not detected               |
| 4    | `ABSORB`        | absorbed in the medium                          |
| 5    | `ESCAPE`        | left the world box                              |
| 6    | `LOST`          | still in flight after `max_steps` steps         |

The last point of a path is its fate. In Python:

```python
from nutel.toy import PointType

paths = data.paths
for k in paths.of_event(paths.event_id[0]):  # the paths of the first stored event
    x, z, t, kind = paths.polyline(k)
n_detected = (paths.fate == PointType.DETECT).sum()
```

## Multi-detector files

A file made with `--family NAME --n-geometries K` holds `K` sampled detectors in `/geometries`, and every event points
to its own with `truth.geometry_index`. Everything else is the same as in a single-detector file; the pulses and hits
of event `i` lie on the modules of `data.geometry(i)`. The family samples the layout, the medium and the readout
efficiency, time jitter and noise of every detector: these are stored per detector in `/geometries`
(`data.detector(i)`), the family's ranges in the provenance (`family`), and `nutelsim info` marks the layout, medium
and these readout values of the stored configuration as "sampled per detector". `nutelsim generate` refuses
`--family` together with `--medium`, a layout option or `--efficiency`, `--time-jitter`, `--noise-rate`; the physics,
emission and prior options, the readout window, the merge window and the pulse time apply to every detector.

```python
data = load("data/t5_family.h5")
print(data.n_geometries, [d.geometry_id for d in data.detectors])
same_detector = data.select(data.truth.geometry_index == 3)  # the events of detector 3
```

## Provenance

| key                                      | content                                                     |
|------------------------------------------|-------------------------------------------------------------|
| `seed`, `first_event_id`                 | random seed and the id of the first event                   |
| `device`, `device_name`, `torch_threads` | where the simulation ran                                    |
| `photons_per_chunk`, `n_chunks`          | chunking of the photon transport (the output depends on it) |
| `level`, `save_paths`                    | the optional output requested                               |
| `family`, `n_geometries`, `geometry_seed`, `events_per_geometry` | the detector family (`name` and `ranges`, or null) and the events per detector |
| `preset`, `medium`, `overrides`          | the CLI preset, medium (`sampled per detector` with a family) and the changed options (from `nutelsim`) |
| `package_version`, `git_commit`          | version of `nutel`; the commit when run from a git checkout |
| `python_version`, `numpy_version`, `torch_version`, `h5py_version`, `platform` | software environment |
| `created`, `wall_time_s`                 | creation time (UTC) and simulation time in seconds          |
| `command_line`                           | the full command line (from `nutelsim`)                     |

A file made by `nutelsim merge` has `merged_from` (the input files, each with its own provenance), `renumbered`,
`package_version` and `created`; a file made by `nutelsim noise` records the added noise under `augmentation`.

## Merging files

`nutelsim merge -o OUT FILE...` (or `merge(paths, out)`) concatenates files in the given order, copying block by
block so that the inputs need not fit in memory. The configurations must agree except for the detectors and the
seed. The detectors of all inputs are concatenated and de-duplicated (same geometry id, medium and readout) and
`geometry_index` is remapped, so files with different detectors merge into one multi-detector file. Merging refuses
inputs with different optional groups, inputs with the same seed (their events would repeat the same random numbers)
and inputs whose event ids overlap: give every shard its own seed and its own `--first-event-id` range, or pass
`--renumber`. [examples/cluster](../examples/cluster/README.md) shows the workflow.

## Storage details

- Every dataset has a `units` attribute (empty for counts and codes) and a `description` attribute.
- Datasets with more than 1024 elements are compressed with gzip (level 4) and the shuffle filter.
- A file is written as `<name>.part` and renamed when complete, so an interrupted job never leaves a truncated file
  under the final name.
- `load` rejects files without `format_version` and files with a newer format version than the installed package.
