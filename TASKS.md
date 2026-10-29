# Tasks: Light in the Deep

These tasks are about the toy neutrino telescope: what we simulate, how the Cherenkov light of a muon or a cascade
reaches the modules, what every optical process and every detector effect does to the data, and how far simple
estimators get from the pulses back to the event. They prepare the next stages of the project, in which neural
networks learn to reconstruct events, first on the toy and then on NuBench.

- **Core tasks:** T0 to T4. **Bonus:** T5.
- The question IDs (T0.1, T2.3, ...) are fixed: we go through the answers by ID together.
- Background: the model and its derivations are in [docs/physics.md](docs/physics.md), the file layout in
  [docs/data_format.md](docs/data_format.md), NuBench in [docs/nubench.md](docs/nubench.md), and
  [notebooks/00_quickstart.ipynb](notebooks/00_quickstart.ipynb) is a guided first look at the data.
- **Where to run.** A laptop is enough (Linux, macOS or Windows; CPU only). Every `nutelsim` command below is a single
  line and works in any shell, and `python examples/make_task_data.py` makes all the data sets of T1 to T5 in one go.

## What to hand in

1. **Figures** (PDF or PNG) with the question ID in the file name, e.g. `T2.2_light_vs_distance.pdf`.
2. **Short answers**: a few sentences per question ID, with the numbers you measured, in one Markdown file or notebook.
3. **Your code** (scripts or notebooks) and the seeds of your data sets, so that every figure can be made again.

Make your own plots too: whatever you find telling, surprising or useful. T1.5 and T2.6 ask for your own proposals,
but every task welcomes more.

**Figure checklist.** A figure should be readable on its own: axis labels with units, a legend that does not hide the
data, and a title or caption that says what is shown (preset, medium, event type, number of events, energy range).
Show statistical uncertainties where they matter (e.g. the error of the mean in each bin), use a log axis when values
span several orders of magnitude, and draw spatial displays with equal axes (`ax.set_aspect("equal")`).

**Where to work.** Keep your scripts on your own branch of the repository (`git switch -c <your-name>`) or in a
folder outside it. Data files are not committed: `data/` and `*.h5` are ignored by git.

## Notation

| symbol | meaning |
|--------|---------|
| $\mathbf{d} = (d_x, d_z)$ | direction of travel (`data.truth.dir_x`, `data.truth.dir_z`) |
| $\alpha = \mathrm{atan2}(d_z, d_x)$ | direction angle (`data.truth.angle`, rad) |
| $\theta_\mathrm{zen}$, $\phi_\mathrm{az}$ | zenith (0 = down-going) and azimuth (0 or $\pi$) of where the particle comes from (`zenith`, `azimuth`) |
| $\mathbf{a} = (x_0, z_0)$, $t_0$ | anchor point of a track (closest point to the origin) or vertex, and the time there (`x0`, `z0`, `t0`) |
| $b$ | signed impact parameter of a track, $\mathbf{a} = b\,(-d_z, d_x)$ |
| $E$ | energy (`data.truth.energy`: muon, neutrino or cascade energy by type) |
| $y$ | inelasticity of a starting track (`data.truth.cascade_fraction`) |
| $L$ | length of the light-emitting muon segment (`track_length`; inside the instrumented area: `length_in_area`) |
| $n$, $\theta_c$, $c_n$ | refractive index, Cherenkov angle ($\cos\theta_c = 1/n$), speed of light in the medium $c/n$ (`data.detector(i).medium`) |
| $\lambda_a$, $\lambda_\mathrm{eff}$, $g$ | absorption length, effective scattering length, scattering anisotropy |
| $s$, $\rho$ | coordinate along the track of the foot of the perpendicular from a module, and the module's distance from the track |
| $s_e$; $s_\mathrm{start}$, $s_\mathrm{end}$ | coordinate along the track where the direct light seen by a module is emitted (`emission_coordinate`); ends of the emitting segment (`s_start`, `s_end`, $L = s_\mathrm{end} - s_\mathrm{start}$) |
| $t_\mathrm{dir}$ | arrival time of direct Cherenkov light (`expected_arrival_time`) |
| $t_\mathrm{res} = t - t_\mathrm{dir}$ | time residual of a pulse or hit (`time_residuals`) |
| $q$, $Q$ | charge of a pulse (number of merged hits), total charge of an event (`total_charge` of `event_features`) |
| $\psi$ | angular error between a reconstructed and the true direction, in $[0°, 180°]$ (`angular_error`) |

---

## T0 Set-up (about 1 hour)

**Goal:** a working installation, a first look at one event and its truth, the cost of a simulation, and the map
between the toy and NuBench.

**T0.1** Clone the repository, create a virtual environment, install the package, run the tests and generate a first
small data set.

```bash
git clone https://github.com/ArseniiGav/fm-nu-telescope.git
cd fm-nu-telescope
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[analysis,dev]"
python -m pytest                 # ends with "... passed, ... deselected": the deselected ones are the slow tests
python -m ipykernel install --user --name fm-nu-telescope --display-name "Python (fm-nu-telescope)"
nutelsim presets                 # the physics ladder, the media and the default configuration
nutelsim generate -n 1000 --seed 1          # -> data/toy_full_water_n1000_seed1.h5
```

The repository is `fm-nu-telescope`; the package it installs is `nutel` (the toy is `nutel.toy`, the command line
`nutelsim`, the same as `python -m nutel.toy`). Windows, a smaller CPU-only PyTorch, Intel Macs and common problems:
[docs/install.md](docs/install.md).

**T0.2** One event: inspect the file, then pick one event and list its truth, its pulses (and hits), and the module
table of its detector. Draw it.

```bash
nutelsim info data/toy_full_water_n1000_seed1.h5
nutelsim display data/toy_full_water_n1000_seed1.h5 -i 0 -o T0.2_event0.png
```

```python
from nutel.toy import load

data = load("data/toy_full_water_n1000_seed1.h5")
print(data)  # SimData(1000 events, 1 geometry, ... pulses, with hits, bursts)
event = data.event(0)
print(event.type_name, event.truth)  # type, direction, anchor or vertex, t0, energies, ...
print(event.pulses.t, event.pulses.charge)  # pulse times and charges of this event
geometry = data.geometry(0)  # the module table: geometry.string, geometry.module, geometry.x, geometry.z
print(geometry.describe())
```

Which modules saw light, in which order, and does that agree with the true direction?

**T0.3** Timing and throughput. Time the generation of a few thousand events on your computer for every preset, both
media and every event type. How does the time scale with the number of events $N$? Which events are expensive, and
why? Optional: compare thread counts (`--threads`) and, if you have one, an NVIDIA GPU (`--device cuda`).

```python
from nutel.toy import generate, preset

for name in ("straight", "absorb", "scatter", "full"):
    for medium in ("water", "ice"):
        data = generate(2000, config=preset(name, medium=medium), seed=3, device="cpu")
        print(name, medium, f"{data.provenance['wall_time_s']:.2f} s")
```

**T0.4** The NuBench dictionary. Make a table: for every field of the toy (truth, hits, pulses, geometry, detector
parameters), the corresponding field or quantity of NuBench, its units and convention, or "none". Use
[docs/nubench.md](docs/nubench.md) and the paper ([arXiv:2511.13111](https://arxiv.org/abs/2511.13111)); no download
is needed. Which toy quantities have no NuBench counterpart (and the other way round)? Where do the conventions differ
(the pulse time, the charge, the noise, the angles)? The toy's zenith and azimuth say where the particle comes from;
NuBench's describe its direction of travel. Write down how the angles convert.
*Optional (about 30 minutes more):* check the conversion on real NuBench events. One test chunk of Triangle (25 000
events, 55 MB; the download recipe and `pip install pyarrow` are in
[docs/nubench.md](docs/nubench.md#one-chunk-without-the-full-archive)) is enough: for a few hundred high-energy CC
events with many pulses, does the time-ordered axis of the pulses (a LineFit in three dimensions) point along the
direction built from `initial_state_zenith` and `initial_state_azimuth`, or against it?

**Deliverables:** the summary line of the tests and one event display (T0.1, T0.2); a plot of the simulation time
against $N$ per preset and medium (T0.3); the dictionary as a table (T0.4) and, if you did it, the fraction of
NuBench events whose LineFit axis agrees with the converted direction.

**Hints**
- `nutelsim generate` prints the time of the simulation at the end; it is also stored in every file as
  `data.provenance["wall_time_s"]`.
- `preset("full", event_type="track")` simulates one event type only (`--event-type track` on the command line).
- For the scaling, a log-log plot of the time against $N$ shows a power law as a straight line. The first call in a
  process is slower (start-up), so time more than one $N$.

## Data for T1 to T5

With the installation working, make the data of the next tasks. All files go to `data/` in the repository root, and
every path in this document is relative to the repository root (a notebook in `notebooks/` reads the same file as
`../data/t1_scatter.h5`). `python examples/make_task_data.py` runs exactly these commands on any operating system (it
keeps files that already exist; `--overwrite` makes them again, `--dry-run` only prints them). Together they take
about 25 minutes on 8 cores of a server CPU (longer on a laptop) and 0.5 GB of disk. On a cluster, run them in a job
or an interactive session, not on the login node (see
[examples/cluster](examples/cluster/README.md)). Files made with the same seed share the directions, energies and
times of their events, which makes comparisons between presets or detectors sharper; the positions (anchor points and
vertices) scale with the size of the detector, so they are shared only between files of the same layout. Such paired
samples are made for differences, not for pooling. The light of T1 and T2 is simulated without noise (presets
`straight`, `absorb` and `scatter`); the noise and the full preset enter with T3.3.

```bash
# T1, T2.3-T2.6: all event types, absorption and scattering, no noise
nutelsim generate --preset scatter -n 50000 --seed 11 -o data/t1_scatter.h5
# T2.1-T2.2: straight light and absorption, exact Cherenkov angle
nutelsim generate --preset straight --event-type track --cherenkov-spread 0 -n 2000 --seed 21 -o data/t2_straight_track.h5
nutelsim generate --preset absorb --event-type track --cherenkov-spread 0 -n 2000 --seed 21 -o data/t2_absorb_track.h5
nutelsim generate --preset straight --event-type cascade --fixed-vertex 3.7 -2.9 --fixed-energy 1000 -n 2000 --seed 22 -o data/t2_straight_cascade.h5
nutelsim generate --preset absorb --event-type cascade --fixed-vertex 3.7 -2.9 --fixed-energy 1000 -n 2000 --seed 22 -o data/t2_absorb_cascade.h5
# T2.3: four decades of energy, tracks and cascades (the 100 TeV end has 10^5-10^6 photons per event)
nutelsim generate --preset scatter --event-type track --track-energy-range 10 100000 -n 2000 --seed 23 -o data/t2_track_he.h5
nutelsim generate --preset scatter --event-type cascade --cascade-energy-range 10 100000 -n 2000 --seed 24 -o data/t2_cascade_he.h5
# T3: the physics ladder in water and ice, the same events everywhere
nutelsim generate --preset straight --medium water -n 20000 --seed 31 -o data/t3_straight_water.h5
nutelsim generate --preset absorb --medium water -n 20000 --seed 31 -o data/t3_absorb_water.h5
nutelsim generate --preset scatter --medium water -n 20000 --seed 31 -o data/t3_scatter_water.h5
nutelsim generate --preset full --medium water -n 20000 --seed 31 -o data/t3_full_water.h5
nutelsim generate --preset straight --medium ice -n 20000 --seed 31 -o data/t3_straight_ice.h5
nutelsim generate --preset absorb --medium ice -n 20000 --seed 31 -o data/t3_absorb_ice.h5
nutelsim generate --preset scatter --medium ice -n 20000 --seed 31 -o data/t3_scatter_ice.h5
nutelsim generate --preset full --medium ice -n 20000 --seed 31 -o data/t3_full_ice.h5
# T3.3: four times more noise (a KM3NeT-like module), and noise added afterwards (the augmentation)
nutelsim generate --noise-rate 200000 -n 20000 --seed 31 -o data/t3_full_water_noise200k.h5
nutelsim noise data/t3_scatter_water.h5 --rate 50000 --seed 33 -o data/t3_scatter_water_noise.h5
# T3.4-T3.5: tracks with and without stochastic losses
nutelsim generate --event-type track -n 20000 --seed 35 -o data/t3_track.h5
nutelsim generate --event-type track --no-stochastic -n 20000 --seed 35 -o data/t3_track_nostochastic.h5
nutelsim generate --event-type track --track-energy-range 100 100000 -n 5000 --seed 36 -o data/t3_track_he.h5
# T4: events with photon paths, a noise-only readout, one event in several layouts and media
nutelsim generate -n 100 --seed 41 --save-paths 10 -o data/t4_events.h5
nutelsim generate --track-yield 0 -n 10 --seed 42 -o data/t4_noise_only.h5
nutelsim generate --preset scatter --medium water --event-type track --fixed-angle -60 --fixed-vertex 0 0 --fixed-energy 1000 --fixed-t0 1500 -n 1 --seed 43 --save-paths 1 -o data/t4_paths_water.h5
nutelsim generate --preset scatter --medium ice --event-type track --fixed-angle -60 --fixed-vertex 0 0 --fixed-energy 1000 --fixed-t0 1500 -n 1 --seed 43 --save-paths 1 -o data/t4_paths_ice.h5
nutelsim generate --event-type track --fixed-angle -60 --fixed-vertex 0 0 --fixed-energy 1000 --fixed-t0 1500 -n 1 --seed 45 -o data/t4_same_default.h5
nutelsim generate --layout core --event-type track --fixed-angle -60 --fixed-vertex 0 0 --fixed-energy 1000 --fixed-t0 1500 -n 1 --seed 45 -o data/t4_same_core.h5
nutelsim generate --layout irregular --event-type track --fixed-angle -60 --fixed-vertex 0 0 --fixed-energy 1000 --fixed-t0 1500 -n 1 --seed 45 -o data/t4_same_irregular.h5
nutelsim generate --medium ice --event-type track --fixed-angle -60 --fixed-vertex 0 0 --fixed-energy 1000 --fixed-t0 1500 -n 1 --seed 45 -o data/t4_same_ice.h5
nutelsim generate --family default --n-geometries 6 --geometry-seed 45 --event-type track --fixed-angle -60 --fixed-vertex 0 0 --fixed-energy 1000 --fixed-t0 1500 -n 6 --seed 45 -o data/t4_same_family.h5
# T5.1-T5.2: one event repeated, and the same line travelled the other way
nutelsim generate --event-type track --fixed-angle -60 --fixed-vertex 0 0 --fixed-energy 1000 --fixed-t0 1500 -n 5000 --seed 51 -o data/t5_same_event.h5
nutelsim generate --event-type track --fixed-angle -60 --fixed-vertex 0 0 --fixed-energy 1000 --fixed-t0 1500 --photon-statistics fixed --no-stochastic -n 5000 --seed 51 -o data/t5_same_event_fixed.h5
nutelsim generate --event-type track --fixed-angle 120 --fixed-vertex 0 0 --fixed-energy 1000 --fixed-t0 1500 -n 5000 --seed 52 -o data/t5_flipped.h5
# T5.3: detector design, tracks only
nutelsim generate --event-type track -n 10000 --seed 53 -o data/t5_design_default.h5
nutelsim generate --event-type track --string-spacing 25 -n 10000 --seed 53 -o data/t5_design_ds25.h5
nutelsim generate --event-type track --string-spacing 100 -n 10000 --seed 53 -o data/t5_design_ds100.h5
nutelsim generate --event-type track --n-strings 4 -n 10000 --seed 53 -o data/t5_design_4strings.h5
nutelsim generate --event-type track --n-strings 16 -n 10000 --seed 53 -o data/t5_design_16strings.h5
nutelsim generate --event-type track --layout core -n 10000 --seed 53 -o data/t5_design_core.h5
nutelsim generate --event-type track --n-strings 16 --modules-per-string 6 --module-spacing 40 -n 10000 --seed 53 -o data/t5_design_16x6.h5
# T5.4: detector effects, one at a time, tracks only
nutelsim generate --event-type track --time-jitter 0 -n 10000 --seed 54 -o data/t5_jitter0.h5
nutelsim generate --event-type track --time-jitter 3 -n 10000 --seed 54 -o data/t5_jitter3.h5
nutelsim generate --event-type track --time-jitter 10 -n 10000 --seed 54 -o data/t5_jitter10.h5
nutelsim generate --event-type track --efficiency 0.5 -n 10000 --seed 54 -o data/t5_eff0.5.h5
nutelsim generate --event-type track --efficiency 0.25 -n 10000 --seed 54 -o data/t5_eff0.25.h5
nutelsim generate --event-type track --noise-rate 0 -n 10000 --seed 54 -o data/t5_noise0.h5
nutelsim generate --event-type track --noise-rate 200000 -n 10000 --seed 54 -o data/t5_noise200k.h5
# T5.5: atmospheric muons
nutelsim generate --event-type atmospheric -n 20000 --seed 55 -o data/t5_atmospheric.h5
# T5.6: a multi-detector data set, 16 sampled detectors x 2000 tracks
nutelsim generate --family default --n-geometries 16 --geometry-seed 56 --event-type track -n 32000 --seed 56 -o data/t5_family.h5
```

Every file is read with `load`:

```python
from nutel.toy import load

data = load("data/t1_scatter.h5")
data.truth.type, data.truth.angle, data.truth.energy, data.truth.t0  # [N]
data.pulses.offset, data.pulses.t, data.pulses.charge  # ragged: event i has rows offset[i]:offset[i + 1]
data.hits.t, data.hits.source, data.hits.n_scatter  # every hit (signal and noise)
data.photons.n_emitted, data.photons.n_detected  # photon bookkeeping per event
```

## T1 What do we simulate? The prior (about 2 hours)

**Goal:** know the distribution of the event parameters, how a flux of tracks crosses a detector, and how injection
differs from physics.

**T1.1** Isotropic lines. Tracks are drawn with a uniform direction angle and a uniform impact parameter $b$. Plot the
distributions of $\alpha$, $b$ and the anchor points, and explain why a uniform $b$ (and not, for example, a uniform
point in a disc) is the isotropic flux[^flux]. Then derive the distribution of the chord length of a line through the
instrumented area (a rectangle of $350 \times 220$ m$^2$)[^chord], and compare it with `length_in_area` of the
tracks. What fraction of the generated tracks crosses the instrumented area at all?

**T1.2** Zenith. Plot the zenith distributions of the four event types. Why are $\theta_\mathrm{zen}$ (and not
$\cos\theta_\mathrm{zen}$, as in three dimensions) uniform for the neutrino-induced events? The atmospheric muons
follow the distribution of single muons at 2 km water equivalent (see [docs/physics.md](docs/physics.md)): why are
they concentrated near the vertical? In a real telescope the
Earth shields the detector from below: explain qualitatively what it removes (atmospheric muons) and what it does to
neutrinos at high energies. Why are atmospheric muons down-going only?

**T1.3** Acceptance. For tracks, plot the fraction of events with at least $k$ pulses ($k$ = 1, 3, 10) as a function of
$\cos\theta_\mathrm{zen}$ (or $\alpha$), of $b$ and of $E$. Turn it into the effective width
$W_\mathrm{eff} = 2 R_\mathrm{gen} \, N_\mathrm{sel}/N_\mathrm{gen}$ (the two-dimensional effective area). Which
directions does the detector prefer, and why?

**T1.4** Energy. The energies are injected log-uniform. Reweight the tracks to a physical spectrum $E^{-2}$ [^weights]
and compare the distributions of $E$ and of the total charge $Q$ before and after the reweighting. How many events
does the reweighted sample effectively contain?

**T1.5** Your own proposal: at least one more question about the prior (for example the starting-track energy split
$y$ and its mean, the vertex distribution, or $t_0$) with a figure.

**Deliverables:** the prior histograms (T1.1, T1.2); the measured chord-length distribution with your prediction on
top (T1.1); the acceptance or effective width against direction and energy (T1.3); the reweighted spectra (T1.4).

**Hints**
- The truth of `data/t1_scatter.h5` has everything; select a type with `data.truth.type == 0` (track) and so on, or
  `data.select(mask)`. The impact parameter is $b = x_0 (-d_z) + z_0 d_x$.
- Some low-energy muons run out of energy and stop before they leave the detector (`data.truth.stopped`); their
  `length_in_area` is shorter than the chord. Use the tracks with `stopped == False` for the chord-length comparison
  (`track_length` is the length of the whole emitting segment, inside the generation box).
- The number of pulses per event is `data.photons.n_pulses` (or `np.diff(data.pulses.offset)`).
- For quantities against $E$, use logarithmic bins and plot the fraction in each bin with its binomial error
  $\sqrt{f(1-f)/N}$.

[^flux]: A line is fixed by its direction and its signed distance from the origin. Imagine a wide parallel beam of
    lines with a fixed direction, spaced evenly across the beam: which variable is uniform? Now rotate the beam.
[^chord]: For a fixed direction, how wide is the rectangle seen from that direction, and how does the chord length
    change as the line is shifted across it? You should find a trapezoid. Check your mean against Cauchy's formula
    $\langle L \rangle = \pi A / P$; the full result is in [docs/physics.md](docs/physics.md#tracks-isotropic-lines-through-the-detector).
[^weights]: The weight of an event is the ratio of the wanted density to the injected one,
    $w \propto E^{-2} / p_\mathrm{inj}(E)$. The effective number of events is $(\sum w)^2 / \sum w^2$.

## T2 Light and the detector response (4 to 5 hours)

**Goal:** understand where and when the Cherenkov light arrives, how much of it there is, and the event as a set of
pulses.

**T2.1** Cherenkov geometry. Derive the arrival time of direct Cherenkov light from a track at a module,
$t_\mathrm{dir} = t_0 + \bigl(s + \rho\sqrt{n^2 - 1}\bigr)/c$ [^arrival]. Check it on `data/t2_straight_track.h5`
(straight light, exact Cherenkov angle, no jitter): plot the residual $t - t_\mathrm{dir}$ of the first pulse of every
hit module, computed with your own code and with `expected_arrival_time`. Explain the width and the sign of what
remains (the module is a disc of radius 0.5 m). Why does only one of the two rays reach a given module?

**T2.2** Light versus distance. From `data/t2_straight_cascade.h5` and `data/t2_absorb_cascade.h5` (cascades of 1 TeV
at the vertex (3.7, -2.9) m, near the centre of the detector), plot the mean number of photons per module against the
distance $\rho$ from the vertex; from the two track files, against the perpendicular distance from the track. Derive
the expected laws for a point and a line source in two dimensions, with and without absorption[^distance], fit them,
and recover $\lambda_a$. Modules are opaque: a module behind another one on the same ray gets no direct light. Plot
the light of the cascades against the angle between the cascade axis and the direction to the module: where is the
maximum? What changes in three dimensions?

**T2.3** Hits and charge. On `data/t1_scatter.h5`: the number of pulses, the number of hit modules and the total charge
$Q$ against the energy, for every event type. The charge of a pulse: what is its distribution, and why is it mostly 1?
When are charges above 1 frequent?
Then over four decades: in `data/t2_track_he.h5` and `data/t2_cascade_he.h5` the energies are log-uniform between
10 GeV and 100 TeV. Plot $Q$ against $E$ for both on log axes. For cascades the slope is 1 at every energy. For tracks
the light is flat at low energy and grows above a critical energy: find it, and relate it to the energy loss of the
muon, $-\mathrm{d}E/\mathrm{d}x = a + bE$ (at which energy do the stochastic losses equal the ionisation loss?). What
happens to tracks below about 50 GeV, and why does the 100 TeV end take so much longer to simulate?

**T2.4** The event as a set. Compute the hand-made features of every event with `event_features` (number of hits,
pulses, modules and strings, $Q$, first time and duration, the charge-weighted centre and its extent, the time-ordered
axis and its speed) and plot their distributions by event type.

```python
import pandas as pd
from nutel.toy import event_features, load

data = load("data/t1_scatter.h5")
table = pd.DataFrame(event_features(data))
table["type"] = data.truth.type_name
```

**T2.5** Which feature knows what? Correlate every feature with the true $E$, $\alpha$, $\theta_\mathrm{zen}$, $b$ and
the type (a matrix of rank correlations, or scatter plots). Which features carry the direction, which the energy, which
the event type? Is there a feature that knows $t_0$?

**T2.6** At least three proposals of your own. For example: the time-ordered axis against the true direction; the
speed of the LineFit and why it is not $c$; the direct and scattered fractions of the light against distance; the
image of an event on the module grid (`image(data)`); the light of the vertex cascade and of the muon of starting
tracks.

**Deliverables:** one figure per question (T2.6: at least three), each with a short interpretation; the derivations
of T2.1 and T2.2 in a few lines.

**Hints**
- The files of T1 and T2 have no noise, no time jitter and no stochastic losses (presets `straight`, `absorb`,
  `scatter`): every hit is light. Noise enters in T3.3.
- Module positions and the truth of an event: `data.geometry(i).positions`, `data.event(i).truth`. The refractive
  index of an event's detector is `data.detector(i).medium.n`.
- `time_residuals(data)` gives $t - t_\mathrm{dir}$ for every pulse of a data set (`level="hits"` for every hit). By
  default it measures $t_\mathrm{dir}$ to the surface of the module; `to_surface=False` uses its centre.
- In T2.2, "photons per module" means hits per module per event, averaged over events; count modules without light as
  zero. With absorption only, $\exp(-\ell/\lambda_a)$ depends on the photon path $\ell$, which for a track is not
  $\rho$.
- In T2.2 the muon emits light only on its segment, from `data.truth.s_start` to `data.truth.s_end` along
  $\mathbf{a} + s\,\mathbf{d}$: keep only the module-event pairs whose Cherenkov emission point $s_e$ lies on it
  (`direct_light_possible`, with `emission_coordinate` for $s_e$). The farther a module is from the track, the farther
  back along the track its light starts, so far modules often lie beyond the reach of the segment: this, not the
  distance law, is why the light of a finite track seems to fall with distance in two dimensions.
- Shadowed modules (behind another one as seen from the emission point or the vertex, fully or in part): drop these
  pairs too (`shadowed`), for the cascades and the tracks. The same selections apply when you compare with your
  prediction in T3.2.

[^arrival]: Write the time at which light emitted at the coordinate $s_e$ along the track reaches the module, and find
    the $s_e$ that makes it earliest (Fermat's principle). Which angle does the earliest photon make with the track?
    Use $\cos\theta_c = 1/n$.
[^distance]: Count how many of the photons emitted into a small angle hit a disc of diameter $2r_m$ at the distance
    $\rho$. For a track, which piece of the track sends light that hits the module, and how long is it?

## T3 Switch the physics on step by step (about 4 hours)

**Goal:** see what every optical process and every readout effect does to the data, in water and in ice, and how
simple estimators cope.

**T3.1** For every preset and both media (the `t3_*` files): the direct and scattered fractions of the detected light
(`data.photons.n_detected_direct`, `n_detected_scattered`), the photon fates, and the distribution of the time
residuals of the hits. Which process changes the amount of light most, which the timing?

**T3.2** The late tail. Look at the residuals of the first pulse of each module and of all hits: how long is the tail in
water and in ice, and how does it grow with the distance from the track? Compare the measured mean number of photons
per module with your prediction of T2.2: where does the straight-line prediction fail once scattering is on? Are the
residuals of the first pulses Gaussian (pulls with the jitter of 1 ns)?

**T3.3** Noise and cleaning. The noise rate follows the medium: 50 kHz per module in water (radioactive $^{40}$K and
bioluminescence), 0.5 kHz in ice. In `data/t3_full_water.h5`, `data/t3_full_water_noise200k.h5` and
`data/t3_full_ice.h5`: how many noise hits per event do you expect[^noise], how many are there, and what fraction of
the pulses do they make? Clean the events with a time window around the light, with the
causality condition $|t_i - t_j| \le |\mathbf{r}_i - \mathbf{r}_j|/c + \text{slack}$, and with a locality condition
(`causal_cleaning`); plot the fraction of noise kept and of signal lost against the slack. Then compare
`data/t3_scatter_water_noise.h5` (noise added afterwards, the NuBench augmentation) with the noise simulated in
`data/t3_full_water.h5`.

**T3.4** Naive estimators. For the tracks: reconstruct the direction with `linefit` (the least-squares line of the
pulse positions against time) and the energy from $Q$ with one calibration constant (`fit_charge_calibration`,
`energy_from_charge`). Plot the bias and the resolution of both against $\theta_\mathrm{zen}$, $E$, the preset and the
medium: the median angular error $\psi$ and the median and the central 68 % of $\log_{10}(E_\mathrm{reco}/E)$.

```python
import numpy as np
from nutel.toy import angular_error, causal_cleaning, linefit_events, load

data = load("data/t3_full_water.h5")
p = data.pulses
tracks = data.truth.type == 0
fit = linefit_events(p.offset, p.x, p.z, p.t, weights=p.charge)  # one LineFit per event, all pulses
psi = angular_error(fit.angle, data.truth.angle)  # degrees
print(f"median angular error of the tracks, all pulses: {np.nanmedian(psi[tracks]):.1f} deg")

keep = causal_cleaning(p)  # mask over the pulses (T3.3)
offset = np.concatenate([[0], np.cumsum(np.bincount(p.event_index()[keep], minlength=len(data)))])
fit = linefit_events(offset, p.x[keep], p.z[keep], p.t[keep], weights=p.charge[keep])
psi = angular_error(fit.angle, data.truth.angle)
print(f"median angular error of the tracks, cleaned pulses: {np.nanmedian(psi[tracks]):.1f} deg")
```

Why does the noise ruin LineFit in water, and why much less in ice?

**T3.5** Energy losses. Compare `data/t3_track.h5` and `data/t3_track_nostochastic.h5` (bursts off; the ionisation
loss stays): the burst signatures in the light along a track (`data.bursts`), $Q$ against the muon energy with and
without bursts, and the deposited energy (`energy_lost`) as a lower bound of the muon energy. Which tracks stop inside
the generation box (`data.truth.stopped`), and how does their light differ? Build a $dE/dx$ proxy (charge per metre
of track in the instrumented area) and show how well it follows $E$. Then on `data/t3_track_he.h5` (muons from
100 GeV to 100 TeV, `full` preset): the fraction of the detected light that comes from bursts against $E$, and the
$dE/dx$ proxy against $E$ on log axes. Over which range of $E$ is the energy of a through-going track measurable at
all, and with what resolution?

**Deliverables:** for T3.1 and T3.2 the quantities of all presets and both media on shared axes; for T3.3 the cleaning
efficiency against the slack; for T3.4 the bias and the resolution of both estimators; for T3.5 the proxy against $E$.

**Hints**
- `data.hits.source == 3` (or `data.hits.is_noise`) marks the noise hits, so you know the answer of the cleaning;
  `data.pulses.n_noise` counts the noise hits in a pulse. From T3.3 on, the `full` files carry noise (about 24 hits
  per event in water): clean the pulses or count the light alone, and say which.
- `angular_error` wraps the difference: 359° and 1° differ by 2°. LineFit is defined only for events with at least two
  pulses at different times; the others give NaN.
- The energy calibration: fit it on one data set and apply it to the others, so that you see how the bias changes with
  the preset and the medium.

[^noise]: Every module gets noise hits at the rate $\nu$ uniformly in the readout window of length $T$. What is the
    mean number per module, and per event of a detector with $M$ modules?

## T4 Event displays (about 2 hours)

**Goal:** draw single events so that the eye sees what a reconstruction sees.

**T4.1** The standard display: all modules, the hit ones with an area proportional to their charge and a colour for
their first time (with a colour bar), the true track or cascade on top, the event type, energy and $Q$ in the title.
Draw it yourself first, then compare with `event_display`.

```python
import matplotlib.pyplot as plt
from nutel.toy import load
from nutel.toy.plot import event_display

data = load("data/t4_events.h5")
fig, ax = plt.subplots(figsize=(7, 5))
event_display(ax, data, 0, linefit=True)
fig.savefig("T4.1_event0.png", dpi=150)
```

**T4.2** A gallery by type: a track, a starting track, a cascade, an atmospheric muon (from `data/t4_events.h5`) and a
noise-only readout (`data/t4_noise_only.h5`). What distinguishes them by eye?

**T4.3** Photon paths: the paths of the photons of one event in water and in ice (`data/t4_paths_water.h5`,
`data/t4_paths_ice.h5`), coloured by fate (detected direct, detected scattered, absorbed, escaped).

**T4.4** An animation of the Cherenkov front sweeping over the strings (`animate_event` or your own
`matplotlib.animation.FuncAnimation`).

```bash
nutelsim animate data/t4_paths_water.h5 -i 0 -o T4.4_front.gif
```

**T4.5** The same event in other detectors: the files `t4_same_*` hold one track with the same direction, anchor,
time and energy in the default line (`t4_same_default.h5`), a dense core, an irregular layout, ice and six sampled
detectors (`t4_same_core.h5`, `t4_same_irregular.h5`, `t4_same_ice.h5`, `t4_same_family.h5`). Show them side
by side with the same colour scale. Which differences come from the layout and which from the medium?

**Deliverables:** one display per question; for T4.4 a GIF or MP4.

**Hints**
- Use one colour scale for all panels you compare (`vmin`, `vmax`), and equal axes.
- `data.paths.of_event(event_id)` gives the paths of one event, `data.paths.polyline(k)` the points `(x, z, t, kind)`
  and `data.paths.fate` the fate of every path; `photon_paths` draws them.
- `gallery(data, indices)` draws several events at once (`panel_options` changes single panels, e.g.
  `{"data": noise_only, "truth": False}`); `draw_geometry` draws a detector alone.
- `event_display(ax, data, i, mask=causal_cleaning(data.pulses))` draws the removed pulses as crosses and fits and
  colours the kept ones; `noise=True` marks the noise pulses from the truth.
- An animation saved with the Pillow writer is a large GIF: keep the number of frames and the dpi modest.

## T5 Bonus: towards reconstruction (2 to 4 hours)

**Goal:** first steps towards the questions that a learned reconstruction will have to answer.

**T5.1** Fluctuations. `data/t5_same_event.h5` holds one track simulated 5000 times. Plot the spread of the features
of T2.4 and of the LineFit direction. Which part of the spread comes from the number of photons and the bursts? Compare
with `data/t5_same_event_fixed.h5` (fixed photon numbers, no bursts). This spread is a floor that no reconstruction
can beat.

**T5.2** Degeneracies. (a) Direction flip: `data/t5_flipped.h5` has the same line travelled the other way. Which
features tell the two apart, and how well does LineFit do it? (b) Track or cascade at few pulses: on
`data/t1_scatter.h5` (no noise) and on `data/t3_full_water.h5` (cleaned as in T3.3), how does the separation of
tracks and cascades by a simple feature (for example the duration or the extent of the hits) depend on the number of
pulses? (c) $t_0$ against the position along the track: show that the point
$\mathbf{a} + \Delta s\,\mathbf{d}$ with the time $t_0 + \Delta s / c$ describes the same track for any $\Delta s$
[^shift]. Why does the toy store the closest point to the origin as the anchor, and what does this mean for a
reconstruction that fits a point and a time?

**T5.3** Detector design. With the `t5_design_*` files: the LineFit resolution and the fraction of tracks with at least
5 pulses against the string spacing, the number of strings, a dense core and the aspect of the array (16 strings of 6
modules against 8 of 12). What would you build with 96 modules?

**T5.4** Detector effects. With the `t5_jitter*`, `t5_eff*` and `t5_noise*` files: the LineFit resolution and the
energy resolution from $Q$ against the time jitter, the efficiency and the noise rate. Which effect hurts which
estimator most?

**T5.5** Up against down. Atmospheric muons outnumber the muons from neutrinos by about $10^6$ to 1 and come only from
above. With `data/t5_atmospheric.h5` and the up-going tracks of `data/t3_track.h5`: with LineFit, what fraction of
atmospheric muons is reconstructed as up-going? How many would leak into an up-going sample for each true neutrino
event, and what resolution would you need?

**T5.6** A multi-detector data set. `data/t5_family.h5` holds 16 detectors sampled from the default family, with 2000
tracks each. Draw the detectors (`nutelsim family --n 16 --seed 56 -o T5.6_family.png`). Compute the same features for
every detector: which features depend mostly on the detector (number of pulses, extent, times) and which mostly on the
event (the direction of the time-ordered axis)? Plot the LineFit resolution per detector against its number of
modules and its string spacing. What would a model need to know about the detector to read all 16? This is the
first look at the question of stage C.

```python
from nutel.toy import event_features, load

data = load("data/t5_family.h5")
features = event_features(data)  # one row per event, with its geometry_index
for k, detector in enumerate(data.detectors):
    g = detector.geometry
    print(k, g.layout, g.n_strings, g.n_modules, f"{detector.medium.absorption_length:.0f} m")
```

**Deliverables:** one or two figures per question with a short answer.

**Hints**
- T5.1: compare the variance of $Q$ with its mean. Which source of fluctuations is gone with fixed photon numbers?
- T5.3 and T5.4: change one thing at a time. The files of one question share their seed: in T5.4 the tracks are the
  same; in T5.3 they share directions, energies and times, but the anchor points scale with the generation radius of
  each layout (compare effective widths, not event by event).
- In water every event has about 24 noise hits: clean the pulses (T3.3) before LineFit, and say how you cleaned.
- T5.5: the angular error alone does not answer the question; count the atmospheric muons whose reconstructed zenith
  is above 90°.
- T5.6: `data.select(data.truth.geometry_index == k)` keeps the events of detector `k`; `data.detector(i)` is the
  detector of event `i`.

[^shift]: Write the position of the muon at the time $t$ for both parameter sets.
