# NuBench in one page

NuBench is an open benchmark for deep-learning event reconstruction in neutrino telescopes: seven large simulated
data sets of muon-neutrino interactions in six detector geometries, with the true event parameters, the detector
response as pulses, and the predictions of four reconstruction models. Everything on this page is taken from the
paper (R. F. Ørsøe et al., [arXiv:2511.13111](https://arxiv.org/abs/2511.13111), version 3 of 26 February 2026) and
from its repository [graphnet-team/NuBench](https://github.com/graphnet-team/NuBench). Stage D of this project
uses it; the toy of this repository is built so that its concepts map onto it (task T0.4 writes that map).

## The seven data sets

About 129.7 million $\nu_\mu$ charged-current (CC) and neutral-current (NC) interactions between 10 GeV and 100 TeV.
Six geometries are inspired by, but not identical to, existing or proposed telescopes; each is simulated in water,
and Hexagon also in ice.

| data set         | inspired by | medium | strings / modules | horizontal spacing        | vertical spacing | events (M) | CC / NC (%) | energy (GeV)  |
|------------------|-------------|--------|-------------------|---------------------------|------------------|------------|-------------|---------------|
| Triangle         | P-ONE       | water  | 3 / 60            | ~100 m                    | ~52 m            | 23.1       | 35 / 65     | 10 - 10^5     |
| Cluster          | Baikal-GVD  | water  | 8 / 288           | ~52 m                     | ~15 m            | 22.9       | 49 / 51     | 10 - 10^5     |
| Flower S         | KM3NeT/ORCA | water  | 150 / 3300        | ~12 m                     | ~9 m             | 20.5       | 40 / 60     | 10 - 10^3     |
| Flower L         | KM3NeT/ARCA | water  | 115 / 2070        | ~72 m                     | ~30 m            | 24.0       | 35 / 65     | 10 - 10^5     |
| Flower XL        | TRIDENT     | water  | 1211 / 24220      | ~90 m                     | ~30 m            | 10.1       | 88 / 12     | 10 - 10^5     |
| Hexagon          | IceCube     | water  | 86 / 5160         | 125 m, plus a denser infill | varies         | 20.5       | 48 / 52     | 10 - 10^5     |
| Hexagon Ice LE   | IceCube     | ice    | 86 / 5160         | as Hexagon                | as Hexagon       | 8.6        | 57 / 43     | 10 - 10^3     |

The three Flower geometries have their strings on a sunflower pattern, which avoids long straight corridors without
modules. Hexagon is a near-hexagonal array of 78 strings with eight more strings at a smaller spacing inside it (the
analogue of IceCube's DeepCore). Detector response per data set (paper, Table 10):

| data set       | PMTs per module | transit-time spread (ns) | noise per PMT (kHz) | optical efficiency (%) |
|----------------|-----------------|--------------------------|---------------------|------------------------|
| Triangle       | 24              | 1.5                      | 7.5                 | 20                     |
| Cluster        | 1               | 3.4                      | 60                  | 17.5                   |
| Flower S       | 31              | 4.5                      | 7.5                 | 20                     |
| Flower L       | 31              | 4.5                      | 7.5                 | 20                     |
| Flower XL      | 31              | 3.7                      | 7.5                 | 20                     |
| Hexagon        | 1               | 3.2                      | 7.5                 | 15                     |
| Hexagon Ice LE | 1               | 3.2                      | 0.3                 | 15                     |

## How the events were made

1. **Physics simulation** with [PROMETHEUS](https://arxiv.org/abs/2304.14526) (Lazar et al., Comput. Phys. Commun.
   304, 109298, 2024): LeptonInjector injects the neutrino interaction directly into the simulation volume around the
   detector, PROPOSAL propagates the muons, and the light is produced and propagated with fennel and hyperion in water
   and with PPC in ice. The output is the Cherenkov photons that reach each module, without their arrival direction.
   No noise and no atmospheric muons are simulated.
2. **Simplified detector response.** A module is a sphere of radius 30 cm with a quantum efficiency of 20 % and acts
   as one effective PMT: uniform angular acceptance in water, the measured IceCube acceptance (less sensitive to
   light from above) in ice. The photon times are shifted into a trigger window of at least 5 µs; the efficiency is
   reduced to the value of the table by subsampling; noise photons are added uniformly in the window on randomly
   chosen modules; at each module, photons within a merging window equal to the transit-time spread are merged into
   one pulse, whose time is the mean time of its photons and whose charge is their number; finally the pulse times
   and charges are smeared by Gaussians of 1 ns and 0.25 p.e. Events with fewer than four signal pulses or more than
   a million are removed.

## Two data levels: training and test

Each data set is split into a training and a test sample; the test sample has about 3 million events.

| partition | directory in the Parquet archive | processing applied                                                        |
|-----------|----------------------------------|---------------------------------------------------------------------------|
| training  | `merged_photons/`                | efficiency and merging only; the rest is to be applied during training     |
| test      | `pulses_no_noise/`               | the full procedure above, one fixed realisation; noise pulses removed      |

The augmentations to apply to every training event, at every pass (paper, appendix B.1):

- time and charge smearing per pulse: $t \to |t + \epsilon_t|$ with $\epsilon_t \sim \mathcal{N}(0, 1\,\mathrm{ns})$,
  $q \to |q + \epsilon_q|$ with $\epsilon_q \sim \mathcal{N}(0, 0.25\,\mathrm{p.e.})$;
- noise pulses (before the smearing): their number is Poisson with mean
  $N_\mathrm{total} = R_\mathrm{OM}\, t_\mathrm{window}\, N_\mathrm{OMs}$, where $R_\mathrm{OM} = N_\mathrm{PMT}
  R_\mathrm{PMT}$ (table above) and $t_\mathrm{window} = 5$ µs; each gets a time uniform in $[0, t_\mathrm{max}]$
  ($t_\mathrm{max}$ = 5 µs, or the latest signal pulse if later), a random module and a charge of 1 p.e.
  For Triangle, $24 \times 7.5\,\mathrm{kHz} \times 5\,\mathrm{µs} \times 60 = 54$ noise pulses per event on average.

The archive also holds `mc_truth/` (the event-level table) and `selections/` (which chunks form the training and the
test partitions).

## Field names

Pulse table (paper Table 2; column names as in the Parquet files of Triangle), one row per pulse:

| column                                       | content                                                      |
|----------------------------------------------|--------------------------------------------------------------|
| `sensor_pos_x`, `sensor_pos_y`, `sensor_pos_z` | position of the module (m)                                 |
| `t`                                          | pulse time (ns)                                              |
| `charge`                                     | pulse charge (p.e.)                                          |
| `string_id`                                  | string number                                                |
| `is_signal`                                  | fraction of signal (non-noise) photons in the merging window |
| `event_no`                                   | event number, the key to the truth table                     |

Truth table (paper Table 3), one row per event: `initial_state_energy` (neutrino energy, GeV),
`initial_state_x`, `initial_state_y`, `initial_state_z` (interaction vertex, m), `initial_state_zenith`,
`initial_state_azimuth` (direction of the neutrino), `initial_state_type` (PDG code), `interaction` (1 = CC,
2 = NC), `bjorken_x`, `bjorken_y`, `visible_inelasticity` (visible hadronic energy over total visible energy),
`muon_zenith`, `muon_azimuth` (outgoing muon of CC events), `event_no`. The paper states that "the zenith and
azimuthal angles describe the direction of travel of the incident neutrino", and the PROMETHEUS paper that a zenith of
0 "corresponds to up-going neutrinos". The paper does not state the injected energy spectra or the inelasticity
distribution.

## Tasks, metrics and models

| task                          | metric reported in the paper                                                                        |
|-------------------------------|-----------------------------------------------------------------------------------------------------|
| energy                        | bias = median and resolution = width of $(E_\mathrm{true} - E_\mathrm{reco})/E_\mathrm{true}$ in % per energy range; band plots of $E_\mathrm{reco}$ vs $E_\mathrm{true}$ (16th-84th percentiles) |
| direction                     | opening angle $\psi$ between true and reconstructed direction: median, and the fractions with $\psi \le 1°$ and $\le 5°$ |
| track / cascade (CC / NC)     | ROC curves and the area under them (AUC), per energy range                                          |
| interaction vertex            | median Euclidean distance $D_{xyz}$ and its components $D_{xy}$ (horizontal) and $D_z$ (vertical)   |
| inelasticity                  | median of $\lvert y_\mathrm{reco} - y_\mathrm{vis} \rvert$ and its 16th-84th percentile width       |

All metrics are given separately for CC and NC events or for energy ranges, on the test partitions, with bootstrap
uncertainties. The four models are implemented in [GraphNeT](https://github.com/graphnet-team/graphnet):

| model       | type                        | parameters | used for                                                       |
|-------------|-----------------------------|------------|----------------------------------------------------------------|
| ParticleNeT | graph neural network        | 0.3 M      | all tasks, all data sets                                       |
| DynEdge     | graph neural network        | 1.3 M      | all tasks, all data sets                                       |
| GRIT        | graph-transformer hybrid    | 8.8 M      | most data sets and tasks (not Flower S)                        |
| DeepIce     | transformer                 | 114 M      | direction                                                      |

DynEdge (IceCube, JINST 17, P11003, 2022) is the dynamic edge convolution of GraphNeT: the nodes are the modules with
pulses, each carrying its position and percentiles (0, 10, 50, 90, 100) of its pulse times and charges plus its number
of pulses, and the edges connect each node to its eight nearest neighbours. Losses: LogCosh of
$\log_{10} E_\mathrm{reco} - \log_{10} E_\mathrm{true}$ for the energy, a von Mises-Fisher likelihood for the
direction, binary cross-entropy for track / cascade, the Euclidean distance (DynEdge) for the vertex. Inputs are
scaled as $x, y \to x/100$, $z \to z/1000$, $t \to t/10^5$, $q \to \log_{10}(q + 1)$.

## Download

The data live on the Electronic Research Data Archive of the University of Copenhagen (ERDA). The repository README
links, for every data set, an SQLite and a Parquet archive (the same content, readable with the dataset classes of
GraphNeT), the model predictions on the test partition, and the trained models. Download with
`wget -P <directory> <link>`; the archives end in `.tar.gz`, but the Parquet and prediction archives are plain tar
files (`tar -xf`), the SQLite and model archives compressed ones (`tar -xzf`).

| data set       | Parquet archive (GB) |
|----------------|----------------------|
| Hexagon Ice LE | 5.1                  |
| Triangle       | 29.0                 |
| Flower XL      | 62.6                 |
| Flower L       | 68.9                 |
| Hexagon        | 95.6                 |
| Cluster        | 110.3                |
| Flower S       | 191.7                |

Hexagon Ice LE and Triangle are the small ones to start with. Triangle unpacks into `merged_photons/` (805 files,
20.3 GB), `pulses_no_noise/` (121 files, 6.3 GB), `mc_truth/` (2.4 GB) and `selections/`.

### One chunk without the full archive

The Parquet archives are uncompressed tar files, and the server answers HTTP range requests, so a single file of an
archive can be fetched alone. Test chunk 900 of Triangle (25 000 events of the test partition, the noise-free
`pulses_no_noise` level) and its truth table take 55 MB:

```bash
mkdir -p data/nubench
curl -L -r 1641735168-1694479014 -o data/nubench/pulses_no_noise_900.parquet https://sid.erda.dk/share_redirect/ediHXAsygn
curl -L -r 27217495040-27220133134 -o data/nubench/mc_truth_900.parquet https://sid.erda.dk/share_redirect/ediHXAsygn
python -m pip install pyarrow
```

The byte ranges are the positions of the two files in the Triangle Parquet archive (52 743 847 and 2 638 095 bytes; a
file of another size is a failed download). Reading them needs pandas and pyarrow:

```python
import pandas as pd

pulses = pd.read_parquet("data/nubench/pulses_no_noise_900.parquet")  # one row per pulse, index event_no
truth = pd.read_parquet("data/nubench/mc_truth_900.parquet")  # one row per event, index event_no
event = pulses.loc[truth.index[0]]  # the pulses of the first event: sensor_pos_x/y/z, t, charge, ...
```

## Citation

R. F. Ørsøe, S. Meighen-Berger, J. Lazar, J. Prado, I. Mozún-Mateo, A. Rosted, P. Weigel and A. Llorente Anaya,
*NuBench: An Open Benchmark for Deep Learning-Based Event Reconstruction in Neutrino Telescopes*,
[arXiv:2511.13111](https://arxiv.org/abs/2511.13111) [hep-ex] (2025). The repository README also asks to cite
PROMETHEUS ([arXiv:2304.14526](https://arxiv.org/abs/2304.14526)) and GraphNeT
([arXiv:2501.03817](https://arxiv.org/abs/2501.03817)).
