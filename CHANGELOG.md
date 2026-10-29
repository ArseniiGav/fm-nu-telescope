# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-29

Stage A: the toy "Light in the Deep".

### Added

- `nutel.toy`: a two-dimensional neutrino telescope with tracks, starting events, cascades and atmospheric muons,
  Cherenkov emission with stochastic losses, photon transport in torch (absorption, scattering, opaque modules) on
  CPU or CUDA, readout to hits and pulses with jitter, efficiency and noise, and the ladder presets `straight`,
  `absorb`, `scatter`, `full` for water and ice.
- Detector layouts (`line`, `core`, `irregular`, `ragged`, `custom`) and sampled detector families for
  multi-detector data sets.
- HDF5 data format with configuration and provenance, `merge` and noise augmentation; the `nutelsim` command line.
- Analysis helpers: event features, LineFit and charge-based energy, event displays, photon paths and animations.
- Documentation, student tasks T0-T5 and tests.

[0.1.0]: https://github.com/ArseniiGav/fm-nu-telescope/releases/tag/v0.1.0
