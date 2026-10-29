"""Light in the Deep: a two-dimensional neutrino-telescope toy.

A vertical slice of a neutrino telescope: strings of optical modules in water or ice record the Cherenkov light of
through-going and starting muon tracks, cascades and atmospheric muons. Photons are emitted and tracked one by one
through absorption and scattering; the modules record hits and pulses. A ladder of presets switches the physics on
step by step (``straight``, ``absorb``, ``scatter``, ``full``); detectors can be sampled from a family for
multi-detector data sets.

Quick start::

    from nutel.toy import generate, preset, save, load, event_features, linefit

    data = generate(10_000, config=preset("full", medium="water"), seed=1)
    data.truth.energy, data.pulses.t, data.geometry(0)
    save(data, "data/toy_full_water_n10000_seed1.h5")

    multi = generate(n_geometries=8, events_per_geometry=1000, family="default", seed=2)

    from nutel.toy import plot  # matplotlib displays (imported on demand)

The same is available on the command line as ``nutelsim generate`` (``nutelsim --help``).

Exported names (``__all__``, sorted):

* simulation: :func:`generate`, :func:`simulate`, :func:`sample_prior`, :func:`resolve_device`, :func:`world_box`,
  :data:`DEFAULT_PHOTONS_PER_CHUNK`, :data:`LEVELS`;
* configuration: :class:`SimConfig` and its parts :class:`Layout`, :class:`Medium`, :class:`Readout`,
  :class:`Physics`, :class:`Emission`, :class:`Prior`; :func:`preset`, :data:`PRESETS`, :func:`describe_presets`,
  :func:`matching_preset`, :func:`medium`, :data:`MEDIA`, :func:`medium_name`, :func:`config_fields`; constants
  :data:`C_VACUUM`, :data:`EVENT_TYPES`, :data:`SOURCES`, :data:`LAYOUTS`;
* detectors: :class:`Geometry`, :func:`make_geometry` (the layouts of :data:`LAYOUTS`), :class:`Detector`,
  :class:`DetectorFamily`, :func:`family`, :data:`FAMILIES`, :func:`sample_detector`, :func:`sample_detectors`;
* data: :class:`SimData` and its tables :class:`Truth`, :class:`PhotonCounts`, :class:`Hits`, :class:`Pulses`,
  :class:`Bursts`, :class:`Paths`, :class:`Event`, :class:`PointType`, :data:`POINT_TYPES`; files: :func:`save`,
  :func:`load`, :func:`merge`, :func:`describe`, :data:`FORMAT_VERSION`; augmentation: :func:`add_noise`;
* directions: :func:`direction_from_angle`, :func:`angle_from_zenith_azimuth`, :func:`zenith_azimuth`;
* analysis: :func:`event_features`, :func:`image`, :data:`FEATURE_NAMES`, :data:`IMAGE_CHANNELS`;
  :func:`linefit`, :func:`linefit_events`, :class:`LineFit`, :func:`expected_arrival_time`,
  :func:`time_residuals`, :func:`emission_coordinate`, :func:`direct_light_possible`, :func:`shadowed`,
  :func:`angular_error`, :func:`causal_cleaning`, :func:`energy_from_charge`, :func:`fit_charge_calibration`.

``plot`` is a submodule (``from nutel.toy import plot``); it needs matplotlib, the rest of the package does not.
"""

from .config import (
    C_VACUUM,
    EVENT_TYPES,
    LAYOUTS,
    MEDIA,
    PRESETS,
    SOURCES,
    Emission,
    Layout,
    Medium,
    Physics,
    Prior,
    Readout,
    SimConfig,
    config_fields,
    describe_presets,
    matching_preset,
    medium,
    medium_name,
    preset,
)
from .data import (
    POINT_TYPES,
    Bursts,
    Detector,
    Event,
    Hits,
    Paths,
    PhotonCounts,
    PointType,
    Pulses,
    SimData,
    Truth,
)
from .events import angle_from_zenith_azimuth, direction_from_angle, sample_prior, zenith_azimuth
from .family import FAMILIES, DetectorFamily, family, sample_detector, sample_detectors
from .features import FEATURE_NAMES, IMAGE_CHANNELS, event_features, image
from .geometry import Geometry, make_geometry
from .io import FORMAT_VERSION, describe, load, merge, save
from .readout import add_noise
from .reco import (
    LineFit,
    angular_error,
    causal_cleaning,
    direct_light_possible,
    emission_coordinate,
    energy_from_charge,
    expected_arrival_time,
    fit_charge_calibration,
    linefit,
    linefit_events,
    shadowed,
    time_residuals,
)
from .simulate import DEFAULT_PHOTONS_PER_CHUNK, LEVELS, generate, resolve_device, simulate, world_box

__all__ = [
    "C_VACUUM",
    "DEFAULT_PHOTONS_PER_CHUNK",
    "EVENT_TYPES",
    "FAMILIES",
    "FEATURE_NAMES",
    "FORMAT_VERSION",
    "IMAGE_CHANNELS",
    "LAYOUTS",
    "LEVELS",
    "MEDIA",
    "POINT_TYPES",
    "PRESETS",
    "SOURCES",
    "Bursts",
    "Detector",
    "DetectorFamily",
    "Emission",
    "Event",
    "Geometry",
    "Hits",
    "Layout",
    "LineFit",
    "Medium",
    "Paths",
    "PhotonCounts",
    "Physics",
    "PointType",
    "Prior",
    "Pulses",
    "Readout",
    "SimConfig",
    "SimData",
    "Truth",
    "add_noise",
    "angle_from_zenith_azimuth",
    "angular_error",
    "causal_cleaning",
    "config_fields",
    "describe",
    "describe_presets",
    "direct_light_possible",
    "direction_from_angle",
    "emission_coordinate",
    "energy_from_charge",
    "event_features",
    "expected_arrival_time",
    "family",
    "fit_charge_calibration",
    "generate",
    "image",
    "linefit",
    "linefit_events",
    "load",
    "make_geometry",
    "matching_preset",
    "medium",
    "medium_name",
    "merge",
    "preset",
    "resolve_device",
    "sample_detector",
    "sample_detectors",
    "sample_prior",
    "save",
    "shadowed",
    "simulate",
    "time_residuals",
    "world_box",
    "zenith_azimuth",
]
