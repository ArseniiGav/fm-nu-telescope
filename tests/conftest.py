"""Shared fixtures of the test suite (all simulations run on the CPU)."""

from __future__ import annotations

import pytest

from nutel.toy import SimData, generate, preset


@pytest.fixture(scope="session")
def full_small() -> SimData:
    """300 events of the full preset in water with hits, bursts and the photon paths of the first 3 events."""
    return generate(300, config=preset("full"), seed=11, device="cpu", save_paths=3)


@pytest.fixture(scope="session")
def multi_small() -> SimData:
    """4 detectors of the default family x 40 events (full preset)."""
    return generate(n_geometries=4, events_per_geometry=40, family="default", seed=12, geometry_seed=3, device="cpu")
