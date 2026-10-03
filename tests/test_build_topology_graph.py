"""Checks for the topology-graph prior definition.

The full build script reads ~1 GB of manifest and 240 flight CSVs, so these tests
guard the cheap-to-get-wrong part: that every physical edge names a real channel of
feature contract v1, and that the two tiers stay disjoint and symmetric.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

from build_topology_graph import (  # noqa: E402
    PHYSICAL_EDGES_TIER1,
    PHYSICAL_EDGES_TIER2,
    build_physical_adjacency,
)

# Canonical channel order of feature contract v1 (26 channels).
CONTRACT_CHANNELS = [
    "actuator_ctrl_1",
    "actuator_ctrl_2",
    "actuator_ctrl_3",
    "actuator_ctrl_4",
    "actuator_out_1",
    "actuator_out_2",
    "actuator_out_3",
    "actuator_out_4",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "baro_alt_relative",
    "att_q_w",
    "att_q_x",
    "att_q_y",
    "att_q_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "acc_norm",
]


def _all_edges() -> list[tuple[str, str]]:
    edges = []
    for groups in (PHYSICAL_EDGES_TIER1, PHYSICAL_EDGES_TIER2):
        for group_edges in groups.values():
            edges.extend(group_edges)
    return edges


def test_every_prior_edge_names_contract_channels():
    known = set(CONTRACT_CHANNELS)
    unknown = sorted({name for edge in _all_edges() for name in edge if name not in known})
    assert unknown == [], f"prior references unknown channels: {unknown}"


def test_prior_edges_do_not_self_loop():
    self_loops = [edge for edge in _all_edges() if edge[0] == edge[1]]
    assert self_loops == []


def test_tier1_and_tier2_are_disjoint():
    tier1 = {frozenset(edge) for edge in PHYSICAL_EDGES_TIER1.values() for edge in [edge]}
    tier2 = {frozenset(edge) for edge in PHYSICAL_EDGES_TIER2.values() for edge in [edge]}
    overlap = tier1 & tier2
    assert overlap == set(), f"edge classified in both tiers: {overlap}"


def test_tier1_encodes_the_rate_control_law():
    """The strongest measured structure on normal flight must stay in tier 1."""
    expected = {
        frozenset({"actuator_ctrl_1", "gyro_x"}),
        frozenset({"actuator_ctrl_2", "gyro_y"}),
        frozenset({"actuator_ctrl_3", "gyro_z"}),
    }
    tier1 = {frozenset(edge) for edge in PHYSICAL_EDGES_TIER1["rate_command_to_rate"]}
    assert expected <= tier1


def test_tier1_encodes_the_mixing_matrix():
    """Every motor output must connect to all four control commands."""
    pairs = {frozenset(edge) for edge in PHYSICAL_EDGES_TIER1["thrust_to_motors"]}
    pairs |= {
        frozenset(edge)
        for edge in PHYSICAL_EDGES_TIER1["differential_command_to_motors"]
    }
    for i in range(1, 5):
        for j in range(1, 5):
            assert frozenset({f"actuator_ctrl_{i}", f"actuator_out_{j}"}) in pairs


def test_adjacency_is_symmetric_with_empty_diagonal():
    adjacency, counts = build_physical_adjacency(CONTRACT_CHANNELS, PHYSICAL_EDGES_TIER1)
    assert adjacency.shape == (26, 26)
    assert np.array_equal(adjacency, adjacency.T)
    assert float(np.trace(adjacency)) == 0.0
    assert set(np.unique(adjacency)) <= {0.0, 1.0}
    assert counts["rate_command_to_rate"] == 3


def test_adjacency_rejects_unknown_channel():
    with pytest.raises(KeyError):
        build_physical_adjacency(
            CONTRACT_CHANNELS, {"bad": [("actuator_ctrl_1", "not_a_channel")]}
        )
