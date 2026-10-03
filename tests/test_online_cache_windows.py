"""Tests for the ground-window filter and the live-session holdout split.

Both are small pure helpers, but getting them wrong silently changes what the model is
told "normal flight" looks like, so they get direct checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "scripts" / "data", REPO_ROOT / "packages" / "uav_fault"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_online_cache_from_windows import (  # noqa: E402
    airborne_mask,
    maneuver_of,
    split_holdout,
)

COLUMNS = ["baro_alt_relative", "gyro_x"]


def test_maneuver_of_reads_the_profile_from_the_filename():
    assert maneuver_of("nf_rflysim_hover_5m_01.csv") == "hover"
    assert maneuver_of("nf_rflysim_box_10m_10.csv") == "box"
    assert maneuver_of("nf_rflysim_mixed_5_15m_03.csv") == "mixed"
    assert maneuver_of("nf_rflysim_vertical_3_10m_07.csv") == "vertical"
    assert maneuver_of("unexpected.csv") == "unknown"


def test_airborne_mask_excludes_ground_by_altitude():
    # 50.0 m is the launch elevation in the live recordings; flying adds ~5 m.
    altitude = np.array([50.0, 50.0, 50.4, 50.6, 55.0, 55.0, 50.2, 50.0])
    matrix = np.column_stack([altitude, np.zeros_like(altitude)])
    mask = airborne_mask(matrix, COLUMNS, margin_m=0.5)
    assert mask.tolist() == [False, False, False, True, True, True, False, False]


def test_airborne_mask_is_permissive_without_an_altitude_channel():
    """No altitude column means we cannot tell, so nothing is filtered."""
    matrix = np.zeros((10, 1))
    assert airborne_mask(matrix, ["gyro_x"]).all()


def test_airborne_mask_handles_all_nan_altitude():
    matrix = np.column_stack([np.full(5, np.nan), np.zeros(5)])
    assert airborne_mask(matrix, COLUMNS).all()


def test_split_holdout_holds_back_each_maneuver():
    files = [f"nf_rflysim_hover_5m_{i:02d}.csv" for i in range(1, 11)]
    files += [f"nf_rflysim_box_10m_{i:02d}.csv" for i in range(1, 11)]
    train, holdout = split_holdout(files, per_maneuver=2)
    assert len(train) == 16
    assert len(holdout) == 4
    assert {maneuver_of(n) for n in holdout} == {"hover", "box"}
    # every held-out session has exactly one of each maneuver
    assert sorted(holdout) == [
        "nf_rflysim_box_10m_01.csv",
        "nf_rflysim_box_10m_02.csv",
        "nf_rflysim_hover_5m_01.csv",
        "nf_rflysim_hover_5m_02.csv",
    ]
    assert set(train) & set(holdout) == set()


def test_split_holdout_zero_keeps_everything_in_train():
    files = [f"nf_rflysim_hover_5m_{i:02d}.csv" for i in range(1, 4)]
    train, holdout = split_holdout(files, per_maneuver=0)
    assert holdout == []
    assert sorted(train) == sorted(files)


def test_split_holdout_does_not_overdraw_a_small_group():
    files = ["nf_rflysim_hover_5m_01.csv"]
    train, holdout = split_holdout(files, per_maneuver=3)
    assert holdout == ["nf_rflysim_hover_5m_01.csv"]
    assert train == []


def test_split_holdout_is_deterministic():
    files = [f"nf_rflysim_vertical_3_10m_{i:02d}.csv" for i in range(1, 11)]
    first = split_holdout(files, per_maneuver=2)
    second = split_holdout(list(reversed(files)), per_maneuver=2)
    assert first == second


@pytest.mark.parametrize("per_maneuver", [1, 2, 3])
def test_holdout_disjoint_from_train_for_real_configuration(per_maneuver):
    """The 40 real sessions: 4 maneuvers x 10 files."""
    files = []
    for maneuver, tag in (("hover", "5m"), ("box", "10m"),
                          ("mixed", "5_15m"), ("vertical", "3_10m")):
        files += [f"nf_rflysim_{maneuver}_{tag}_{i:02d}.csv" for i in range(1, 11)]
    train, holdout = split_holdout(files, per_maneuver=per_maneuver)
    assert len(holdout) == 4 * per_maneuver
    assert set(train) & set(holdout) == set()
    assert len(train) + len(holdout) == 40
