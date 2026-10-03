from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "data" / "compute_window_norm_stats.py"
)
SPEC = spec_from_file_location("compute_window_norm_stats", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_aggregate_case_windows_matches_materialized_windows() -> None:
    values = np.array(
        [
            [1.0, 10.0, -2.0],
            [2.0, 12.0, -1.0],
            [4.0, 15.0, 0.0],
            [8.0, 19.0, 2.0],
            [16.0, 24.0, 5.0],
            [32.0, 30.0, 9.0],
        ]
    )
    window_size = 3
    stride = 2
    relative_index = 1

    n_windows, observations, sums, sums_sq = MODULE.aggregate_case_windows(
        values,
        window_size=window_size,
        stride=stride,
        relative_feature_index=relative_index,
    )

    materialized = []
    for start in range(0, len(values) - window_size + 1, stride):
        window = values[start : start + window_size].copy()
        window[:, relative_index] -= window[0, relative_index]
        materialized.append(window)
    stacked = np.concatenate(materialized, axis=0)

    assert n_windows == 2
    assert observations == len(stacked)
    np.testing.assert_allclose(sums, stacked.sum(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        sums_sq, np.square(stacked).sum(axis=0), rtol=0, atol=1e-12
    )


def test_short_case_contributes_no_windows() -> None:
    values = np.ones((2, 3), dtype=float)
    n_windows, observations, sums, sums_sq = MODULE.aggregate_case_windows(
        values,
        window_size=3,
        stride=1,
        relative_feature_index=1,
    )

    assert n_windows == 0
    assert observations == 0
    np.testing.assert_array_equal(sums, np.zeros(3))
    np.testing.assert_array_equal(sums_sq, np.zeros(3))
