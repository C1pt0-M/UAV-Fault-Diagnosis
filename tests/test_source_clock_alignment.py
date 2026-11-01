"""Clock-reference checks without dataset or model access."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
from check_source_clock_alignment import bridge_errors


def test_simulator_clock_has_its_own_origin_not_px4_seconds():
    px4_us = np.array([35423780, 35443780, 35463780])
    simulator_s = np.array([22.07878, 22.09878, 22.11878])
    check = bridge_errors(px4_us, simulator_s, 34635, 21.29)
    assert check['max_bridge_error_s'] < 1e-12
    assert check['max_step_difference_s'] < 1e-12
    assert check['observed_sim_minus_px4_s'] == pytest.approx(-13.345)


def test_wrong_unit_or_origin_is_visible_not_auto_corrected():
    px4_us = np.array([10000000, 10020000, 10040000])
    correct = np.array([5, 5.02, 5.04])
    assert bridge_errors(px4_us, correct + 1, 10000, 5)['max_bridge_error_s'] == pytest.approx(1)
    assert bridge_errors(px4_us, correct * 1000, 10000, 5)['max_bridge_error_s'] > 1000


@pytest.mark.parametrize('px4,sim', [([], []), ([1], [1]), ([1, 1], [1, 2]),
                                      ([2, 1], [1, 2]), ([1, 2], [2, 1]),
                                      ([1, np.nan], [1, 2]), ([1, 2], [1])])
def test_missing_or_unordered_clock_is_rejected(px4, sim):
    with pytest.raises(ValueError):
        bridge_errors(px4, sim, 0, 0)
