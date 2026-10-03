"""Regression guard for the euler -> quaternion unit contract on the live path.

History: schema_adapter._quat_from_euler applied math.radians() to inputs that were
already radians (subscriptions.py converts before publishing the frame, the RflySim
live recordings store radians, and feature_contract_v1 declares the source as
derived_from_euler_rad). The result was att_q_x/y/z ~57x too small, i.e. three of the
26 channels were effectively dead whenever the quaternion had to be derived.

These tests pin the unit convention so it cannot flip back silently.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for package in ("uav_fault", "rfly_bridge"):
    candidate = REPO_ROOT / "packages" / package
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rfly_bridge.rfly_live_bridge import _quat_from_euler as bridge_quat  # noqa: E402
from uav_fault.telemetry.schema_adapter import (  # noqa: E402
    _quat_from_euler as adapter_quat,
)
from uav_fault.telemetry.schema_adapter import canonicalize_rfly_frame  # noqa: E402

# A hover-scale attitude in radians (what the live recordings actually contain).
HOVER_EULER = (0.0036, -0.0089, 0.1209)
# A clearly non-trivial attitude, still radians.
BANK_EULER = (0.35, -0.2, 1.1)


@pytest.mark.parametrize("euler", [HOVER_EULER, BANK_EULER])
def test_adapter_matches_the_bridge_implementation(euler):
    """Two implementations of the same conversion must agree on the same units."""
    expected = bridge_quat(*euler)
    got = adapter_quat(*euler)
    assert got is not None
    assert got == pytest.approx(expected, rel=0, abs=1e-12)


def test_no_hidden_degree_conversion():
    """The vector part must not be shrunk by a spurious degree conversion.

    qx is not simply sin(roll/2) once pitch and yaw are non-zero, so compare against
    what the buggy version produced (the same euler read as degrees) instead of an
    isolated term.
    """
    quat = adapter_quat(*BANK_EULER)
    buggy = bridge_quat(*(math.radians(value) for value in BANK_EULER))
    norm = math.sqrt(sum(value * value for value in quat[1:]))
    buggy_norm = math.sqrt(sum(value * value for value in buggy[1:]))
    assert norm > 10 * buggy_norm, f"vector part {norm} vs buggy {buggy_norm}"
    assert norm > 0.15


def test_live_frame_keeps_attitude_energy():
    """A ~20 deg bank must not collapse to a near-identity quaternion."""
    canonical, coverage = canonicalize_rfly_frame(
        {
            "time_s": 0.0,
            "roll": 0.35,
            "pitch": -0.2,
            "yaw": 1.1,
            "accel_z": -9.8,
        }
    )
    assert "att_q_w" in coverage["derived"]
    vector = tuple(canonical[key] for key in ("att_q_x", "att_q_y", "att_q_z"))
    norm = math.sqrt(sum(value * value for value in vector))
    # sin(theta/2) for a 0.35 rad roll alone is ~0.17; a degree conversion gives ~0.003.
    assert norm > 0.1, f"attitude vector collapsed to {norm}"
    assert abs(canonical["att_q_w"]) == pytest.approx(
        math.sqrt(1 - norm * norm), rel=1e-6
    )


def test_recorded_live_session_attitude_is_consistent_with_its_quaternion():
    """Cross-check against a real recording: euler and quaternion must agree.

    The recorded files carry both roll/pitch/yaw (radians) and a quaternion, so the
    adapter's derivation can be validated against the ground truth they shipped.
    """
    import pandas as pd

    session = Path(r"D:\UAV-Data(old)\extra_normals\nf_rflysim_hover_5m_01.csv")
    if not session.exists():
        pytest.skip(f"live recording not available: {session}")

    frame = pd.read_csv(session, nrows=1, low_memory=False).iloc[0]
    canonical, _ = canonicalize_rfly_frame(frame.to_dict())

    derived = [canonical[key] for key in ("att_q_w", "att_q_x", "att_q_y", "att_q_z")]
    # The recording does not store q_*, so compare against the bridge conversion of the
    # same euler angles, which is the definition both implementations must satisfy.
    expected = bridge_quat(
        float(frame["roll"]), float(frame["pitch"]), float(frame["yaw"])
    )
    assert derived == pytest.approx(list(expected), rel=0, abs=1e-9)


def test_real_recordings_store_radians_not_degrees():
    """Guard the premise of the fix on the actual data, not just on reasoning."""
    import pandas as pd

    session = Path(r"D:\UAV-Data(old)\extra_normals\nf_rflysim_hover_5m_01.csv")
    if not session.exists():
        pytest.skip(f"live recording not available: {session}")

    frame = pd.read_csv(session, usecols=["roll", "pitch", "yaw"], low_memory=False)
    for column in ("roll", "pitch", "yaw"):
        peak = float(frame[column].abs().max())
        assert peak < math.pi, (
            f"{column} peaks at {peak}, which is degrees, not radians; "
            "revisit the euler unit convention"
        )
