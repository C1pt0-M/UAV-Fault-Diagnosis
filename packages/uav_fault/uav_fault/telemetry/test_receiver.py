from __future__ import annotations

from uav_fault.telemetry.receiver import _coerce_seq


def test_coerce_seq_accepts_numeric_string() -> None:
    assert _coerce_seq("42", default=0) == 42


def test_coerce_seq_falls_back_on_invalid_value() -> None:
    assert _coerce_seq("not-a-number", default=7) == 7
