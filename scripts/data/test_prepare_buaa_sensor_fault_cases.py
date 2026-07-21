from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_buaa_sensor_fault_cases.py")
SPEC = importlib.util.spec_from_file_location("prepare_buaa_sensor_fault_cases", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sensor_cases_cover_four_sensor_types() -> None:
    cases = MODULE.build_cases()
    assert len(cases) == 12
    assert [case["CaseID"] for case in cases] == list(range(4521, 4533))
    assert {case["_metadata"]["sensor"] for case in cases} == {
        "accelerometer",
        "gyroscope",
        "magnetometer",
        "barometer",
    }


def test_accelerometer_parameters_are_interleaved_bias_and_noise() -> None:
    cases = MODULE.build_cases()
    assert cases[0]["_metadata"]["fault_parameters"] == [3.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert "2,6,123452,3,0.2,0,0,0,0,0" in cases[0]["ControlSequence"]


def test_barometer_cases_use_pressure_bias_mode() -> None:
    cases = MODULE.build_cases()
    assert [case["_metadata"]["bias"] for case in cases[-3:]] == [50.0, 100.0, 150.0]
    assert [case["_metadata"]["model_bias"] for case in cases[-3:]] == [0.5, 1.0, 1.5]
    assert cases[-1]["_metadata"]["fault_parameters"] == [3.0, 1.5, 0.0]
