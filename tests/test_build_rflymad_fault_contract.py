from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "data"
    / "build_rflymad_fault_contract.py"
)
SPEC = spec_from_file_location("build_rflymad_fault_contract", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _simulation_row(fault_type: str) -> dict[str, str]:
    return {
        "plan_index": "1",
        "flight_status": "hover",
        "fault_type": fault_type,
        "raw_case_directory": "TestCase_1_1000000000",
        "source_path": rf"D:\RflyMAD\SIL\hover\{fault_type}\TestCase_1_1000000000",
        "output_filename": "Case_1000000000.csv",
        "final_output_path": r"D:\UAV-Data\processed_SIL_50hz\Case_1000000000.csv",
        "row_count": "2000",
        "column_count": "47",
        "median_step_us": "20000",
        "output_sha256": "a" * 64,
        "included": "True",
    }


def _real_row(
    *,
    main_label: str,
    subtype: str,
    disposition: str,
    window_usage: str,
    source_fault_type: str,
) -> dict[str, str]:
    return {
        "plan_index": "2",
        "flight_status": "waypoint",
        "source_fault_type": source_fault_type,
        "raw_case_directory": "123_1",
        "source_path": rf"D:\RflyMAD\Real\waypoint\{source_fault_type}\123_1",
        "output_filename": "Case_3100000000.csv",
        "final_output_path": r"D:\UAV-Data\processed_Real_50hz\Case_3100000000.csv",
        "row_count": "3000",
        "column_count": "72",
        "output_sha256": "b" * 64,
        "disposition": disposition,
        "metadata_status": "ok",
        "effective_main3_label": main_label,
        "effective_fault_subtype": subtype,
        "window_usage": window_usage,
        "included_in_final_dataset": "True",
    }


def test_simulation_sensor_maps_to_hierarchical_closed_set_labels():
    result = MODULE.build_contract_row(
        _simulation_row("gyroscope"), "SIL", check_output_exists=False
    )

    assert result is not None
    assert result["level1_label"] == "fault"
    assert result["level2_label"] == "sensor"
    assert result["level3_label"] == "gyroscope"
    assert result["closed_set_eligible"] is True
    assert result["open_set_eligible"] is False
    assert result["closed_set_fault_window_eligible"] is True
    assert result["paired_topology_eligible"] is True


def test_simulation_extra_fault_is_open_set_and_has_no_closed_set_label():
    result = MODULE.build_contract_row(
        _simulation_row("propeller"), "HIL", check_output_exists=False
    )

    assert result is not None
    assert result["level1_label"] == "fault"
    assert result["level2_label"] == ""
    assert result["level3_label"] == ""
    assert result["closed_set_eligible"] is False
    assert result["open_set_eligible"] is True
    assert result["open_set_fault_type"] == "propeller"
    assert result["open_set_fault_window_eligible"] is True
    assert result["paired_topology_eligible"] is False


def test_real_normal_only_case_stays_normal_despite_fault_source_directory():
    row = _real_row(
        main_label="normal",
        subtype="normal",
        disposition="retain_normal_only",
        window_usage="normal_windows_only",
        source_fault_type="motor",
    )

    result = MODULE.build_contract_row(row, "REAL", check_output_exists=False)

    assert result is not None
    assert result["source_fault_type"] == "motor"
    assert result["level1_label"] == "normal"
    assert result["level2_label"] == "normal"
    assert result["level3_label"] == "normal"
    assert result["normal_window_eligible"] is True
    assert result["closed_set_fault_window_eligible"] is False
    assert result["event_detection_eligible"] is False


def test_real_sensor_subtype_conflict_retains_effective_sensor_subtype():
    row = _real_row(
        main_label="sensor",
        subtype="magnetometer",
        disposition="retain_sensor_family_subtype_conflict",
        window_usage="row_level_main3_windows",
        source_fault_type="GPS",
    )

    result = MODULE.build_contract_row(row, "REAL", check_output_exists=False)

    assert result is not None
    assert result["source_fault_type"] == "GPS"
    assert result["level2_label"] == "sensor"
    assert result["level3_label"] == "magnetometer"
    assert result["paired_topology_eligible"] is True


def test_unknown_simulation_fault_type_is_rejected():
    with pytest.raises(ValueError, match="unsupported SIL/HIL fault type"):
        MODULE.build_contract_row(
            _simulation_row("unknown_fault"), "SIL", check_output_exists=False
        )


def test_excluded_source_manifest_row_is_skipped():
    row = _simulation_row("motor")
    row["included"] = "False"

    assert MODULE.build_contract_row(row, "HIL", check_output_exists=False) is None
