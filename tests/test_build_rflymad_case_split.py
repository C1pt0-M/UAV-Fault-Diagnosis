from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "data"
    / "build_rflymad_case_split.py"
)
SPEC = spec_from_file_location("build_rflymad_case_split", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _row(
    domain: str,
    output_number: str,
    *,
    level3: str = "motor",
    flight_status: str = "hover",
    closed: bool = True,
    open_type: str = "",
) -> dict[str, str]:
    level1 = "fault" if level3 != "normal" or open_type else "normal"
    level2 = (
        ""
        if open_type
        else "sensor"
        if level3 not in {"normal", "motor"}
        else level3
    )
    return {
        "domain": domain,
        "case_uid": f"{domain}/{flight_status}/{level3 or open_type}/{output_number}",
        "flight_status": flight_status,
        "source_fault_type": open_type or level3,
        "output_filename": f"Case_{output_number}.csv",
        "final_output_path": rf"D:\output\{domain}\Case_{output_number}.csv",
        "level1_label": level1,
        "level2_label": level2,
        "level3_label": "" if open_type else level3,
        "closed_set_eligible": str(closed),
        "open_set_eligible": str(not closed),
        "open_set_fault_type": open_type,
        "paired_topology_eligible": str(closed and level3 not in {"normal", ""}),
        "window_usage": (
            "row_level_open_set_windows"
            if open_type
            else "normal_windows_only"
            if level3 == "normal"
            else "row_level_closed_set_windows"
        ),
    }


def test_apportion_group_counts_preserves_total_and_expected_ratio():
    counts = MODULE.apportion_group_counts(10)

    assert counts == {"train": 7, "val": 2, "test": 1}
    assert sum(counts.values()) == 10


def test_global_rebalance_hits_exact_quota_without_removing_stratum_coverage():
    group_counts = {f"stratum-{index}": 44 for index in range(41)}
    group_counts["stratum-41"] = 47
    counts = {
        stratum: MODULE.apportion_group_counts(group_count)
        for stratum, group_count in group_counts.items()
    }

    rebalanced, targets = MODULE.rebalance_stratum_counts(counts, group_counts)

    assert targets == {"train": 1296, "val": 278, "test": 277}
    assert {
        split: sum(row[split] for row in rebalanced.values())
        for split in MODULE.INTERNAL_SPLITS
    } == targets
    assert all(
        row[split] >= 1
        for row in rebalanced.values()
        for split in MODULE.INTERNAL_SPLITS
    )


def test_simulation_pair_key_matches_sil_and_hil_counterparts():
    sil = _row("SIL", "1005000007", level3="accelerometer")
    hil = _row("HIL", "2005000007", level3="accelerometer")

    assert MODULE.simulation_pair_key(sil) == "005000007"
    assert MODULE.simulation_pair_key(hil) == "005000007"


def test_paired_simulation_cases_are_assigned_to_the_same_split():
    rows = []
    for index in range(10):
        suffix = f"0000000{index:02d}"
        rows.extend(
            [
                _row("SIL", f"1{suffix}", level3="motor"),
                _row("HIL", f"2{suffix}", level3="motor"),
            ]
        )

    result = MODULE.attach_split_fields(rows, "seed")
    split_by_key: dict[str, set[str]] = {}
    for row in result:
        split_by_key.setdefault(row["simulation_pair_key"], set()).add(row["split"])

    assert all(len(splits) == 1 for splits in split_by_key.values())
    assert {row["split"] for row in result} == {"train", "val", "test"}


def test_real_cases_are_external_test_only():
    result = MODULE.attach_split_fields(
        [_row("REAL", "3000000000", level3="motor")], "seed"
    )

    assert result[0]["split"] == "external_test"
    assert result[0]["normalization_fit_eligible"] is False
    assert result[0]["model_selection_eligible"] is False
    assert result[0]["threshold_tuning_eligible"] is False
    assert result[0]["final_evaluation_only"] is True


def test_open_set_simulation_cases_are_open_set_test_only():
    rows = [
        _row(
            "SIL",
            "1001000000",
            level3="",
            closed=False,
            open_type="propeller",
        ),
        _row(
            "HIL",
            "2001000000",
            level3="",
            closed=False,
            open_type="propeller",
        ),
    ]

    result = MODULE.attach_split_fields(rows, "seed")

    assert {row["split"] for row in result} == {"open_set_test"}
    assert {row["simulation_pair_status"] for row in result} == {"paired"}


def test_grouping_rejects_label_mismatch_between_sil_and_hil():
    rows = [
        _row("SIL", "1005000000", level3="accelerometer"),
        _row("HIL", "2005000000", level3="gyroscope"),
    ]

    with pytest.raises(ValueError, match="inconsistent labels"):
        MODULE.group_simulation_rows(rows)
