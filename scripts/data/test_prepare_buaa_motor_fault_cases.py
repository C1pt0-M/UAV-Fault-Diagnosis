from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_buaa_motor_fault_cases.py")
SPEC = importlib.util.spec_from_file_location("prepare_buaa_motor_fault_cases", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_cases_covers_four_motors_and_three_severities() -> None:
    cases = MODULE.build_cases()
    assert len(cases) == 12
    assert [case["CaseID"] for case in cases] == list(range(4501, 4513))
    assert {case["_metadata"]["motor_index"] for case in cases} == {1, 2, 3, 4}
    assert {case["_metadata"]["efficiency_coefficient"] for case in cases} == {
        0.99,
        0.97,
        0.95,
    }


def test_motor_command_has_explicit_four_motor_coefficients() -> None:
    first = MODULE.build_cases()[0]
    assert "2,6,123450,1,0.99,1.0,1.0,1.0" in first["ControlSequence"]
    assert first["ControlSequence"].endswith("1,1,15")
    assert ";2,5" not in first["ControlSequence"]


def test_motor_launcher_resets_stale_px4_instance(tmp_path: Path) -> None:
    platform = tmp_path / "platform"
    bat_dir = platform / "bat"
    bat_dir.mkdir(parents=True)
    (bat_dir / "FaultModelSITL.bat").write_text(
        "set DLLModel=0\n"
        "set PX4SitlFrame=iris\n"
        "wsl -d %WSL_DISTRO% pkill -x px4\n",
        encoding="utf-8",
    )
    output = MODULE._write_motor_launcher(platform).read_text(encoding="utf-8")
    assert "set DLLModel=FaultModel450d_new" in output
    assert "set PX4SitlFrame=quadrotor_x" in output
    assert "RMDIR /S /Q" in output
    assert "px4_sitl_default\\instance_1" in output
