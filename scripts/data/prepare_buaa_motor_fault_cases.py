from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


FAULT_ID = 123450
CASE_ID_START = 4501
MOTOR_COUNT = 4
# The DLL multiplies absolute PWM before converting it back to normalized
# throttle. At a 1500 us hover point these values are roughly 3%, 9% and 15%
# normalized-command loss, so 0.9 would be much more severe than "10% loss".
SEVERITIES = (0.99, 0.97, 0.95)

INTERESTED_LOGS = [
    "*actuator_outputs_0.csv",
    "*actuator_controls_0_0.csv",
    "*estimator_status_0.csv",
    "*vehicle_command_0.csv",
    "*sensor_combined_0.csv",
    "*vehicle_magnetometer_0.csv",
    "*estimator_attitude_0.csv",
    "*vehicle_attitude_0.csv",
    "*vehicle_gps_position_0.csv",
    "*vehicle_global_position_0.csv",
    "*vehicle_local_position_0.csv",
    "*vehicle_air_data_0.csv",
    "*rfly_ctrl_lxl_0.csv",
]


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    case_id = CASE_ID_START
    for coefficient in SEVERITIES:
        for motor_index in range(1, MOTOR_COUNT + 1):
            coefficients = [1.0] * MOTOR_COUNT
            coefficients[motor_index - 1] = coefficient
            coefficient_text = ",".join(
                "1.0" if value == 1.0 else f"{value:.2f}" for value in coefficients
            )
            control_sequence = ";".join(
                [
                    "2,1,0.5,1,15",
                    "1,1,10",
                    f"2,6,{FAULT_ID},1,{coefficient_text}",
                    "1,1,15",
                ]
            )
            cases.append(
                {
                    "CaseID": case_id,
                    "Subsystem": "Power subsystem",
                    "FaultType": str(FAULT_ID),
                    "ControlSequence": control_sequence,
                    "InterestedLog": {},
                    "TestStatus": "UnFinished",
                    "UniqueIndex": 1_000_000_000 + case_id,
                    "_metadata": {
                        "fault_class": "motor",
                        "fault_name": "motor_efficiency_loss",
                        "fault_id": FAULT_ID,
                        "fault_mode": 1,
                        "motor_index": motor_index,
                        "efficiency_coefficient": coefficient,
                        "injection_after_hover_s": 10,
                        "fault_duration_s": 15,
                        "target_altitude_m": 15,
                        "termination": "platform_cleanup_after_fault_window",
                        "dataset_role": "held_out_rfly_validation",
                    },
                }
            )
            case_id += 1
    return cases


def _write_motor_launcher(platform_root: Path) -> Path:
    source = platform_root / "bat" / "FaultModelSITL.bat"
    destination = platform_root / "bat" / "FaultModelSITLMotor.bat"
    text = source.read_text(encoding="utf-8-sig")
    text, dll_count = re.subn(
        r"(?im)^\s*set\s+DLLModel\s*=.*$",
        "set DLLModel=FaultModel450d_new",
        text,
        count=1,
    )
    text, frame_count = re.subn(
        r"(?im)^\s*set\s+PX4SitlFrame\s*=.*$",
        "set PX4SitlFrame=quadrotor_x",
        text,
        count=1,
    )
    if dll_count != 1 or frame_count != 1:
        raise RuntimeError("Could not patch DLLModel/PX4SitlFrame in FaultModelSITL.bat")
    cleanup_command = (
        'IF EXIST "%PSP_PATH%\\Firmware\\build\\px4_sitl_default\\instance_1" '
        'RMDIR /S /Q "%PSP_PATH%\\Firmware\\build\\px4_sitl_default\\instance_1"'
    )
    if cleanup_command not in text:
        text, cleanup_count = re.subn(
            r"(?im)^(wsl -d %WSL_DISTRO% pkill -x px4)[ \t]*$",
            lambda match: f"{match.group(1)}\r\n{cleanup_command}",
            text,
            count=1,
        )
        if cleanup_count != 1:
            raise RuntimeError("Could not add PX4 instance cleanup to motor launcher")
    destination.write_text(text, encoding="utf-8", newline="\r\n")
    return destination


def _backup(path: Path, backup_dir: Path) -> Path | None:
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target = backup_dir / f"{path.name}.{timestamp}.bak"
    shutil.copy2(path, target)
    return target


def prepare(
    platform_root: Path,
    output_root: Path,
    selected_case_ids: list[int],
) -> tuple[Path, Path, Path]:
    cases_with_metadata = build_cases()
    case_ids = {int(case["CaseID"]) for case in cases_with_metadata}
    invalid = sorted(set(selected_case_ids) - case_ids)
    if invalid:
        raise ValueError(f"Unknown motor case IDs: {invalid}; valid IDs: {sorted(case_ids)}")

    api_dir = platform_root / "AutoTestAPI"
    db_path = api_dir / "db.json"
    launcher_path = _write_motor_launcher(platform_root)
    backup_path = _backup(db_path, output_root / "db_backups")

    public_cases = []
    manifest_cases = []
    for case in cases_with_metadata:
        public_case = {key: value for key, value in case.items() if key != "_metadata"}
        public_cases.append(public_case)
        manifest_cases.append({**public_case, **case["_metadata"]})

    db = {
        "faultcase": public_cases,
        "testdata": {pattern: [] for pattern in INTERESTED_LOGS},
        "testcase": ",".join(str(value) for value in selected_case_ids),
        "PlatFormPath": "D:/PX4PSP",
        "testbat": "FaultModelSITLMotor",
        "Vision": "off",
    }
    db_path.write_text(json.dumps(db, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "motor_fault_case_manifest.json"
    manifest = {
        "schema_version": 1,
        "platform_root": str(platform_root),
        "platform_db": str(db_path),
        "launcher": str(launcher_path),
        "fault_model": "FaultModel450d_new",
        "px4_frame": "quadrotor_x",
        "selected_case_ids": selected_case_ids,
        "cases": manifest_cases,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if backup_path:
        print(f"[INFO] backed up db.json: {backup_path}")
    print(f"[OK] wrote motor launcher: {launcher_path}")
    print(f"[OK] wrote platform cases: {db_path}")
    print(f"[OK] wrote case manifest: {manifest_path}")
    print(f"[INFO] selected cases: {selected_case_ids}")
    return db_path, launcher_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare controlled BuaaAutoTestPlatform motor-efficiency fault cases."
    )
    parser.add_argument(
        "--platform-root",
        default=r"D:\BuaaAutoTestPlatForm",
        help="BuaaAutoTestPlatform root directory.",
    )
    parser.add_argument(
        "--output-root",
        default=r"D:\UAV-Data\rfly_motor_fault_validation",
        help="Fault dataset metadata/output directory.",
    )
    parser.add_argument(
        "--case-id",
        type=int,
        action="append",
        dest="case_ids",
        help="Case ID to select; repeat for multiple cases. Defaults to smoke case 4501.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Select all 12 cases.",
    )
    args = parser.parse_args()

    all_case_ids = [int(case["CaseID"]) for case in build_cases()]
    selected = all_case_ids if args.all else (args.case_ids or [CASE_ID_START])
    prepare(Path(args.platform_root), Path(args.output_root), selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
