from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


CASE_ID_START = 4521

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

VECTOR_SENSOR_SPECS = [
    {
        "fault_id": 123452,
        "sensor": "accelerometer",
        "bias": 0.2,
        "columns": [
            "_sensor_combined_0_accelerometer_m_s2[0]",
            "_sensor_combined_0_accelerometer_m_s2[1]",
            "_sensor_combined_0_accelerometer_m_s2[2]",
        ],
        "unit": "m/s^2",
        "minimum_shift": 0.05,
    },
    {
        "fault_id": 123453,
        "sensor": "gyroscope",
        "bias": 0.05,
        "columns": [
            "_sensor_combined_0_gyro_rad[0]",
            "_sensor_combined_0_gyro_rad[1]",
            "_sensor_combined_0_gyro_rad[2]",
        ],
        "unit": "rad/s",
        "minimum_shift": 0.005,
    },
    {
        "fault_id": 123455,
        "sensor": "magnetometer",
        "bias": 0.05,
        "columns": [
            "_vehicle_magnetometer_0_magnetometer_ga[0]",
            "_vehicle_magnetometer_0_magnetometer_ga[1]",
            "_vehicle_magnetometer_0_magnetometer_ga[2]",
        ],
        "unit": "gauss",
        "minimum_shift": 0.01,
    },
]

BAROMETER_BIASES_PA = (50.0, 100.0, 150.0)


def _vector_fault_parameters(axis_index: int, bias: float) -> list[float]:
    parameters = [3.0]
    for current_axis in range(3):
        parameters.extend([bias if current_axis == axis_index else 0.0, 0.0])
    return parameters


def _format_parameter(value: float) -> str:
    if value == 0:
        return "0"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _build_case(
    case_id: int,
    fault_id: int,
    sensor: str,
    parameters: list[float],
    metadata: dict[str, object],
) -> dict[str, object]:
    parameter_text = ",".join(_format_parameter(value) for value in parameters)
    control_sequence = ";".join(
        [
            "2,1,0.5,1,15",
            "1,1,10",
            f"2,6,{fault_id},{parameter_text}",
            "1,1,15",
        ]
    )
    return {
        "CaseID": case_id,
        "Subsystem": "Sensor subsystem",
        "FaultType": str(fault_id),
        "ControlSequence": control_sequence,
        "InterestedLog": {},
        "TestStatus": "UnFinished",
        "UniqueIndex": 1_000_000_000 + (fault_id - 123450) * 1_000_000 + case_id,
        "_metadata": {
            "fault_class": "sensor",
            "fault_name": f"{sensor}_bias",
            "fault_id": fault_id,
            "fault_mode": 3,
            "sensor": sensor,
            "fault_parameters": parameters,
            "injection_after_hover_s": 10,
            "fault_duration_s": 15,
            "target_altitude_m": 15,
            "termination": "platform_cleanup_after_fault_window",
            "dataset_role": "held_out_rfly_validation",
            **metadata,
        },
    }


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    case_id = CASE_ID_START
    axis_names = ("x", "y", "z")
    for spec in VECTOR_SENSOR_SPECS:
        for axis_index, axis_name in enumerate(axis_names):
            parameters = _vector_fault_parameters(axis_index, float(spec["bias"]))
            cases.append(
                _build_case(
                    case_id,
                    int(spec["fault_id"]),
                    str(spec["sensor"]),
                    parameters,
                    {
                        "axis": axis_name,
                        "axis_index": axis_index,
                        "bias": float(spec["bias"]),
                        "unit": str(spec["unit"]),
                        "signal_column": spec["columns"][axis_index],
                        "minimum_signal_shift": float(spec["minimum_shift"]),
                    },
                )
            )
            case_id += 1

    for bias_pa in BAROMETER_BIASES_PA:
        # The DLL writes MAVLink HIL_SENSOR abs_pressure in hPa/mbar, while
        # vehicle_air_data exports pressure in Pa.
        model_bias_hpa = bias_pa / 100.0
        cases.append(
            _build_case(
                case_id,
                123454,
                "barometer",
                [3.0, model_bias_hpa, 0.0],
                {
                    "axis": "pressure",
                    "axis_index": 0,
                    "bias": bias_pa,
                    "unit": "Pa",
                    "model_bias": model_bias_hpa,
                    "model_bias_unit": "hPa",
                    "signal_column": "_vehicle_air_data_0_baro_pressure_pa",
                    "minimum_signal_shift": 20.0,
                },
            )
        )
        case_id += 1
    return cases


def _write_sensor_launcher(platform_root: Path) -> Path:
    source = platform_root / "bat" / "FaultModelSITL.bat"
    destination = platform_root / "bat" / "FaultModelSITLSensor.bat"
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
            raise RuntimeError("Could not add PX4 instance cleanup to sensor launcher")
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
        raise ValueError(f"Unknown sensor case IDs: {invalid}; valid IDs: {sorted(case_ids)}")

    db_path = platform_root / "AutoTestAPI" / "db.json"
    launcher_path = _write_sensor_launcher(platform_root)
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
        "testbat": "FaultModelSITLSensor",
        "Vision": "off",
    }
    db_path.write_text(json.dumps(db, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "sensor_fault_case_manifest.json"
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
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if backup_path:
        print(f"[INFO] backed up db.json: {backup_path}")
    print(f"[OK] wrote sensor launcher: {launcher_path}")
    print(f"[OK] wrote platform cases: {db_path}")
    print(f"[OK] wrote case manifest: {manifest_path}")
    print(f"[INFO] selected cases: {selected_case_ids}")
    return db_path, launcher_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare controlled sensor-bias fault cases.")
    parser.add_argument("--platform-root", default=r"D:\BuaaAutoTestPlatForm")
    parser.add_argument(
        "--output-root", default=r"D:\UAV-Data\rfly_sensor_fault_validation"
    )
    parser.add_argument("--case-id", type=int, action="append", dest="case_ids")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    all_case_ids = [int(case["CaseID"]) for case in build_cases()]
    selected = all_case_ids if args.all else (args.case_ids or [CASE_ID_START])
    prepare(Path(args.platform_root), Path(args.output_root), selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
