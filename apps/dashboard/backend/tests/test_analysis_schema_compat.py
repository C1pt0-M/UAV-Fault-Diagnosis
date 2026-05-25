from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[4]
UAV_FAULT_ROOT = REPO_ROOT / "packages" / "uav_fault"
if str(UAV_FAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(UAV_FAULT_ROOT))

from uav_dashboard_api.services.analysis import build_fields_response, build_timeseries_response


def _write_new_schema_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "trueTime,_actuator_outputs_0_output[0],_actuator_outputs_0_output[1],_actuator_outputs_0_output[2],_actuator_outputs_0_output[3],_vehicle_attitude_0_q[0],_vehicle_attitude_0_q[1],_vehicle_attitude_0_q[2],_vehicle_attitude_0_q[3],_vehicle_air_data_0_baro_alt_meter,_vehicle_local_position_0_z",
                "1.5,1200,1190,1210,1185,0.9553364891,0,0,0.2955202067,15.2,-10.0",
                "2.5,1210,1200,1220,1195,0.9800665778,0,0,0.1986693308,15.5,-10.4",
            ]
        ),
        encoding="utf-8",
    )


def test_build_fields_response_exposes_compat_fields_for_new_processed_csv(
    tmp_path: Path,
):
    csv_path = tmp_path / "case_new_schema.csv"
    _write_new_schema_csv(csv_path)

    payload = build_fields_response(str(csv_path))

    assert payload["time"]["key"] == "time_s"
    assert payload["time"]["min"] == 1.5
    assert payload["time"]["max"] == 2.5

    fields = {
        item["field"]
        for group in payload["field_groups"]
        for item in group["fields"]
    }
    assert {"motor1", "motor2", "yaw", "baro_alt", "pos_z"}.issubset(fields)


def test_build_timeseries_response_reads_new_processed_csv_via_compat_fields(
    tmp_path: Path,
):
    csv_path = tmp_path / "case_new_schema.csv"
    _write_new_schema_csv(csv_path)

    payload = build_timeseries_response(
        str(csv_path),
        fields=["motor1", "yaw", "pos_z"],
        start_t=1.5,
        end_t=2.5,
        max_points=10,
    )

    assert payload["time_key"] == "time_s"
    assert payload["x"] == [1.5, 2.5]

    series = {item["field"]: item["values"] for item in payload["series"]}
    assert series["motor1"] == [1200.0, 1210.0]
    assert abs(series["yaw"][0] - 0.6) < 1e-3
    assert abs(series["yaw"][1] - 0.4) < 1e-3
    assert series["pos_z"] == [-10.0, -10.4]
