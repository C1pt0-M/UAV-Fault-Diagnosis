from pathlib import Path

import pandas as pd

from uav_dashboard_api.services.severity_recommendation import evaluate_severity_and_recommendations


RULE_PATH = Path(__file__).resolve().parents[1] / 'app' / 'configs' / 'severity_rules_v1.yaml'


def _write_csv(tmp_path, name: str, columns: dict):
    df = pd.DataFrame(columns)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def test_normal_should_return_none_level(tmp_path):
    csv_path = _write_csv(tmp_path, 'normal.csv', {'battery_voltage_v': [11.4, 11.5, 11.3]})
    data = evaluate_severity_and_recommendations(
        csv_path=csv_path,
        fault_label='normal',
        confidence=0.99,
        config_path=str(RULE_PATH),
    )

    assert data['severity']['level'] == 'none'
    assert data['recommendations']
    assert data['recommendation_meta']['fallback_used'] is False


def test_battery_failure_should_reach_heavy(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        'battery.csv',
        {
            'battery_warning': [2, 2, 2, 2, 2],
            'battery_faults': [1, 1, 1, 1, 1],
            'battery_remaining': [14, 13, 12, 11, 10],
            'battery_voltage_v': [11.6, 11.2, 10.9, 10.6, 10.3],
        },
    )
    data = evaluate_severity_and_recommendations(
        csv_path=csv_path,
        fault_label='battery_failure',
        confidence=0.95,
        config_path=str(RULE_PATH),
    )

    assert data['severity']['level'] == 'heavy'
    assert data['severity']['score'] is not None
    assert data['recommendations']


def test_motor_failure_should_reach_heavy(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        'motor.csv',
        {
            'motor1': [1.0] * 8,
            'motor2': [0.2] * 8,
            'motor3': [0.25] * 8,
            'motor4': [1.1] * 8,
            'esc_rpm_1': [8000] * 8,
            'esc_rpm_2': [3800] * 8,
            'esc_rpm_3': [4000] * 8,
            'esc_rpm_4': [8200] * 8,
            'gyro_vibration_metric': [1.3] * 8,
            'accel_vibration_metric': [3.2] * 8,
        },
    )
    data = evaluate_severity_and_recommendations(
        csv_path=csv_path,
        fault_label='motor_failure',
        confidence=0.93,
        config_path=str(RULE_PATH),
    )

    assert data['severity']['level'] == 'heavy'
    assert len(data['recommendations']) >= 1


def test_prop_imbalance_should_reach_heavy(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        'prop.csv',
        {
            'imbalanced_prop_metric': [0.18] * 6,
            'gyro_vibration_metric': [1.25] * 6,
            'motor1': [1.0] * 6,
            'motor2': [0.4] * 6,
            'motor3': [0.5] * 6,
            'motor4': [1.1] * 6,
        },
    )
    data = evaluate_severity_and_recommendations(
        csv_path=csv_path,
        fault_label='prop_imbalance',
        confidence=0.91,
        config_path=str(RULE_PATH),
    )

    assert data['severity']['level'] == 'heavy'
    assert data['recommendation_meta']['severity_level'] == 'heavy'


def test_thrust_loss_should_reach_heavy(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        'thrust.csv',
        {
            'thrust_z': [0.92, 0.95, 0.91, 0.9, 0.89, 0.93],
            'unalloc_thrust_z': [0.28, 0.31, 0.26, 0.27, 0.29, 0.3],
            'vz': [-2.2, -2.0, -1.9, -2.1, -1.8, -2.3],
        },
    )
    data = evaluate_severity_and_recommendations(
        csv_path=csv_path,
        fault_label='thrust_loss',
        confidence=0.92,
        config_path=str(RULE_PATH),
    )

    assert data['severity']['level'] == 'heavy'
    assert data['recommendations']


def test_missing_required_metrics_should_return_unknown(tmp_path):
    csv_path = _write_csv(tmp_path, 'unknown.csv', {'time_s': [0, 1, 2, 3]})
    data = evaluate_severity_and_recommendations(
        csv_path=csv_path,
        fault_label='motor_failure',
        confidence=0.88,
        config_path=str(RULE_PATH),
    )

    assert data['severity']['level'] == 'unknown'
    assert data['recommendation_meta']['fallback_used'] is True
    assert any(item.get('status') == 'missing' for item in data['severity']['evidence'])


def test_primary_metric_threshold_for_motor_failure(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        'motor_primary.csv',
        {
            'fault_severity': [0.24, 0.25, 0.26, 0.25, 0.24],
            'motor1': [1.0] * 5,
            'motor2': [0.8] * 5,
            'motor3': [0.9] * 5,
            'motor4': [0.85] * 5,
        },
    )
    data = evaluate_severity_and_recommendations(
        csv_path=csv_path,
        fault_label='motor_failure',
        confidence=0.9,
        config_path=str(RULE_PATH),
    )

    assert data['severity']['level'] == 'medium'
    assert data['recommendation_meta']['mode'] == 'primary_metric'


def test_primary_metric_threshold_for_thrust_loss_should_return_medium(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        'thrust_primary.csv',
        {
            'fault_severity': [0.32, 0.31, 0.33, 0.34, 0.35],
            'thrust_z': [0.91, 0.92, 0.93, 0.94, 0.95],
        },
    )
    data = evaluate_severity_and_recommendations(
        csv_path=csv_path,
        fault_label='thrust_loss',
        confidence=0.92,
        config_path=str(RULE_PATH),
    )

    assert data['severity']['level'] == 'medium'
    assert data['recommendation_meta']['mode'] == 'primary_metric'
