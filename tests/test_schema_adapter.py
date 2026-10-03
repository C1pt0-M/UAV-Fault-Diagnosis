import math

from uav_fault.telemetry.schema_adapter import (
    ACTIVE_FEATURE_SCHEMA,
    CANONICAL_SCHEMA_VERSION,
    SEQ_COLUMNS,
    adapt_px4_live_frame,
    assess_canonical_input_contract,
    canonicalize_rfly_frame,
)


def test_canonicalize_rfly_frame_returns_full_canonical_columns_with_defaults():
    # roll/pitch/yaw are radians on every live path (see
    # tests/test_euler_quaternion_units.py).
    canonical, coverage = canonicalize_rfly_frame({
        'time_s': 1.25,
        '_actuator_outputs_0_output[0]': 0.2,
        'roll': 0.1745,
        'pitch': -0.0873,
        'yaw': 1.5708,
    })

    assert CANONICAL_SCHEMA_VERSION == 'rfly_seq_v1'
    assert ACTIVE_FEATURE_SCHEMA == 'rfly_seq_columns_v2'
    assert len(SEQ_COLUMNS) == 26
    assert all(column in canonical for column in SEQ_COLUMNS)
    assert canonical['actuator_out_1'] == 0.2
    assert canonical['actuator_out_2'] == 0.0
    assert canonical['gyro_x'] == 0.0
    assert canonical['mag_z'] == 0.0
    assert canonical['time_s'] == 1.25
    assert 'actuator_out_2' in coverage['default']
    assert 'gyro_x' in coverage['default']
    assert 'att_q_w' in coverage['derived']


def test_canonicalize_rfly_frame_maps_live_canonical_aliases_for_replay():
    canonical, coverage = canonicalize_rfly_frame({
        'time_s': 1.0,
        'motor1': 1300.0,
        'motor2': 1310.0,
        'motor3': 1290.0,
        'motor4': 1305.0,
        'vx': 1.2,
        'vy': -0.3,
        'vz': -0.8,
        'baro_alt_meter': 12.5,
        'baro_pressure_pa': 100900.0,
        'q_w': 0.99,
        'q_x': 0.01,
        'q_y': -0.02,
        'q_z': 0.03,
    })

    assert canonical['actuator_out_1'] == 1300.0
    assert canonical['actuator_out_4'] == 1305.0
    assert canonical['vel_x'] == 1.2
    assert canonical['vel_y'] == -0.3
    assert canonical['vel_z'] == -0.8
    assert canonical['baro_alt'] == 12.5
    assert canonical['baro_pressure'] == 100900.0
    assert canonical['att_q_w'] == 0.99
    assert canonical['att_q_x'] == 0.01
    assert canonical['att_q_y'] == -0.02
    assert canonical['att_q_z'] == 0.03
    assert 'baro_pressure' in coverage['mapped']
    assert 'att_q_w' in coverage['mapped']


def test_canonicalize_rfly_frame_keeps_normalized_motor_replay_untrusted():
    records = [
        canonicalize_rfly_frame({
            'time_s': idx * 0.02,
            'motor1': 0.13,
            'motor2': 0.13,
            'motor3': 0.13,
            'motor4': 0.13,
            'accel_z': -9.8,
            'baro_pressure_pa': 101325.0,
        })[0]
        for idx in range(150)
    ]

    status = assess_canonical_input_contract(records)

    assert records[-1]['baro_pressure'] == 101325.0
    assert status['detection_blocked'] is True
    assert 'actuator_outputs' in status['zero_signal_groups']


def test_adapt_px4_live_frame_returns_full_canonical_columns_and_derived_values():
    canonical, coverage = adapt_px4_live_frame({
        'time_s': 2.0,
        'seq': 7,
        'motor1': 0.1,
        'motor2': 0.2,
        'motor3': 0.3,
        'motor4': 0.4,
        'accel_x': 3.0,
        'accel_y': 4.0,
        'accel_z': 0.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
    })

    assert all(column in canonical for column in SEQ_COLUMNS)
    assert canonical['time_s'] == 2.0
    assert canonical['seq'] == 7
    assert canonical['actuator_out_1'] == 0.1
    assert canonical['actuator_out_4'] == 0.4
    # baro_alt_relative 在单帧阶段拿绝对高度；本帧没有气压输入，所以为 0。
    assert canonical['baro_alt_relative'] == 0.0
    assert canonical['acc_norm'] == 5.0
    # 26 通道 schema 已剔除 PWM 聚合量。
    assert 'actuator_mean' not in SEQ_COLUMNS
    assert 'actuator_effort' not in SEQ_COLUMNS
    assert 'actuator_mean' not in canonical
    assert 'actuator_effort' not in canonical


def test_adapt_px4_live_frame_prefers_actuator_outputs_over_normalized_motors():
    canonical, coverage = adapt_px4_live_frame({
        'time_s': 2.0,
        'seq': 7,
        'motor1': 0.1,
        'motor2': 0.2,
        'motor3': 0.3,
        'motor4': 0.4,
        'actuator_out_1': 1300.0,
        'actuator_out_2': 1310.0,
        'actuator_out_3': 1290.0,
        'actuator_out_4': 1305.0,
    })

    assert canonical['actuator_out_1'] == 1300.0
    assert canonical['actuator_out_2'] == 1310.0
    assert canonical['actuator_out_3'] == 1290.0
    assert canonical['actuator_out_4'] == 1305.0
    assert 'actuator_out_1' in coverage['mapped']


def test_adapt_px4_live_frame_prefers_live_quaternion_magnetometer_and_baro_pressure_fields():
    canonical, coverage = adapt_px4_live_frame({
        'time_s': 3.0,
        'seq': 9,
        'q_w': 0.9,
        'q_x': 0.1,
        'q_y': -0.2,
        'q_z': 0.3,
        'mag_x': 0.01,
        'mag_y': -0.02,
        'mag_z': 0.03,
        'baro_pressure_pa': 100100.0,
        'roll': 0.0873,
        'pitch': 0.1047,
        'yaw': 0.1222,
    })

    assert canonical['att_q_w'] == 0.9
    assert canonical['att_q_x'] == 0.1
    assert canonical['att_q_y'] == -0.2
    assert canonical['att_q_z'] == 0.3
    assert canonical['mag_x'] == 0.01
    assert canonical['mag_y'] == -0.02
    assert canonical['mag_z'] == 0.03
    assert canonical['baro_pressure'] == 100100.0
    assert 'att_q_w' in coverage['mapped']
    assert 'mag_x' in coverage['mapped']
    assert 'baro_pressure' in coverage['mapped']


def test_adapt_px4_live_frame_maps_live_baro_temp_and_actuator_controls():
    canonical, coverage = adapt_px4_live_frame({
        'time_s': 4.0,
        'seq': 10,
        'baro_temp': 24.5,
        'actuator_ctrl_1': 0.11,
        'actuator_ctrl_2': -0.22,
        'actuator_ctrl_3': 0.33,
        'actuator_ctrl_4': 0.44,
    })

    assert canonical['baro_temp'] == 24.5
    assert canonical['actuator_ctrl_1'] == 0.11
    assert canonical['actuator_ctrl_2'] == -0.22
    assert canonical['actuator_ctrl_3'] == 0.33
    assert canonical['actuator_ctrl_4'] == 0.44
    assert 'baro_temp' in coverage['mapped']
    assert 'actuator_ctrl_1' in coverage['mapped']
    assert 'actuator_ctrl_4' in coverage['mapped']


def test_assess_canonical_input_contract_blocks_untrusted_zero_signals():
    records = [
        {
            'actuator_out_1': 0.0,
            'actuator_out_2': 0.0,
            'actuator_out_3': 0.0,
            'actuator_out_4': 0.0,
            'accel_z': 0.0,
            'vel_x': 0.0,
            'vel_y': 0.0,
            'vel_z': 0.0,
            'baro_alt': 0.0,
        }
        for _ in range(150)
    ]

    status = assess_canonical_input_contract(records)

    assert status['status'] == 'degraded'
    assert status['reason'] == 'untrusted_zero_signals'
    assert status['detection_blocked'] is True
    assert 'actuator_outputs' in status['zero_signal_groups']
    assert 'imu_accel_z' in status['zero_signal_groups']


def test_assess_canonical_input_contract_marks_proxy_degraded_without_blocking():
    records = [
        {
            'actuator_out_1': 1300.0,
            'actuator_out_2': 1310.0,
            'actuator_out_3': 1290.0,
            'actuator_out_4': 1305.0,
            'actuator_source': 'proxy',
            'accel_z': -9.8,
            'vel_x': 0.1,
            'vel_y': 0.0,
            'vel_z': -0.2,
            'baro_alt': 12.0,
        }
        for _ in range(150)
    ]

    status = assess_canonical_input_contract(records, actuator_source='proxy')

    assert status['status'] == 'degraded'
    assert status['reason'] == 'proxy_actuator_source'
    assert status['detection_blocked'] is False


def test_assess_canonical_input_contract_blocks_preflight_motionless_window():
    records = [
        {
            'actuator_out_1': 1000.0,
            'actuator_out_2': 1000.0,
            'actuator_out_3': 1000.0,
            'actuator_out_4': 1000.0,
            'accel_z': -9.8,
            'vel_x': 0.0,
            'vel_y': 0.0,
            'vel_z': 0.0,
            'baro_alt': 0.0,
        }
        for _ in range(150)
    ]

    status = assess_canonical_input_contract(records)

    assert status['status'] == 'degraded'
    assert status['reason'] == 'untrusted_zero_signals'
    assert status['detection_blocked'] is True
    assert 'motion_reference' in status['zero_signal_groups']


def test_assess_canonical_input_contract_blocks_preflight_idle_with_gps_noise():
    records = [
        {
            'actuator_out_1': 900.0,
            'actuator_out_2': 900.0,
            'actuator_out_3': 900.0,
            'actuator_out_4': 900.0,
            'accel_z': -9.8,
            'vel_x': 0.02,
            'vel_y': -0.06,
            'vel_z': -0.01,
            'baro_alt': 0.0,
        }
        for _ in range(150)
    ]

    status = assess_canonical_input_contract(records)

    assert status['detection_blocked'] is True
    assert 'preflight_idle' in status['zero_signal_groups']
