from uav_fault.features.px4_adapter import adapt_px4_row_to_shared_raw


def test_adapt_px4_row_to_shared_raw_maps_core_live_fields():
    row = {
        "time_s": 2.5,
        "gyro_x": 0.1,
        "gyro_y": -0.2,
        "gyro_z": 0.3,
        "accel_x": 1.1,
        "accel_y": 2.2,
        "accel_z": 3.3,
        "baro_alt_meter": 9.8,
        "baro_pressure_pa": 101325.0,
        "q_w": 1.0,
        "q_x": 0.0,
        "q_y": 0.0,
        "q_z": 0.0,
        "vx": 4.0,
        "vy": 5.0,
        "vz": 6.0,
    }

    shared = adapt_px4_row_to_shared_raw(row)

    assert shared["time_s"] == 2.5
    assert shared["gyro_x"] == 0.1
    assert shared["accel_z"] == 3.3
    assert shared["baro_pressure"] == 101325.0
    assert shared["att_q0"] == 1.0
    assert shared["vel_z"] == 6.0


def test_adapt_px4_row_to_shared_raw_derives_quaternion_from_euler_and_maps_optional_fields():
    row = {
        "time_s": 1.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "mag_x": 0.01,
        "mag_y": -0.02,
        "mag_z": 0.03,
        "esc_rpm_1": 410.0,
        "esc_rpm_2": 420.0,
        "esc_rpm_3": 430.0,
        "esc_rpm_4": 440.0,
    }

    shared = adapt_px4_row_to_shared_raw(row)

    assert shared["att_q0"] == 1.0
    assert shared["att_q1"] == 0.0
    assert shared["att_q2"] == 0.0
    assert shared["att_q3"] == 0.0
    assert shared["mag_x"] == 0.01
    assert shared["rpm_1"] == 410.0
    assert shared["rpm_4"] == 440.0
