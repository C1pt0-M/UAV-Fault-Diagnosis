from uav_fault.features.rfly_adapter import adapt_rfly_row_to_shared_raw


def test_adapt_rfly_row_to_shared_raw_maps_core_fields_and_defaults_missing_values():
    row = {
        "trueTime": 12.5,
        "_sensor_combined_0_gyro_rad[0]": 0.1,
        "_sensor_combined_0_gyro_rad[1]": -0.2,
        "_sensor_combined_0_gyro_rad[2]": 0.3,
        "_sensor_combined_0_accelerometer_m_s2[0]": 1.1,
        "_sensor_combined_0_accelerometer_m_s2[1]": 2.2,
        "_sensor_combined_0_accelerometer_m_s2[2]": 3.3,
        "_vehicle_air_data_0_baro_alt_meter": 9.8,
        "_vehicle_air_data_0_baro_pressure_pa": 101325.0,
        "_vehicle_attitude_0_q[0]": 1.0,
        "_vehicle_attitude_0_q[1]": 0.0,
        "_vehicle_attitude_0_q[2]": 0.0,
        "_vehicle_attitude_0_q[3]": 0.0,
        "_vehicle_local_position_0_vx": 4.0,
        "_vehicle_local_position_0_vy": 5.0,
        "_vehicle_local_position_0_vz": 6.0,
    }

    shared = adapt_rfly_row_to_shared_raw(row)

    assert shared["time_s"] == 12.5
    assert shared["gyro_x"] == 0.1
    assert shared["accel_z"] == 3.3
    assert shared["baro_pressure"] == 101325.0
    assert shared["att_q0"] == 1.0
    assert shared["vel_z"] == 6.0


def test_adapt_rfly_row_to_shared_raw_maps_optional_mag_and_rpm_proxy_fields():
    row = {
        "Timestamp": 1500000,
        "_vehicle_magnetometer_0_magnetometer_ga[0]": 0.01,
        "_vehicle_magnetometer_0_magnetometer_ga[1]": -0.02,
        "_vehicle_magnetometer_0_magnetometer_ga[2]": 0.03,
        "TrueState_data_motorRPMs[1]": 4100.0,
        "TrueState_data_motorRPMs[2]": 4200.0,
        "TrueState_data_motorRPMs[3]": 4300.0,
        "TrueState_data_motorRPMs[4]": 4400.0,
    }

    shared = adapt_rfly_row_to_shared_raw(row)

    assert shared["time_s"] == 1.5
    assert shared["mag_x"] == 0.01
    assert shared["mag_y"] == -0.02
    assert shared["mag_z"] == 0.03
    assert shared["rpm_1"] == 4100.0
    assert shared["rpm_4"] == 4400.0
