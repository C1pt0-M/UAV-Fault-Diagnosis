"""MAVSDK 遥测订阅协程，每个协程负责一组字段，异步更新共享状态。"""

import math

from mavsdk.telemetry import FlightMode

from .state import TelemetryState

# MAVSDK FlightMode → PX4 nav_state 映射
_FLIGHT_MODE_TO_NAV_STATE = {
    FlightMode.MANUAL: 0,
    FlightMode.ALTCTL: 1,
    FlightMode.POSCTL: 2,
    FlightMode.MISSION: 3,
    FlightMode.HOLD: 4,
    FlightMode.RETURN_TO_LAUNCH: 5,
    FlightMode.ACRO: 10,
    FlightMode.OFFBOARD: 14,
    FlightMode.STABILIZED: 15,
    FlightMode.TAKEOFF: 17,
    FlightMode.LAND: 18,
    FlightMode.FOLLOW_ME: 19,
}


async def subscribe_attitude(drone, state: TelemetryState) -> None:
    """attitude_euler → roll, pitch, yaw (度→弧度，匹配离线数据)"""
    async for euler in drone.telemetry.attitude_euler():
        state.update(
            roll=math.radians(euler.roll_deg),
            pitch=math.radians(euler.pitch_deg),
            yaw=math.radians(euler.yaw_deg),
        )


async def subscribe_angular_velocity(drone, state: TelemetryState) -> None:
    """attitude_angular_velocity_body → rollspeed, pitchspeed, yawspeed (rad/s)"""
    async for av in drone.telemetry.attitude_angular_velocity_body():
        state.update(
            rollspeed=av.roll_rad_s,
            pitchspeed=av.pitch_rad_s,
            yawspeed=av.yaw_rad_s,
        )


async def subscribe_position_velocity(drone, state: TelemetryState) -> None:
    """position_velocity_ned → vx, vy, vz, pos_z"""
    async for pv in drone.telemetry.position_velocity_ned():
        state.update(
            vx=pv.velocity.north_m_s,
            vy=pv.velocity.east_m_s,
            vz=pv.velocity.down_m_s,
            pos_z=pv.position.down_m,
        )


async def subscribe_imu(drone, state: TelemetryState) -> None:
    """imu → accel_x/y/z, gyro_x/y/z (体轴系 FRD)"""
    async for imu_data in drone.telemetry.imu():
        state.update(
            accel_x=imu_data.acceleration_frd.forward_m_s2,
            accel_y=imu_data.acceleration_frd.right_m_s2,
            accel_z=imu_data.acceleration_frd.down_m_s2,
            gyro_x=imu_data.angular_velocity_frd.forward_rad_s,
            gyro_y=imu_data.angular_velocity_frd.right_rad_s,
            gyro_z=imu_data.angular_velocity_frd.down_rad_s,
        )


async def subscribe_battery(drone, state: TelemetryState) -> None:
    """battery → voltage, remaining (归一化到 0~1)"""
    async for bat in drone.telemetry.battery():
        remaining = bat.remaining_percent
        if remaining > 1.0:
            remaining = remaining / 100.0
        state.update(
            battery_voltage_v=bat.voltage_v,
            battery_remaining=max(0.0, min(1.0, remaining)),
        )


async def subscribe_actuator(drone, state: TelemetryState) -> None:
    """actuator_output_status → motor1~4 (归一化到 0~1，匹配离线 actuator_motors.control)"""
    async for act in drone.telemetry.actuator_output_status():
        values = list(act.actuator)
        if len(values) >= 4:
            state.update(
                motor1=values[0] / 1000.0,
                motor2=values[1] / 1000.0,
                motor3=values[2] / 1000.0,
                motor4=values[3] / 1000.0,
            )


async def subscribe_altitude(drone, state: TelemetryState) -> None:
    """altitude → baro_alt_meter (MAVSDK fallback，pymavlink HIGHRES_IMU 会覆盖)"""
    async for alt in drone.telemetry.altitude():
        state.update(baro_alt_meter=alt.altitude_local_m)


async def subscribe_flight_mode(drone, state: TelemetryState) -> None:
    """flight_mode → nav_state (PX4 导航状态编号)"""
    async for mode in drone.telemetry.flight_mode():
        nav = _FLIGHT_MODE_TO_NAV_STATE.get(mode, -1)
        state.update(nav_state=nav)
