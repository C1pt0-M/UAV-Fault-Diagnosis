"""pymavlink 补充线程，获取 MAVSDK 无法精确提供的字段。

监听 PX4 GCS 端口，提取：
- HIGHRES_IMU.pressure_alt   → baro_alt_meter
- BATTERY_STATUS.charge_state → battery_warning  (0=正常,1=低,2=严重,3=紧急)
- SYS_STATUS.onboard_control_sensors_health → sys_status_health (位掩码)
- HEARTBEAT.system_status    → system_status     (MAV_STATE 枚举)
- ATTITUDE_TARGET            → rates_sp_roll/pitch/yaw, thrust_z
- VIBRATION                  → accel_vibration_metric, gyro_vibration_metric
- ESC_STATUS                 → esc_rpm_1..4
"""

from pymavlink import mavutil

from .config import PYMAV_ADDRESS
from .state import TelemetryState

# MAVLink charge_state → 简化 warning 等级
# 参考 MAV_BATTERY_CHARGE_STATE 枚举
_CHARGE_STATE_TO_WARNING = {
    0: 0,  # UNDEFINED → 正常
    1: 0,  # OK
    2: 1,  # LOW
    3: 2,  # CRITICAL
    4: 3,  # EMERGENCY
    5: 3,  # FAILED
    6: 0,  # UNHEALTHY (不一定是电量问题)
    7: 0,  # CHARGING
}

# 需要接收的 MAVLink 消息类型
_MSG_TYPES = [
    "HIGHRES_IMU",
    "BATTERY_STATUS",
    "SYS_STATUS",
    "HEARTBEAT",
    "ATTITUDE_TARGET",
    "ACTUATOR_CONTROL_TARGET",
    "VIBRATION",
    "ESC_STATUS",
]


def _request_stream(conn, msg_id: int, interval_us: int) -> None:
    """通过 SET_MESSAGE_INTERVAL 请求指定消息的发送速率。"""
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,  # confirmation
        msg_id,  # param1: message ID
        interval_us,  # param2: interval in microseconds
        0,
        0,
        0,
        0,
        0,
    )


def run_pymav_listener(
    state: TelemetryState,
    address: str = PYMAV_ADDRESS,
    stop_event: "threading.Event | None" = None,
) -> None:
    """阻塞函数，需在独立线程中运行。可通过 stop_event 通知退出。"""
    import threading

    if stop_event is None:
        stop_event = threading.Event()

    print(f"[pymavlink] 正在连接 {address} ...")
    try:
        conn = mavutil.mavlink_connection(address)
        print(f"[pymavlink] 连接已建立，等待心跳...")
        conn.wait_heartbeat(timeout=30)
        print(
            f"[pymavlink] 已连接 (sysid={conn.target_system} compid={conn.target_component})，开始接收补充数据"
        )
    except Exception as e:
        print(f"[pymavlink] 连接失败: {e}")
        return

    # 请求额外消息流：提升 ATTITUDE_TARGET / ACTUATOR_CONTROL_TARGET / VIBRATION / ESC_STATUS 的发送速率
    # ATTITUDE_TARGET (msg_id=83): 默认 2Hz → 提升到 50Hz (20000us)
    _request_stream(conn, 83, 20_000)
    # ACTUATOR_CONTROL_TARGET (msg_id=140): 提升到 50Hz (20000us)
    _request_stream(conn, 140, 20_000)
    # VIBRATION (msg_id=241): 默认 0.1Hz → 提升到 10Hz (100000us)
    _request_stream(conn, 241, 100_000)
    # ESC_STATUS (msg_id=291): 默认 1Hz → 提升到 10Hz (100000us)
    _request_stream(conn, 291, 100_000)
    print(
        "[pymavlink] 已请求 ATTITUDE_TARGET(50Hz), ACTUATOR_CONTROL_TARGET(50Hz), VIBRATION(10Hz), ESC_STATUS(10Hz)"
    )

    while not stop_event.is_set():
        msg = conn.recv_match(
            type=_MSG_TYPES,
            blocking=True,
            timeout=1.0,
        )
        if msg is None:
            continue

        msg_type = msg.get_type()

        if msg_type == "HIGHRES_IMU":
            state.update(
                baro_alt_meter=msg.pressure_alt,
                baro_pressure_pa=float(msg.abs_pressure) * 100.0,
                baro_temp=msg.temperature,
                mag_x=msg.xmag,
                mag_y=msg.ymag,
                mag_z=msg.zmag,
                pymav_highres_imu_seen=1.0,
            )

        elif msg_type == "BATTERY_STATUS":
            charge_state = getattr(msg, "charge_state", 0)
            warning = _CHARGE_STATE_TO_WARNING.get(charge_state, 0)
            state.update(battery_warning=warning)

        elif msg_type == "SYS_STATUS":
            state.update(sys_status_health=msg.onboard_control_sensors_health)

        elif msg_type == "HEARTBEAT":
            # 过滤 GCS 的心跳，只取飞控的 (component_id=1 即 autopilot)
            if msg.get_srcComponent() == 1:
                state.update(system_status=msg.system_status)

        elif msg_type == "ATTITUDE_TARGET":
            state.update(
                rates_sp_roll=msg.body_roll_rate,  # rad/s
                rates_sp_pitch=msg.body_pitch_rate,  # rad/s
                rates_sp_yaw=msg.body_yaw_rate,  # rad/s
                thrust_z=msg.thrust,  # 0..1 归一化推力
                pymav_attitude_target_seen=1.0,
            )

        elif msg_type == "ACTUATOR_CONTROL_TARGET":
            if getattr(msg, "group_mlx", None) == 0:
                controls = list(getattr(msg, "controls", []) or [])
                if len(controls) >= 4:
                    state.update(
                        actuator_ctrl_1=float(controls[0]),
                        actuator_ctrl_2=float(controls[1]),
                        actuator_ctrl_3=float(controls[2]),
                        actuator_ctrl_4=float(controls[3]),
                        pymav_actuator_control_seen=1.0,
                    )

        elif msg_type == "VIBRATION":
            # PX4 复用了 VIBRATION 消息的字段:
            # vibration_z = accel_vibration_metric
            # vibration_y = gyro_vibration_metric
            state.update(
                accel_vibration_metric=msg.vibration_z,
                gyro_vibration_metric=msg.vibration_y,
            )

        elif msg_type == "ESC_STATUS":
            # ESC_STATUS 按 motor function 索引，最多 4 路
            rpm = getattr(msg, "rpm", None)
            if rpm and len(rpm) >= 4:
                state.update(
                    esc_rpm_1=float(rpm[0]),
                    esc_rpm_2=float(rpm[1]),
                    esc_rpm_3=float(rpm[2]),
                    esc_rpm_4=float(rpm[3]),
                )

    conn.close()
    print("[pymavlink] 已停止")
