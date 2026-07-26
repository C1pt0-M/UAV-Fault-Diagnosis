"""实时遥测采集服务 — 全局配置"""

# ── MAVSDK 连接 ──
MAVSDK_ADDRESS = "udpin://0.0.0.0:14550"  # PX4 GCS 链路远程端口
MAVSDK_GRPC_PORT = 50052  # 避免与注入脚本的默认 50051 冲突

# ── pymavlink 连接 (补充字段) ──
PYMAV_ADDRESS = "udpin:0.0.0.0:14030"  # PX4 onboard payload 链路远程端口 (SITL 下空闲)

# ── UDP 转发 ──
UDP_FORWARD_HOST = "127.0.0.1"
UDP_FORWARD_PORT = 15550

# ── 采样率 ──
EMIT_RATE_HZ = 50

# ── ML 特征列 ──
# 原始 28 列 + 新增 MAVLink 通道 + 派生通道
FEATURE_COLUMNS = [
    # 电机输出 (MAVSDK actuator_output_status)
    "motor1", "motor2", "motor3", "motor4",
    # 姿态 (MAVSDK attitude_euler)
    "roll", "pitch", "yaw",
    # 角速率 (MAVSDK attitude_angular_velocity_body)
    "rollspeed", "pitchspeed", "yawspeed",
    # 体轴加速度 (MAVSDK imu)
    "accel_x", "accel_y", "accel_z",
    # 体轴角速率 (MAVSDK imu)
    "gyro_x", "gyro_y", "gyro_z",
    # 融合加速度 (NED 速度差分, state.py 派生)
    "ax", "ay", "az",
    # NED 速度 (MAVSDK position_velocity_ned)
    "vx", "vy", "vz",
    # 合成速度 (state.py 派生)
    "speed_xy_m_s", "speed_3d_m_s",
    # 位置 (MAVSDK position_velocity_ned)
    "pos_z",
    # 气压高度/温度/压力 (pymavlink HIGHRES_IMU)
    "baro_alt_meter", "baro_temp", "baro_pressure_pa",
    # 磁力计 (pymavlink HIGHRES_IMU)
    "mag_x", "mag_y", "mag_z",
    # 电池 (MAVSDK battery)
    "battery_voltage_v", "battery_remaining",
    # ── 以下为新增通道 ──
    # 执行器控制目标 (pymavlink ACTUATOR_CONTROL_TARGET)
    "actuator_ctrl_1", "actuator_ctrl_2", "actuator_ctrl_3", "actuator_ctrl_4",
    # 角速率设定值 + 推力指令 (pymavlink ATTITUDE_TARGET)
    "rates_sp_roll", "rates_sp_pitch", "rates_sp_yaw",
    "thrust_z",
    # 振动指标 (pymavlink VIBRATION)
    "accel_vibration_metric", "gyro_vibration_metric",
    # ESC 转速 (pymavlink ESC_STATUS, SITL 可能为 0)
    "esc_rpm_1", "esc_rpm_2", "esc_rpm_3", "esc_rpm_4",
    # 角速率跟踪误差 (state.py 派生: rates_sp - actual)
    "rate_err_roll", "rate_err_pitch", "rate_err_yaw",
    # 电机交叉统计 (state.py 派生)
    "motor_cross_mean", "motor_cross_std",
    # 加速度范数 (state.py 派生)
    "acc_norm",
    # 电机努力度 (state.py 派生: motor_mean / |accel_z|)
    "motor_effort",
]

# ── 辅助列 (实时故障检测/告警) ──
# nav_state:          MAVSDK flight_mode 映射的 PX4 导航状态编号
# battery_warning:    pymavlink BATTERY_STATUS.charge_state (0=正常, 1=低, 2=严重, 3=紧急)
# sys_status_health:  pymavlink SYS_STATUS.onboard_control_sensors_health 位掩码
# system_status:      pymavlink HEARTBEAT.system_status (MAV_STATE 枚举)
# armed:              pymavlink HEARTBEAT base_mode 的 armed 标志
AUX_COLUMNS = [
    "actuator_out_1",
    "actuator_out_2",
    "actuator_out_3",
    "actuator_out_4",
    "actuator_source",
    "nav_state",
    "battery_warning",
    "sys_status_health",
    "system_status",
    "armed",
    "pymav_heartbeat_seen",
    "pymav_highres_imu_seen",
    "pymav_attitude_target_seen",
    "pymav_actuator_control_seen",
    "mec_enabled",
    "mec_inference_location",
    "mec_rx_ms",
    "mec_tx_ms",
    "mec_fault_prob",
    "mec_label",
    "mec_confidence",
    "mec_inference_ms",
    "mec_avg_inference_ms",
    "mec_p95_inference_ms",
    "mec_alert_active",
    "mec_event_seq",
    "mec_inference_count",
    "mec_frames_forwarded",
    "mec_inference_queue_drops",
    "mec_contract_status",
    "mec_model_status",
]
