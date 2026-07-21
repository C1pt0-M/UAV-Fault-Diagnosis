"""线程安全的共享遥测状态，支持快照读取和派生列计算。"""

import math
import threading
import time


class TelemetryState:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[str, float] = {
            "ax": 0.0,
            "ay": 0.0,
            "az": 0.0,
            "armed": 0.0,
            "pymav_heartbeat_seen": 0.0,
            "pymav_highres_imu_seen": 0.0,
            "pymav_attitude_target_seen": 0.0,
            "pymav_actuator_control_seen": 0.0,
        }
        self._start_time = time.time()
        # 用于速度差分计算融合加速度
        self._prev_vx = 0.0
        self._prev_vy = 0.0
        self._prev_vz = 0.0
        self._prev_vel_time: float | None = None

    def update(self, **kwargs: float) -> None:
        with self._lock:
            self._data.update(kwargs)

    def snapshot(self) -> dict[str, float]:
        """读取当前状态快照，计算派生列，返回完整行。

        注意：此方法应仅由 emit_loop 单线程调用，差分状态在锁内更新。
        """
        with self._lock:
            row = dict(self._data)

            now = time.time()
            row["time_s"] = round(now - self._start_time, 4)

            # 派生：合成速度
            vx = row.get("vx", 0.0)
            vy = row.get("vy", 0.0)
            vz = row.get("vz", 0.0)
            row["speed_xy_m_s"] = math.sqrt(vx * vx + vy * vy)
            row["speed_3d_m_s"] = math.sqrt(vx * vx + vy * vy + vz * vz)

            # 派生：融合加速度 (对 NED 速度求时间差分)
            if self._prev_vel_time is not None:
                dt = now - self._prev_vel_time
                if dt > 0.005:
                    row["ax"] = (vx - self._prev_vx) / dt
                    row["ay"] = (vy - self._prev_vy) / dt
                    row["az"] = (vz - self._prev_vz) / dt
            self._prev_vx, self._prev_vy, self._prev_vz = vx, vy, vz
            self._prev_vel_time = now

            # 派生：角速率跟踪误差 (setpoint - actual)
            rs_roll = row.get("rates_sp_roll", 0.0)
            rs_pitch = row.get("rates_sp_pitch", 0.0)
            rs_yaw = row.get("rates_sp_yaw", 0.0)
            row["rate_err_roll"] = rs_roll - row.get("rollspeed", 0.0)
            row["rate_err_pitch"] = rs_pitch - row.get("pitchspeed", 0.0)
            row["rate_err_yaw"] = rs_yaw - row.get("yawspeed", 0.0)

            # 派生：电机交叉统计
            m1 = row.get("motor1", 0.0)
            m2 = row.get("motor2", 0.0)
            m3 = row.get("motor3", 0.0)
            m4 = row.get("motor4", 0.0)
            motors = [m1, m2, m3, m4]
            motor_mean = sum(motors) / 4.0
            motor_var = sum((m - motor_mean) ** 2 for m in motors) / 4.0
            row["motor_cross_mean"] = motor_mean
            row["motor_cross_std"] = math.sqrt(motor_var)

            # 派生：加速度范数
            ax_body = row.get("accel_x", 0.0)
            ay_body = row.get("accel_y", 0.0)
            az_body = row.get("accel_z", 0.0)
            row["acc_norm"] = math.sqrt(ax_body**2 + ay_body**2 + az_body**2)

            # 派生：电机努力度 (motor_mean / |accel_z|)
            az_abs = abs(az_body)
            row["motor_effort"] = motor_mean / max(az_abs, 0.1)

        return row
