"""自动飞行控制器 — 让 PX4 SITL 起飞并随机游走。

供 TelemetryProcessManager 作为子进程启动，确保 SITL 不是地面怠速。
连接 14540 端口（与 Collector 的 14550 不冲突）。

用法:
    python -m telemetry.flight_controller
    python -m telemetry.flight_controller --sim-time 120 --seed 42
    python -m telemetry.flight_controller --fault-type motor_failure --fault-delay 20 --fault-scale 0.1 --motor-index 2
"""

import argparse
import asyncio
import math
import random
import signal
import subprocess
import time

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

from uav_fault.telemetry.flight_status import decide_exit_code


# ── Gazebo Transport 故障注入工具 ──────────────────────────────


def gz_publish_int(topic: str, value: int) -> bool:
    """通过 gz topic 发布 Int32 消息。"""
    try:
        subprocess.run(
            ["gz", "topic", "-t", topic, "-m", "gz.msgs.Int32", "-p", f"data: {value}"],
            check=True,
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        print(f"[flight_ctrl] gz publish int 失败: {e}")
        return False


def gz_publish_float(topic: str, value: float) -> bool:
    """通过 gz topic 发布 Float 消息。"""
    try:
        subprocess.run(
            ["gz", "topic", "-t", topic, "-m", "gz.msgs.Float", "-p", f"data: {value}"],
            check=True,
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        print(f"[flight_ctrl] gz publish float 失败: {e}")
        return False


async def arm_with_retry(drone, max_retries: int = 10, interval: float = 2.0) -> bool:
    """尝试解锁，PX4 可能因为预检未通过而拒绝。"""
    for i in range(max_retries):
        try:
            await drone.action.arm()
            print("[flight_ctrl] 已解锁")
            return True
        except Exception as e:
            if i < max_retries - 1:
                print(
                    f"[flight_ctrl] 解锁失败 ({e})，{interval}s 后重试 ({i + 1}/{max_retries})"
                )
                await asyncio.sleep(interval)
            else:
                print(f"[flight_ctrl] 解锁最终失败: {e}")
    return False


async def run_flight(
    sim_time: float = 60.0,
    seed: int | None = None,
    fault_type: str = "none",
    fault_delay: float = 20.0,
    fault_scale: float = 0.0,
    motor_index: int = 1,
    gz_model_name: str = "x500_0",
    vibration_amp: float = 0.1,
    vibration_freq: float = 10.0,
    battery_drain: float = 20.0,
    flight_mode: str = "hover",
    inject_mode: str = "step",
    fault_duration: float = 0.0,
    curve: str = "linear",
) -> int:
    """完整飞行流程：起飞 → hover/随机游走 → (可选故障注入) → 降落。"""
    rng = random.Random(seed)

    drone = System(port=50051)
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("[flight_ctrl] 等待 PX4 连接 ...")
    start = time.time()
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[flight_ctrl] 已连接")
            break
        if time.time() - start > 60.0:
            print("[flight_ctrl] 连接超时")
            return

    # 等待 global/home ready；armable 仅作观测，不再作为硬门槛
    print("[flight_ctrl] 等待系统 ready (global/home) ...")
    stable_start = None
    async for health in drone.telemetry.health():
        ready = health.is_global_position_ok and health.is_home_position_ok
        if ready:
            if stable_start is None:
                stable_start = time.time()
            stable_elapsed = time.time() - stable_start
            print(
                f"[flight_ctrl] health ready, armable={health.is_armable}, stable={stable_elapsed:.1f}s"
            )
            if stable_elapsed >= 3.0:
                print("[flight_ctrl] 系统 ready，准备解锁")
                break
        else:
            stable_start = None
            print(
                "[flight_ctrl] waiting health: "
                f"global={health.is_global_position_ok} "
                f"home={health.is_home_position_ok} "
                f"armable={health.is_armable}"
            )

    # 电池故障预设（必须在起飞前设置，允许电量降至极低）
    if fault_type == "battery_failure":
        try:
            await drone.param.set_param_float("SIM_BAT_DRAIN", 60.0)
            await drone.param.set_param_float("SIM_BAT_MIN_PCT", 1.0)
            print("[flight_ctrl] 电池参数已预设（SIM_BAT_MIN_PCT=1%）")
        except Exception as e:
            print(f"[flight_ctrl] 电池参数预设失败: {e}")

    # 解锁
    if not await arm_with_retry(drone):
        return decide_exit_code(
            arm_ok=False, offboard_ok=False, mission_completed=False
        )

    # 起飞
    takeoff_alt = rng.uniform(5, 15)
    print(f"[flight_ctrl] 起飞到 {takeoff_alt:.1f}m")
    await drone.action.set_takeoff_altitude(takeoff_alt)
    await drone.action.takeoff()

    # 等待到达起飞高度（粗略检测）
    await asyncio.sleep(8)

    # 切换到 Offboard 模式，hover / 随机游走
    max_speed = rng.uniform(5, 12)
    if flight_mode == "hover":
        print("[flight_ctrl] 进入 offboard 定点悬停")
    else:
        print(f"[flight_ctrl] 进入 offboard 随机游走，最大速度 {max_speed:.1f}m/s")

    # 先设置一个初始 setpoint（offboard 要求先发一帧）
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"[flight_ctrl] offboard 启动失败: {e}，回退到 hover")
        await asyncio.sleep(sim_time)
        await drone.action.land()
        return decide_exit_code(arm_ok=True, offboard_ok=False, mission_completed=False)

    # 随机游走循环
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    flight_start = time.time()
    heading = rng.uniform(0, 360)
    segment_end = 0.0
    fault_injected = False
    fault_active = False
    ramp_start_time = 0.0
    last_ramp_update = 0.0
    cyclic_state = False
    fault_clear_time = fault_delay + fault_duration if fault_duration > 0 else None

    try:
        while not stop_event.is_set():
            elapsed = time.time() - flight_start
            if elapsed >= sim_time:
                break

            # ── 故障注入（一次性 step） ──
            if not fault_injected and fault_type != "none" and elapsed >= fault_delay:
                fault_injected = True
                if fault_type == "motor_failure":
                    topic_num = f"/model/{gz_model_name}/motor_failure/motor_number"
                    topic_eff = f"/model/{gz_model_name}/motor_failure/effectiveness"
                    if inject_mode == "step":
                        fault_active = True
                        gz_publish_float(topic_eff, fault_scale)
                        gz_publish_int(topic_num, motor_index)
                        print(
                            f"[flight_ctrl] !! 注入 motor_failure(step): motor={motor_index}, effectiveness={fault_scale}"
                        )
                    elif inject_mode == "continuous":
                        fault_active = True
                        ramp_start_time = elapsed
                        last_ramp_update = elapsed
                        gz_publish_float(topic_eff, 1.0)
                        gz_publish_int(topic_num, motor_index)
                        print(
                            f"[flight_ctrl] !! 开始 motor_failure(continuous): motor={motor_index}, target={fault_scale}, duration={fault_duration}s, curve={curve}"
                        )
                    elif inject_mode == "cyclic":
                        fault_active = True
                        gz_publish_int(topic_num, motor_index)
                        cyclic_state = True
                        gz_publish_float(topic_eff, fault_scale)
                        print(
                            f"[flight_ctrl] !! 开始 motor_failure(cyclic): motor={motor_index}, effectiveness={fault_scale}, duration={fault_duration}s"
                        )
                elif fault_type == "thrust_loss":
                    try:
                        await drone.param.set_param_float("MPC_THR_MAX", fault_scale)
                        val = await drone.param.get_param_float("MPC_THR_MAX")
                        if abs(val - fault_scale) > 0.01:
                            print(
                                f"[flight_ctrl] thrust_loss 参数验证失败: 实际={val}, 目标={fault_scale}"
                            )
                        else:
                            print(
                                f"[flight_ctrl] !! 注入 thrust_loss: MPC_THR_MAX={val}"
                            )
                    except Exception as e:
                        print(f"[flight_ctrl] thrust_loss 注入失败: {e}")
                elif fault_type == "prop_imbalance":
                    topic_motor = (
                        f"/model/{gz_model_name}/motor_failure/vibration/motor"
                    )
                    topic_amp = (
                        f"/model/{gz_model_name}/motor_failure/vibration/amplitude"
                    )
                    topic_freq = (
                        f"/model/{gz_model_name}/motor_failure/vibration/frequency"
                    )
                    gz_publish_float(topic_amp, vibration_amp)
                    gz_publish_float(topic_freq, vibration_freq)
                    gz_publish_int(topic_motor, motor_index)
                    print(
                        f"[flight_ctrl] !! 注入 prop_imbalance: motor={motor_index}, amp={vibration_amp}, freq={vibration_freq}Hz"
                    )
                elif fault_type == "battery_failure":
                    try:
                        await drone.param.set_param_float(
                            "SIM_BAT_DRAIN", battery_drain
                        )
                        val = await drone.param.get_param_float("SIM_BAT_DRAIN")
                        if abs(val - battery_drain) > 0.5:
                            print(
                                f"[flight_ctrl] battery_failure 参数验证失败: 实际={val}, 目标={battery_drain}"
                            )
                        else:
                            print(
                                f"[flight_ctrl] !! 注入 battery_failure: SIM_BAT_DRAIN={val}s"
                            )
                    except Exception as e:
                        print(f"[flight_ctrl] battery_failure 注入失败: {e}")

            if fault_type == "motor_failure" and fault_injected and fault_active:
                topic_eff = f"/model/{gz_model_name}/motor_failure/effectiveness"
                if inject_mode == "continuous":
                    if (
                        fault_duration > 0
                        and elapsed - ramp_start_time >= fault_duration
                    ):
                        gz_publish_float(topic_eff, fault_scale)
                    elif elapsed - last_ramp_update >= 0.5 and fault_duration > 0:
                        last_ramp_update = elapsed
                        progress = min(
                            1.0, (elapsed - ramp_start_time) / max(fault_duration, 1e-3)
                        )
                        if curve == "exp":
                            progress = progress**2
                        curr_eff = 1.0 - (1.0 - fault_scale) * progress
                        gz_publish_float(topic_eff, curr_eff)
                elif inject_mode == "cyclic":
                    period_s = float(curve) if curve not in ("linear", "exp") else 4.0
                    half_period = max(period_s / 2.0, 0.5)
                    phase = int((elapsed - fault_delay) / half_period)
                    desired_state = phase % 2 == 0
                    if desired_state != cyclic_state:
                        cyclic_state = desired_state
                        gz_publish_float(
                            topic_eff, fault_scale if cyclic_state else 1.0
                        )

                if fault_clear_time is not None and elapsed >= fault_clear_time:
                    topic_num = f"/model/{gz_model_name}/motor_failure/motor_number"
                    gz_publish_int(topic_num, 0)
                    gz_publish_float(topic_eff, 1.0)
                    fault_active = False
                    print("[flight_ctrl] motor_failure 持续时间结束，已自动恢复")

            if flight_mode == "hover":
                heading = 0.0
                vn = 0.0
                ve = 0.0
                vd = 0.0
            else:
                # 每 3-8 秒换一次方向
                if elapsed >= segment_end:
                    heading = rng.uniform(0, 360)
                    speed = rng.uniform(2, max_speed)
                    segment_end = elapsed + rng.uniform(3, 8)

                rad = math.radians(heading)
                vn = speed * math.cos(rad)
                ve = speed * math.sin(rad)
                # 小幅随机高度变化
                vd = rng.uniform(-1, 1)

            await drone.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, vd, heading))

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.2)
                break
            except asyncio.TimeoutError:
                pass

    finally:
        # ── 故障清理 ──
        if fault_injected:
            if fault_type == "motor_failure":
                topic_num = f"/model/{gz_model_name}/motor_failure/motor_number"
                topic_eff = f"/model/{gz_model_name}/motor_failure/effectiveness"
                gz_publish_int(topic_num, 0)
                gz_publish_float(topic_eff, 1.0)
                print("[flight_ctrl] 已复位电机故障（motor=0, effectiveness=1.0）")
            elif fault_type == "thrust_loss":
                try:
                    await drone.param.set_param_float("MPC_THR_MAX", 1.0)
                    print("[flight_ctrl] 已复位 MPC_THR_MAX=1.0")
                except Exception:
                    pass
            elif fault_type == "prop_imbalance":
                topic_amp = f"/model/{gz_model_name}/motor_failure/vibration/amplitude"
                gz_publish_float(topic_amp, 0.0)
                print("[flight_ctrl] 已关闭螺旋桨振动（amplitude=0.0）")
            elif fault_type == "battery_failure":
                try:
                    await drone.param.set_param_float("SIM_BAT_DRAIN", 60.0)
                    print("[flight_ctrl] 已复位电池参数（SIM_BAT_DRAIN=60.0）")
                except Exception:
                    pass

    # 停止 offboard，降落
    print("[flight_ctrl] 随机游走结束，准备降落")
    try:
        await drone.offboard.stop()
    except OffboardError:
        pass

    await drone.action.land()
    print("[flight_ctrl] 降落指令已发送，等待着陆 ...")

    # 等待 disarm（最多 30s）
    try:
        async for armed in drone.telemetry.armed():
            if not armed:
                print("[flight_ctrl] 已降落并上锁")
                break
    except Exception:
        pass

    print("[flight_ctrl] 飞行完成")
    return decide_exit_code(arm_ok=True, offboard_ok=True, mission_completed=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PX4 SITL 自动飞行控制器")
    parser.add_argument("--sim-time", type=float, default=60.0, help="飞行时长 (秒)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument(
        "--fault-type",
        choices=[
            "none",
            "motor_failure",
            "thrust_loss",
            "prop_imbalance",
            "battery_failure",
        ],
        default="none",
        help="故障类型",
    )
    parser.add_argument(
        "--fault-delay", type=float, default=20.0, help="起飞后多少秒注入故障"
    )
    parser.add_argument(
        "--fault-scale",
        type=float,
        default=0.0,
        help="故障强度 (motor_failure: 0.0-0.4, thrust_loss: 0.4-0.65)",
    )
    parser.add_argument(
        "--motor-index",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="失效电机编号 (motor_failure/prop_imbalance)",
    )
    parser.add_argument("--gz-model-name", default="x500_0", help="Gazebo 模型名")
    parser.add_argument(
        "--vibration-amp", type=float, default=0.1, help="振动振幅 (prop_imbalance)"
    )
    parser.add_argument(
        "--vibration-freq",
        type=float,
        default=10.0,
        help="振动频率 Hz (prop_imbalance)",
    )
    parser.add_argument(
        "--battery-drain",
        type=float,
        default=20.0,
        help="电池放电时间秒 (battery_failure, 正常60s)",
    )
    parser.add_argument(
        "--flight-mode", choices=["hover", "random_walk"], default="hover"
    )
    parser.add_argument(
        "--inject-mode", choices=["step", "continuous", "cyclic"], default="step"
    )
    parser.add_argument("--fault-duration", type=float, default=0.0)
    parser.add_argument("--curve", type=str, default="linear")
    args = parser.parse_args()

    exit_code = asyncio.run(
        run_flight(
            sim_time=args.sim_time,
            seed=args.seed,
            fault_type=args.fault_type,
            fault_delay=args.fault_delay,
            fault_scale=args.fault_scale,
            motor_index=args.motor_index,
            gz_model_name=args.gz_model_name,
            vibration_amp=args.vibration_amp,
            vibration_freq=args.vibration_freq,
            battery_drain=args.battery_drain,
            flight_mode=args.flight_mode,
            inject_mode=args.inject_mode,
            fault_duration=args.fault_duration,
            curve=args.curve,
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
