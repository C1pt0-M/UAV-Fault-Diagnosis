"""无人机侧遥测采集主入口。

启动 MAVSDK 订阅 + pymavlink 补充线程，50Hz 快照并通过 UDP 转发。

用法:
    python -m uav_fault.telemetry.collector
    python -m uav_fault.telemetry.collector --emit-rate 100
    python -m uav_fault.telemetry.collector --no-pymav
"""

import argparse
import asyncio
import signal
import threading
import time

from mavsdk import System

from uav_fault.telemetry.config import (
    EMIT_RATE_HZ,
    MAVSDK_ADDRESS,
    MAVSDK_GRPC_PORT,
    PYMAV_ADDRESS,
    UDP_FORWARD_HOST,
    UDP_FORWARD_PORT,
)
from uav_fault.telemetry.forwarder import UDPForwarder
from uav_fault.telemetry.pymav_supplement import run_pymav_listener
from uav_fault.telemetry.state import TelemetryState
from uav_fault.telemetry.subscriptions import (
    subscribe_actuator,
    subscribe_altitude,
    subscribe_angular_velocity,
    subscribe_attitude,
    subscribe_battery,
    subscribe_flight_mode,
    subscribe_imu,
    subscribe_position_velocity,
)


async def _guarded_subscribe(coro, name: str) -> None:
    """包装订阅协程，捕获异常并记录，避免静默停止。"""
    try:
        await coro
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"[collector] 订阅 {name} 异常退出: {e}")


async def emit_loop(state: TelemetryState, forwarder: UDPForwarder, rate_hz: int) -> None:
    interval = 1.0 / rate_hz
    print(f"[collector] 开始转发，频率 {rate_hz}Hz → {forwarder._target}")
    while True:
        row = state.snapshot()
        forwarder.send(row)
        await asyncio.sleep(interval)


async def run(args: argparse.Namespace) -> None:
    state = TelemetryState()
    forwarder = UDPForwarder(args.host, args.port)

    # 连接 MAVSDK (自动启动 mavsdk_server，使用独立 gRPC 端口)
    drone = System(port=args.grpc_port)
    await drone.connect(system_address=args.mavsdk_address)

    print(f"[collector] 正在连接 PX4 ({args.mavsdk_address}) ...")
    start = time.time()
    async for conn_state in drone.core.connection_state():
        if conn_state.is_connected:
            print("[collector] 已连接")
            break
        if time.time() - start > 30.0:
            print("[collector] 连接超时")
            return

    # 设置遥测发布频率
    rate = float(args.emit_rate)
    await drone.telemetry.set_rate_attitude_euler(rate)
    await drone.telemetry.set_rate_position_velocity_ned(rate)
    await drone.telemetry.set_rate_imu(rate)
    await drone.telemetry.set_rate_battery(5.0)
    await drone.telemetry.set_rate_actuator_output_status(rate)
    await drone.telemetry.set_rate_altitude(10.0)

    # pymavlink 补充线程（传入 stop_event 以便优雅退出）
    pymav_stop = threading.Event()
    pymav_thread = None
    if not args.no_pymav:
        pymav_thread = threading.Thread(
            target=run_pymav_listener,
            args=(state, args.pymav_address, pymav_stop),
            daemon=True,
        )
        pymav_thread.start()

    # 启动所有订阅协程（带异常保护）+ 转发循环
    tasks = [
        asyncio.create_task(_guarded_subscribe(subscribe_attitude(drone, state), "attitude")),
        asyncio.create_task(_guarded_subscribe(subscribe_angular_velocity(drone, state), "angular_velocity")),
        asyncio.create_task(_guarded_subscribe(subscribe_position_velocity(drone, state), "position_velocity")),
        asyncio.create_task(_guarded_subscribe(subscribe_imu(drone, state), "imu")),
        asyncio.create_task(_guarded_subscribe(subscribe_battery(drone, state), "battery")),
        asyncio.create_task(_guarded_subscribe(subscribe_actuator(drone, state), "actuator")),
        asyncio.create_task(_guarded_subscribe(subscribe_altitude(drone, state), "altitude")),
        asyncio.create_task(_guarded_subscribe(subscribe_flight_mode(drone, state), "flight_mode")),
        asyncio.create_task(emit_loop(state, forwarder, args.emit_rate)),
    ]

    # 注册信号处理，优雅退出
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        # 等待停止信号或任务异常
        done, _ = await asyncio.wait(
            [asyncio.create_task(stop_event.wait())] + tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        # 打印异常退出的任务
        for t in done:
            if t.exception() is not None:
                print(f"[collector] 任务异常退出: {t.exception()}")
    except asyncio.CancelledError:
        pass
    finally:
        # 取消所有订阅任务
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        forwarder.close()
        # 停止 pymavlink 线程
        pymav_stop.set()
        if pymav_thread is not None:
            pymav_thread.join(timeout=3.0)
        print("[collector] 已停止")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无人机实时遥测采集服务")
    parser.add_argument("--mavsdk-address", default=MAVSDK_ADDRESS)
    parser.add_argument("--grpc-port", type=int, default=MAVSDK_GRPC_PORT)
    parser.add_argument("--pymav-address", default=PYMAV_ADDRESS)
    parser.add_argument("--host", default=UDP_FORWARD_HOST, help="UDP 转发目标地址")
    parser.add_argument("--port", type=int, default=UDP_FORWARD_PORT, help="UDP 转发目标端口")
    parser.add_argument("--emit-rate", type=int, default=EMIT_RATE_HZ, help="发送频率 (Hz, >= 1)")
    parser.add_argument("--no-pymav", action="store_true", help="禁用 pymavlink 补充")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.emit_rate < 1:
        print("[collector] emit-rate 必须 >= 1，已设为 1")
        args.emit_rate = 1
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
