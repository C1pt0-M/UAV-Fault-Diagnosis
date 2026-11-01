"""实时故障检测服务 — 集成遥测接收与在线检测。

在接收 UDP 遥测数据的同时，实时运行 CNN1D 故障检测模型。

用法:
    python -m uav_fault.online.realtime_detection_service --model assets/models/online_detector_CNN
    python -m uav_fault.online.realtime_detection_service --model assets/models/online_detector_CNN --port 15550
"""

import argparse
import sys
from collections import deque
from pathlib import Path

from uav_fault.telemetry.receiver import TelemetryReceiver
from uav_fault.telemetry.schema_adapter import (
    assess_canonical_input_contract,
    canonicalize_rfly_frame,
)
from uav_fault.online.detection_policy import OnlineDetectionPolicy, input_contract_check_due
from uav_fault.online.event_aggregator import event_timing
from uav_fault.online.monitoring_session import (
    OnlineDetectionRunner,
    OnlineMonitoringSession,
)


class RealtimeDetectionService:
    """实时故障检测服务，集成遥测接收与 CNN1D 检测。"""

    def __init__(
        self,
        model_dir: str,
        port: int = 15550,
        csv_path: str | None = None,
        window_size: int = 150,
        device: str = "cpu",
        policy: OnlineDetectionPolicy | None = None,
    ):
        self.policy = policy or OnlineDetectionPolicy()
        self.runner = OnlineDetectionRunner(
            model_dir,
            device=device,
            window_size=window_size,
            policy=self.policy,
        )
        self.detector = self.runner.detector
        self.aggregator = self.runner.aggregator
        self.monitoring_session = OnlineMonitoringSession()
        self.monitoring_session.start_detection(
            self.runner,
            model_summary={"path": str(model_dir)},
        )
        self.canonical_window: deque[dict] = deque(maxlen=self.detector.window_size)
        self.input_contract_status: dict | None = None

        if csv_path is None:
            import time
            csv_path = f"telemetry_realtime_{int(time.time())}.csv"

        self.receiver = TelemetryReceiver(
            port=port,
            csv_path=csv_path,
            window_size=window_size,
            auto_stop=5.0,
            recv_timeout=30.0,
        )

        self.frame_count = 0
        self.detection_count = 0
        self.fault_events = []

        # 注册回调
        self.receiver.callbacks.append(self._on_frame)

        print(f"[realtime] 故障检测服务已初始化")
        print(f"  模型: {model_dir}")
        print(f"  窗口: {window_size} 帧")
        print(f"  端口: {port}")
        print(f"  CSV: {csv_path}")
        print(f"  检测策略: {self.policy.to_dict()}")
        print("  输入检查: 窗口就绪后每帧检查（独立于推理步长）")

    def _on_frame(self, frame: dict, window: deque) -> None:
        """每帧回调：当窗口满时运行检测。"""
        self.frame_count += 1
        canonical, coverage = canonicalize_rfly_frame(frame)
        self.canonical_window.append(canonical)

        if not input_contract_check_due(
            frame_idx=self.frame_count, window_length=len(self.canonical_window),
            min_window_size=self.detector.window_size, every_frames=1,
            has_previous=self.input_contract_status is not None,
        ):
            return

        self.input_contract_status = assess_canonical_input_contract(
            list(self.canonical_window),
            coverage=coverage,
            actuator_source=canonical.get("actuator_source")
            or frame.get("actuator_source"),
            window_size=self.detector.window_size,
        )
        if self.input_contract_status.get("detection_blocked"):
            return
        if not self.policy.should_infer(
            frame_idx=self.frame_count,
            window_length=len(self.canonical_window),
            window_size=self.detector.window_size,
        ):
            return

        update = self.monitoring_session.ingest(
            frame=canonical,
            window=list(self.canonical_window),
            frame_idx=self.frame_count,
            generation=self.monitoring_session.generation,
        )
        result = update.result
        if result is None or update.stale:
            return
        self.detection_count += 1
        fault_prob = result["fault_prob"]
        new_events = update.new_events

        # 打印新检出的故障事件
        for event in new_events:
            self.fault_events.append(event)
            print(f"\n[realtime] ⚠️  故障事件检出:")
            print(f"  时间: {event.start_t:.2f}s - {event.end_t:.2f}s (持续 {event.end_t - event.start_t:.2f}s)")
            print(f"  最大概率: {event.max_fault_prob:.3f}")
            print(f"  平均概率: {event.mean_fault_prob:.3f}")
            print(f"  数据点数: {event.n_points}")
            print(f"  时刻记录: {event_timing(event)}")

        # 每 100 次检测打印一次状态
        if self.detection_count % 100 == 0:
            print(f"[realtime] 已检测 {self.detection_count} 次 | 当前 fault_prob={fault_prob:.3f} | 事件数={len(self.fault_events)}")

    def start(self) -> None:
        """启动接收服务（阻塞）。"""
        print(f"[realtime] 开始接收遥测数据并实时检测 ...")
        try:
            self.receiver.start()
        finally:
            final_events = self.monitoring_session.stop_detection().new_events
            for event in final_events:
                self.fault_events.append(event)
                print(f"\n[realtime] ⚠️  故障事件检出 (flush):")
                print(f"  时间: {event.start_t:.2f}s - {event.end_t:.2f}s")
                print(f"  最大概率: {event.max_fault_prob:.3f}")
                print(f"  时刻记录: {event_timing(event)}")

            print(f"\n[realtime] 检测完成")
            print(f"  总帧数: {self.frame_count}")
            print(f"  检测次数: {self.detection_count}")
            print(f"  故障事件: {len(self.fault_events)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时故障检测服务")
    parser.add_argument("--model", required=True, help="模型目录路径")
    parser.add_argument("--port", type=int, default=15550, help="UDP 监听端口")
    parser.add_argument("--csv", default=None, help="CSV 输出路径（默认自动生成）")
    parser.add_argument("--window", type=int, default=150, help="滑动窗口大小")
    parser.add_argument("--device", default="cpu", help="推理设备 (cpu/cuda)")
    parser.add_argument("--every", type=int, default=1, help="每 N 帧推理一次，默认保持每帧推理")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 验证模型目录
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型目录不存在: {args.model}")
        sys.exit(1)

    service = RealtimeDetectionService(
        model_dir=args.model,
        port=args.port,
        csv_path=args.csv,
        window_size=args.window,
        device=args.device,
        policy=OnlineDetectionPolicy(inference_every=args.every),
    )
    service.start()


if __name__ == "__main__":
    main()
