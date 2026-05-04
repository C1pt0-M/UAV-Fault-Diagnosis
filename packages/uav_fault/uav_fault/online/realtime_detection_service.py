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
from uav_fault.online.fault_detector import FaultDetector
from uav_fault.online.event_aggregator import FaultEventAggregator


class RealtimeDetectionService:
    """实时故障检测服务，集成遥测接收与 CNN1D 检测。"""

    def __init__(
        self,
        model_dir: str,
        port: int = 15550,
        csv_path: str | None = None,
        window_size: int = 150,
        device: str = "cpu",
    ):
        self.detector = FaultDetector(model_dir=model_dir, device=device, window_size=window_size)
        self.aggregator = FaultEventAggregator(
            onset_th=0.6, clear_th=0.4, k_on=3, k_off=3,
            min_event_s=0.5, merge_gap_s=0.5,
        )

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

    def _on_frame(self, frame: dict, window: deque) -> None:
        """每帧回调：当窗口满时运行检测。"""
        self.frame_count += 1

        if len(window) < self.detector.window_size:
            return

        # 运行检测
        result = self.detector.predict(list(window))
        if result is None:
            return

        self.detection_count += 1
        t = frame.get("time_s", self.frame_count / 50.0)
        fault_prob = result["fault_prob"]

        # 聚合故障事件
        new_events = self.aggregator.update(t=t, fault_prob=fault_prob, frame_idx=self.frame_count)

        # 打印新检出的故障事件
        for event in new_events:
            self.fault_events.append(event)
            print(f"\n[realtime] ⚠️  故障事件检出:")
            print(f"  时间: {event.start_t:.2f}s - {event.end_t:.2f}s (持续 {event.end_t - event.start_t:.2f}s)")
            print(f"  最大概率: {event.max_fault_prob:.3f}")
            print(f"  平均概率: {event.mean_fault_prob:.3f}")
            print(f"  数据点数: {event.n_points}")

        # 每 100 次检测打印一次状态
        if self.detection_count % 100 == 0:
            print(f"[realtime] 已检测 {self.detection_count} 次 | 当前 fault_prob={fault_prob:.3f} | 事件数={len(self.fault_events)}")

    def start(self) -> None:
        """启动接收服务（阻塞）。"""
        print(f"[realtime] 开始接收遥测数据并实时检测 ...")
        try:
            self.receiver.start()
        finally:
            # 接收结束后，flush 聚合器
            final_events = self.aggregator.flush()
            for event in final_events:
                self.fault_events.append(event)
                print(f"\n[realtime] ⚠️  故障事件检出 (flush):")
                print(f"  时间: {event.start_t:.2f}s - {event.end_t:.2f}s")
                print(f"  最大概率: {event.max_fault_prob:.3f}")

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
    )
    service.start()


if __name__ == "__main__":
    main()
