from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

from uav_fault.online.event_aggregator import FaultEventAggregator
from uav_fault.online.realtime_feature_watcher import RealtimeFeatureWatcher
from uav_fault.online.risk_evaluator import RiskEvaluator
from uav_fault.telemetry.receiver import TelemetryReceiver


class RealtimeFeatureAlertEngine:
    def __init__(
        self,
        *,
        warmup_frames: int = 150,
        baseline_window: int = 600,
    ) -> None:
        self.watcher = RealtimeFeatureWatcher(
            warmup_frames=warmup_frames,
            baseline_window=baseline_window,
        )
        self.risk_evaluator = RiskEvaluator()
        self.risk_evaluator.watch_th = 0.12
        self.risk_evaluator.warning_th = 0.24
        self.risk_evaluator.critical_th = 0.45
        self.aggregator = FaultEventAggregator(
            onset_th=0.55,
            clear_th=0.35,
            k_on=3,
            k_off=3,
            min_event_s=0.4,
            merge_gap_s=0.5,
        )
        self.events = []
        self.last_stage = "normal"

    def update(self, row: dict) -> dict:
        watch = self.watcher.update(row)
        risk = self.risk_evaluator.update(fault_prob=watch["risk_score"])
        event_time = float(
            row.get("recv_time_s")
            or row.get("time_s")
            or 0.0
        )
        new_events = self.aggregator.update(
            t=event_time,
            fault_prob=watch["risk_score"],
        )
        self.events.extend(new_events)
        result = {
            **watch,
            **risk,
            "event_count": len(self.events),
            "new_event_count": len(new_events),
        }
        self.last_stage = result["warning_stage"]
        return result


class RealtimeFeatureWatchService:
    def __init__(
        self,
        *,
        port: int = 15550,
        raw_csv_path: str | None = None,
        score_csv_path: str | None = None,
        window_size: int = 150,
        warmup_frames: int = 150,
        baseline_window: int = 600,
    ) -> None:
        if raw_csv_path is None:
            raw_csv_path = f"telemetry_feature_watch_{int(time.time())}.csv"
        if score_csv_path is None:
            score_csv_path = f"feature_watch_scores_{int(time.time())}.csv"

        self.engine = RealtimeFeatureAlertEngine(
            warmup_frames=warmup_frames,
            baseline_window=baseline_window,
        )
        self.receiver = TelemetryReceiver(
            port=port,
            csv_path=raw_csv_path,
            window_size=window_size,
            auto_stop=5.0,
            recv_timeout=30.0,
            trim_on_stop=False,
        )
        self.receiver.callbacks.append(self._on_frame)
        self.score_csv_path = Path(score_csv_path)
        self._score_file = None
        self._score_writer = None
        self.frame_count = 0

        print("[feature-watch] 轻量实时告警服务已初始化")
        print(f"  UDP 端口: {port}")
        print(f"  原始 CSV: {raw_csv_path}")
        print(f"  评分 CSV: {score_csv_path}")

    def _ensure_writer(self, row: dict) -> None:
        if self._score_writer is not None:
            return
        self._score_file = self.score_csv_path.open("w", newline="", encoding="utf-8")
        fieldnames = list(row.keys())
        self._score_writer = csv.DictWriter(self._score_file, fieldnames=fieldnames)
        self._score_writer.writeheader()

    def _on_frame(self, frame: dict, window: deque) -> None:
        self.frame_count += 1
        result = self.engine.update(frame)
        output = {
            "frame_idx": self.frame_count,
            "time_s": frame.get("time_s", 0.0),
            "recv_time_s": frame.get("recv_time_s", 0.0),
            "fault_prob": result["fault_prob"],
            "risk_score": result["risk_score"],
            "warning_stage": result["warning_stage"],
            "warning_reason": result["warning_reason"],
            "baseline_ready": result["baseline_ready"],
            "event_count": result["event_count"],
            "attitude_score": result["component_scores"]["attitude"],
            "angular_rate_score": result["component_scores"]["angular_rate"],
            "speed_score": result["component_scores"]["speed"],
            "acceleration_score": result["component_scores"]["acceleration"],
        }
        self._ensure_writer(output)
        self._score_writer.writerow(output)
        if self.frame_count % 100 == 0 and self._score_file is not None:
            self._score_file.flush()
            print(
                f"[feature-watch] frames={self.frame_count} "
                f"fault_prob={result['fault_prob']:.3f} "
                f"risk_score={result['risk_score']:.3f} "
                f"stage={result['warning_stage']} "
                f"events={result['event_count']}"
            )
        if result["new_event_count"] > 0:
            print(
                f"[feature-watch] ⚠️ 新告警事件: "
                f"stage={result['warning_stage']} risk={result['risk_score']:.3f}"
            )

    def start(self) -> None:
        try:
            self.receiver.start()
        finally:
            if self._score_file is not None:
                self._score_file.flush()
                self._score_file.close()
            final_events = self.engine.aggregator.flush()
            self.engine.events.extend(final_events)
            print(
                f"[feature-watch] 完成: frames={self.frame_count} events={len(self.engine.events)}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="轻量实时特征告警服务")
    parser.add_argument("--port", type=int, default=15550)
    parser.add_argument("--raw-csv", default=None)
    parser.add_argument("--score-csv", default=None)
    parser.add_argument("--window", type=int, default=150)
    parser.add_argument("--warmup-frames", type=int, default=150)
    parser.add_argument("--baseline-window", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = RealtimeFeatureWatchService(
        port=args.port,
        raw_csv_path=args.raw_csv,
        score_csv_path=args.score_csv,
        window_size=args.window,
        warmup_frames=args.warmup_frames,
        baseline_window=args.baseline_window,
    )
    service.start()


if __name__ == "__main__":
    main()
