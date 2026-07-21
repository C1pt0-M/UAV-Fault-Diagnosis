"""Process-isolated MEC node for RflySim telemetry inference."""

from __future__ import annotations

import argparse
import csv
import json
import queue
import socket
import threading
import time
from collections import deque
from pathlib import Path

from uav_fault.online.event_aggregator import FaultEventAggregator
from uav_fault.telemetry.schema_adapter import (
    assess_canonical_input_contract,
    canonicalize_rfly_frame,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _decision_latency_ms(net_tx_ms, completed_ms: int) -> float | None:
    try:
        value = float(net_tx_ms)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, float(completed_ms) - value), 3)


def _event_payload(event) -> dict:
    return {
        "start_t": float(event.start_t),
        "end_t": float(event.end_t),
        "start_frame_idx": event.start_frame_idx,
        "end_frame_idx": event.end_frame_idx,
        "n_points": int(event.n_points),
        "max_fault_prob": float(event.max_fault_prob),
        "mean_fault_prob": float(event.mean_fault_prob),
    }


class MecTelemetryService:
    def __init__(
        self,
        *,
        listen_host: str,
        listen_port: int,
        forward_host: str,
        forward_port: int,
        model_dir: str,
        window_size: int = 150,
        stride: int = 5,
        metrics_csv: str | None = None,
        device: str = "cpu",
    ):
        self.listen = (listen_host, int(listen_port))
        self.forward = (forward_host, int(forward_port))
        self.window_size = max(1, int(window_size))
        self.stride = max(1, int(stride))
        self.metrics_csv = Path(metrics_csv) if metrics_csv else None
        self.model_dir = str(model_dir)
        self.device = str(device)
        self.detector = None
        self.aggregator = self._new_aggregator()
        self.window: deque[dict] = deque(maxlen=self.window_size)
        self.inference_queue: queue.Queue = queue.Queue(maxsize=1)
        self.inference_generation = 0
        self.flight_active = False
        self.result_lock = threading.Lock()
        self.latest_result: dict = {
            "label": "normal",
            "confidence": 1.0,
            "fault_prob": 0.0,
            "inference_ms": None,
            "decision_latency_ms": None,
            "alert_active": False,
            "event_seq": 0,
            "events": [],
            "contract_status": "waiting_for_arm",
            "model_status": "waiting_for_arm",
        }
        self.frame_count = 0
        self.inference_count = 0
        self.inference_drops = 0
        self.inference_samples: deque[float] = deque(maxlen=1000)
        self.decision_latency_samples: deque[float] = deque(maxlen=1000)
        self.stop_event = threading.Event()

    @staticmethod
    def _new_aggregator() -> FaultEventAggregator:
        return FaultEventAggregator(
            onset_th=0.6,
            clear_th=0.4,
            k_on=3,
            k_off=3,
            min_event_s=0.5,
            merge_gap_s=0.5,
        )

    def _publish_result(self, result: dict) -> None:
        with self.result_lock:
            self.latest_result.update(result)

    def _result_snapshot(self) -> dict:
        with self.result_lock:
            return dict(self.latest_result)

    def _enqueue_inference(self, item: tuple[dict, list[dict], int, int]) -> None:
        try:
            self.inference_queue.put_nowait(item)
        except queue.Full:
            try:
                self.inference_queue.get_nowait()
            except queue.Empty:
                pass
            self.inference_drops += 1
            self.inference_queue.put_nowait(item)

    def _inference_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                canonical, window, frame_idx, generation = self.inference_queue.get(
                    timeout=0.2
                )
            except queue.Empty:
                continue
            if generation != self.inference_generation or not self.flight_active:
                continue
            if self.detector is None:
                self._publish_result({"model_status": "loading"})
                try:
                    # Keep torch/model initialization out of the RflySim startup path.
                    from uav_fault.online.fault_detector import FaultDetector

                    self.detector = FaultDetector(
                        model_dir=self.model_dir,
                        device=self.device,
                        window_size=self.window_size,
                    )
                    self._publish_result({"model_status": "ready"})
                except Exception as exc:
                    self._publish_result({"model_status": "failed"})
                    print(f"[mec] model load failed: {exc}", flush=True)
                    return
            started = time.perf_counter()
            detector = self.detector
            if detector is None:
                continue
            result = detector.predict(window)
            inference_ms = (time.perf_counter() - started) * 1000.0
            completed_ms = _now_ms()
            decision_latency_ms = _decision_latency_ms(
                canonical.get("net_tx_ms"), completed_ms
            )
            if generation != self.inference_generation or not self.flight_active:
                continue
            self.inference_count += 1
            self.inference_samples.append(inference_ms)
            if decision_latency_ms is not None:
                self.decision_latency_samples.append(decision_latency_ms)
            if result is None:
                continue
            fault_prob = float(result.get("fault_prob", 0.0))
            frame_time = float(canonical.get("time_s", frame_idx / 50.0))
            events = self.aggregator.update(
                t=frame_time,
                fault_prob=fault_prob,
                frame_idx=frame_idx,
            )
            active = getattr(self.aggregator, "_active", None) is not None
            previous = self._result_snapshot()
            event_seq = int(previous.get("event_seq") or 0)
            event_payloads = [_event_payload(event) for event in events]
            if event_payloads:
                event_seq += len(event_payloads)
            self._publish_result(
                {
                    "label": str(result.get("label") or "normal"),
                    "confidence": float(result.get("confidence", 0.0)),
                    "fault_prob": fault_prob,
                    "inference_ms": round(inference_ms, 3),
                    "decision_latency_ms": decision_latency_ms,
                    "alert_active": bool(active or fault_prob >= self.aggregator.onset_th),
                    "event_seq": event_seq,
                    "events": event_payloads,
                    "model_status": "ready",
                }
            )

    def _clear_inference_queue(self) -> None:
        while True:
            try:
                self.inference_queue.get_nowait()
            except queue.Empty:
                return

    def _reset_for_disarmed(self) -> None:
        if self.flight_active or self.window or not self.inference_queue.empty():
            self.inference_generation += 1
        self.flight_active = False
        self.window.clear()
        self._clear_inference_queue()
        self.aggregator = self._new_aggregator()
        self._publish_result(
            {
                "label": "normal",
                "confidence": 1.0,
                "fault_prob": 0.0,
                "inference_ms": None,
                "decision_latency_ms": None,
                "alert_active": False,
                "events": [],
                "contract_status": "waiting_for_arm",
                "model_status": "waiting_for_arm",
            }
        )

    def _handle_frame(self, row: dict) -> dict:
        self.frame_count += 1
        canonical, coverage = canonicalize_rfly_frame(row)
        armed = float(canonical.get("armed", row.get("armed", 0.0)) or 0.0) > 0.5
        if not armed:
            self._reset_for_disarmed()
            return self._decorate_frame(row)

        if not self.flight_active:
            self.flight_active = True
            self.inference_generation += 1
            self.window.clear()
            self.aggregator = self._new_aggregator()
            self._publish_result(
                {
                    "label": "normal",
                    "confidence": 1.0,
                    "fault_prob": 0.0,
                    "inference_ms": None,
                    "decision_latency_ms": None,
                    "alert_active": False,
                    "events": [],
                    "contract_status": "waiting_for_window",
                    "model_status": "waiting_for_window",
                }
            )

        self.window.append(canonical)
        if len(self.window) >= self.window_size and self.frame_count % self.stride == 0:
            contract = assess_canonical_input_contract(
                list(self.window),
                coverage=coverage,
                actuator_source=canonical.get("actuator_source")
                or row.get("actuator_source"),
            )
            self._publish_result({"contract_status": contract.get("status", "unknown")})
            if contract.get("detection_blocked"):
                self._publish_result(
                    {
                        "label": "normal",
                        "confidence": 1.0,
                        "fault_prob": 0.0,
                        "inference_ms": None,
                        "decision_latency_ms": None,
                        "alert_active": False,
                        "events": [],
                        "model_status": "blocked_by_contract",
                    }
                )
            else:
                self._enqueue_inference(
                    (
                        canonical,
                        list(self.window),
                        self.frame_count,
                        self.inference_generation,
                    )
                )
        return self._decorate_frame(row)

    def _decorate_frame(self, row: dict) -> dict:
        result = self._result_snapshot()
        samples = sorted(self.inference_samples)
        average = sum(samples) / len(samples) if samples else None
        p95 = samples[int(0.95 * (len(samples) - 1))] if samples else None
        decision_samples = sorted(self.decision_latency_samples)
        decision_average = (
            sum(decision_samples) / len(decision_samples) if decision_samples else None
        )
        decision_p95 = (
            decision_samples[int(0.95 * (len(decision_samples) - 1))]
            if decision_samples
            else None
        )
        row.update(
            {
                "mec_enabled": 1,
                "mec_inference_location": "mec",
                "mec_rx_ms": _now_ms(),
                "mec_fault_prob": result.get("fault_prob", 0.0),
                "mec_label": result.get("label", "normal"),
                "mec_confidence": result.get("confidence", 0.0),
                "mec_inference_ms": result.get("inference_ms"),
                "mec_avg_inference_ms": None if average is None else round(average, 3),
                "mec_p95_inference_ms": None if p95 is None else round(p95, 3),
                "mec_decision_latency_ms": result.get("decision_latency_ms"),
                "mec_avg_decision_latency_ms": None
                if decision_average is None
                else round(decision_average, 3),
                "mec_p95_decision_latency_ms": None
                if decision_p95 is None
                else round(decision_p95, 3),
                "mec_alert_active": int(bool(result.get("alert_active"))),
                "mec_event_seq": int(result.get("event_seq") or 0),
                "mec_events": result.get("events") or [],
                "mec_inference_count": self.inference_count,
                "mec_frames_forwarded": self.frame_count,
                "mec_inference_queue_drops": self.inference_drops,
                "mec_contract_status": result.get("contract_status", "waiting"),
                "mec_model_status": result.get("model_status", "loading"),
                "mec_tx_ms": _now_ms(),
            }
        )
        return row

    def _write_metrics(self) -> None:
        if self.metrics_csv is None:
            return
        self.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        samples = sorted(self.inference_samples)
        avg = sum(samples) / len(samples) if samples else 0.0
        p95 = samples[int(0.95 * (len(samples) - 1))] if samples else 0.0
        decision_samples = sorted(self.decision_latency_samples)
        decision_avg = (
            sum(decision_samples) / len(decision_samples) if decision_samples else 0.0
        )
        decision_p95 = (
            decision_samples[int(0.95 * (len(decision_samples) - 1))]
            if decision_samples
            else 0.0
        )
        with self.metrics_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "frames_forwarded",
                    "inference_count",
                    "inference_queue_drops",
                    "inference_avg_ms",
                    "inference_p95_ms",
                    "decision_latency_avg_ms",
                    "decision_latency_p95_ms",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "frames_forwarded": self.frame_count,
                    "inference_count": self.inference_count,
                    "inference_queue_drops": self.inference_drops,
                    "inference_avg_ms": round(avg, 3),
                    "inference_p95_ms": round(p95, 3),
                    "decision_latency_avg_ms": round(decision_avg, 3),
                    "decision_latency_p95_ms": round(decision_p95, 3),
                }
            )

    def run(self) -> None:
        sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_in.bind(self.listen)
        sock_in.settimeout(0.5)
        sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(
            f"[mec] listen={self.listen[0]}:{self.listen[1]} "
            f"forward={self.forward[0]}:{self.forward[1]} stride={self.stride}"
        )
        worker = threading.Thread(
            target=self._inference_worker,
            name="mec-inference",
            daemon=True,
        )
        worker.start()
        try:
            while not self.stop_event.is_set():
                try:
                    payload, _ = sock_in.recvfrom(65535)
                except socket.timeout:
                    continue
                try:
                    row = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                decorated = self._handle_frame(row)
                sock_out.sendto(
                    json.dumps(decorated, separators=(",", ":")).encode("utf-8"),
                    self.forward,
                )
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            worker.join(timeout=5)
            self._write_metrics()
            sock_in.close()
            sock_out.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RflySim MEC edge inference service")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=15552)
    parser.add_argument("--forward-host", default="127.0.0.1")
    parser.add_argument("--forward-port", type=int, default=15550)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--window-size", type=int, default=150)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MecTelemetryService(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        forward_host=args.forward_host,
        forward_port=args.forward_port,
        model_dir=args.model_dir,
        window_size=args.window_size,
        stride=args.stride,
        metrics_csv=args.metrics_csv,
        device=args.device,
    ).run()


if __name__ == "__main__":
    main()
