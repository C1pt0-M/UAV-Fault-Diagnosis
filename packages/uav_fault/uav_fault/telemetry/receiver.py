"""云端遥测接收服务。

监听 UDP 端口，将实时遥测数据记录到 CSV，并维护滑动窗口供后续 ML 推理使用。

用法:
    python -m telemetry.receiver
    python -m telemetry.receiver --port 15550 --csv telemetry.csv --window 150
"""

import argparse
import csv
import json
import socket
import threading
import time
from collections import deque

from uav_fault.telemetry.config import AUX_COLUMNS, FEATURE_COLUMNS, UDP_FORWARD_PORT


def _coerce_seq(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _required_fields_ready(row: dict, fields: list[str]) -> bool:
    for field in fields:
        try:
            if float(row.get(field, 0) or 0) <= 0.5:
                return False
        except (TypeError, ValueError):
            return False
    return True


class TelemetryReceiver:

    def __init__(self, port: int, csv_path: str, window_size: int, auto_stop: float = 5.0,
                 recv_timeout: float = 30.0, trim_on_stop: bool = True,
                 required_ready_fields: list[str] | None = None):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(recv_timeout if recv_timeout > 0 else None)
        self._sock.bind(("0.0.0.0", port))
        self._port = port
        self._csv_path = csv_path
        self._auto_stop = auto_stop
        self._recv_timeout = recv_timeout
        self._trim_on_stop = trim_on_stop
        self._required_ready_fields = list(required_ready_fields or [])
        self._stop_event = threading.Event()
        self.exit_reason: str | None = None
        self.window: deque[dict] = deque(maxlen=window_size)
        self.callbacks: list = []

    def start(self) -> None:
        self._stop_event.clear()
        self.exit_reason = None
        columns = ["seq", "time_s"] + FEATURE_COLUMNS + AUX_COLUMNS
        csv_file = open(self._csv_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        print(f"[receiver] 监听 UDP :{self._port}")
        print(f"[receiver] 记录到 {self._csv_path}")

        count = 0
        lost = 0
        bad_packets = 0
        expected_seq: int | None = None
        idle_since: float | None = None
        motors_seen = False
        armed_seen = False
        waiting_logged = False
        try:
            while not self._stop_event.is_set():
                try:
                    data, _ = self._sock.recvfrom(65535)
                except socket.timeout:
                    print(f"[receiver] 超过 {self._recv_timeout}s 未收到数据，自动停止")
                    self.exit_reason = "timeout"
                    break
                except OSError:
                    if self._stop_event.is_set():
                        break
                    raise

                # 解析 JSON，坏包丢弃继续
                try:
                    row = json.loads(data.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    bad_packets += 1
                    if bad_packets <= 5:
                        print(f"[receiver] 坏包丢弃 ({e})")
                    elif bad_packets == 6:
                        print(f"[receiver] 后续坏包将不再打印")
                    continue

                net_tx_ms = row.get("net_tx_ms")
                if net_tx_ms is not None:
                    net_rx_ms = int(time.time() * 1000)
                    row["net_rx_ms"] = net_rx_ms
                    try:
                        row["net_e2e_latency_ms"] = round(
                            max(0.0, net_rx_ms - float(net_tx_ms)),
                            3,
                        )
                    except (TypeError, ValueError):
                        row["net_e2e_latency_ms"] = ""

                if not _required_fields_ready(row, self._required_ready_fields):
                    if not waiting_logged:
                        print(
                            "[receiver] 等待遥测字段就绪: "
                            + ", ".join(self._required_ready_fields)
                        )
                        waiting_logged = True
                    continue
                if waiting_logged:
                    print("[receiver] 遥测字段已就绪，开始记录")
                    waiting_logged = False

                writer.writerow(row)
                self.window.append(row)
                count += 1

                # 丢包检测
                seq = _coerce_seq(
                    row.get("net_seq", row.get("seq", expected_seq or 0)),
                    expected_seq or 0,
                )
                if expected_seq is not None and seq > expected_seq:
                    lost += seq - expected_seq
                expected_seq = seq + 1

                if count % 500 == 0:
                    csv_file.flush()
                    loss_rate = (lost / (lost + count) * 100) if (lost + count) > 0 else 0
                    print(f"[receiver] 已接收 {count} 帧 | 丢包 {lost} ({loss_rate:.1f}%) | 坏包 {bad_packets}")

                for cb in self.callbacks:
                    try:
                        cb(row, self.window)
                    except Exception as e:
                        print(f"[receiver] callback 异常: {e}")

                # 优先根据 PX4 armed 状态停止；没有心跳字段时兼容旧电机归零逻辑。
                if self._auto_stop > 0:
                    heartbeat_seen = float(row.get("pymav_heartbeat_seen", 0) or 0) > 0.5
                    if heartbeat_seen:
                        if float(row.get("armed", 0) or 0) > 0.5:
                            armed_seen = True
                            idle_since = None
                        elif armed_seen:
                            if idle_since is None:
                                idle_since = time.time()
                            elif time.time() - idle_since >= self._auto_stop:
                                print(f"[receiver] 飞行器解除武装超过 {self._auto_stop}s，自动停止")
                                self.exit_reason = "auto_stop"
                                break
                    else:
                        motors_zero = all(
                            abs(row.get(f"motor{i}", 1)) < 1e-3 for i in range(1, 5)
                        )
                        if not motors_zero:
                            motors_seen = True
                            idle_since = None
                        elif motors_seen:
                            if idle_since is None:
                                idle_since = time.time()
                            elif time.time() - idle_since >= self._auto_stop:
                                print(f"[receiver] 电机停转超过 {self._auto_stop}s，自动停止")
                                self.exit_reason = "auto_stop"
                                break
        except KeyboardInterrupt:
            self.exit_reason = "interrupted"
            print(f"\n[receiver] 手动停止")
        finally:
            csv_file.close()
            self._sock.close()
            loss_rate = (lost / (lost + count) * 100) if (lost + count) > 0 else 0
            print(f"[receiver] 共接收 {count} 帧 | 丢包 {lost} ({loss_rate:.1f}%) | 坏包 {bad_packets}")
            if self._trim_on_stop:
                self._trim_tail()

    def stop(self) -> None:
        """Stop the receive loop and unblock a pending socket read."""
        if self.exit_reason is None:
            self.exit_reason = "stopped"
        self._stop_event.set()
        try:
            self._sock.close()
        except OSError:
            pass


    def _trim_tail(self) -> None:
        """裁掉电机全部归零后的数据（降落/坠机后的地面帧）。"""
        try:
            import pandas as pd

            df = pd.read_csv(self._csv_path)
            if df.empty:
                return

            motors = df[["motor1", "motor2", "motor3", "motor4"]]
            nonzero_mask = (motors.abs() > 1e-3).any(axis=1)

            if not nonzero_mask.any():
                print(f"[receiver] 警告：电机全程为零，不裁剪")
                print(f"[receiver] 已保存到 {self._csv_path}")
                return

            last_nonzero = nonzero_mask[::-1].idxmax()
            trimmed = df.iloc[: last_nonzero + 1]
            removed = len(df) - len(trimmed)

            trimmed.to_csv(self._csv_path, index=False)
            print(f"[receiver] 裁掉尾部 {removed} 帧（电机归零后），保留 {len(trimmed)} 帧")
            print(f"[receiver] 已保存到 {self._csv_path}")
        except Exception as e:
            print(f"[receiver] _trim_tail 异常: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="云端遥测接收服务")
    parser.add_argument("--port", type=int, default=UDP_FORWARD_PORT, help="监听端口")
    parser.add_argument("--csv", default=f"telemetry_{int(time.time())}.csv", help="CSV 输出路径")
    parser.add_argument("--window", type=int, default=150, help="滑动窗口大小 (帧数，50Hz 下 150=3 秒)")
    parser.add_argument("--auto-stop", type=float, default=5.0, help="电机停转后自动停止的秒数 (0=禁用)")
    parser.add_argument("--recv-timeout", type=float, default=30.0, help="接收超时秒数，超时自动停止 (0=禁用)")
    parser.add_argument(
        "--require-ready-field",
        action="append",
        default=[],
        help="字段值大于 0.5 后才开始记录，可重复指定",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receiver = TelemetryReceiver(
        args.port,
        args.csv,
        args.window,
        args.auto_stop,
        args.recv_timeout,
        required_ready_fields=args.require_ready_field,
    )
    receiver.start()


if __name__ == "__main__":
    main()
