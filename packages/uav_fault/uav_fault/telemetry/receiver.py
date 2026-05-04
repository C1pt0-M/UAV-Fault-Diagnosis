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
import time
from collections import deque

from uav_fault.telemetry.config import AUX_COLUMNS, FEATURE_COLUMNS, UDP_FORWARD_PORT


def _coerce_seq(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


class TelemetryReceiver:

    def __init__(self, port: int, csv_path: str, window_size: int, auto_stop: float = 5.0,
                 recv_timeout: float = 30.0, trim_on_stop: bool = True):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(recv_timeout if recv_timeout > 0 else None)
        self._sock.bind(("0.0.0.0", port))
        self._port = port
        self._csv_path = csv_path
        self._auto_stop = auto_stop
        self._recv_timeout = recv_timeout
        self._trim_on_stop = trim_on_stop
        self.window: deque[dict] = deque(maxlen=window_size)
        self.callbacks: list = []

    def start(self) -> None:
        columns = ["seq", "time_s"] + FEATURE_COLUMNS + AUX_COLUMNS
        csv_file = open(self._csv_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        print(f"[receiver] 监听 UDP :{self._port}")
        print(f"[receiver] 记录到 {self._csv_path}")

        count = 0
        lost = 0
        bad_packets = 0
        expected_seq = 0
        idle_since: float | None = None
        motors_seen = False
        try:
            while True:
                try:
                    data, _ = self._sock.recvfrom(4096)
                except socket.timeout:
                    print(f"[receiver] 超过 {self._recv_timeout}s 未收到数据，自动停止")
                    break

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

                writer.writerow(row)
                self.window.append(row)
                count += 1

                # 丢包检测
                seq = _coerce_seq(row.get("seq", expected_seq), expected_seq)
                if seq > expected_seq:
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

                # 自动停止：电机曾经转动过之后，全部归零持续 auto_stop 秒才停止
                if self._auto_stop > 0:
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
                            break
        except KeyboardInterrupt:
            print(f"\n[receiver] 手动停止")
        finally:
            csv_file.close()
            self._sock.close()
            loss_rate = (lost / (lost + count) * 100) if (lost + count) > 0 else 0
            print(f"[receiver] 共接收 {count} 帧 | 丢包 {lost} ({loss_rate:.1f}%) | 坏包 {bad_packets}")
            if self._trim_on_stop:
                self._trim_tail()


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receiver = TelemetryReceiver(args.port, args.csv, args.window, args.auto_stop, args.recv_timeout)
    receiver.start()


if __name__ == "__main__":
    main()
