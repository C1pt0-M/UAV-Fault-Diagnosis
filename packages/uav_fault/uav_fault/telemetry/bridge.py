"""桥接脚本：运行 TelemetryReceiver 并将每帧数据以 JSON line 输出到 stdout。

供 Node 后端通过 spawn 调用，实现 Python receiver → Node socket.io → React 前端的数据链路。

用法:
    python -m telemetry.bridge
    python -m telemetry.bridge --csv-dir /path/to/csv
"""

import argparse
import json
import os
import sys
import time

from .config import UDP_FORWARD_PORT
from .receiver import TelemetryReceiver


def stdout_callback(row: dict, window) -> None:
    """将每帧数据作为 JSON line 写入 stdout 供 Node 消费。"""
    msg = {"type": "frame", "data": row}
    sys.stdout.write(json.dumps(msg, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="遥测桥接服务 (receiver → stdout JSON)")
    parser.add_argument("--port", type=int, default=UDP_FORWARD_PORT, help="UDP 监听端口")
    parser.add_argument("--window", type=int, default=150, help="滑动窗口大小")
    parser.add_argument("--csv-dir", default=None, help="CSV 输出目录 (默认: /tmp)")
    parser.add_argument("--scene-id", default=None, help="场景ID，用于CSV文件命名")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    filename = f"{args.scene_id}.csv" if args.scene_id else f"telemetry_live_{int(time.time())}.csv"

    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        csv_path = os.path.join(args.csv_dir, filename)
    else:
        csv_path = os.path.join("/tmp", filename)

    receiver = TelemetryReceiver(
        port=args.port,
        csv_path=csv_path,
        window_size=args.window,
        auto_stop=0,
        recv_timeout=300.0,
        trim_on_stop=False,
    )
    receiver.callbacks.append(stdout_callback)

    # 通知 Node 已就绪
    sys.stdout.write(json.dumps({"type": "ready", "port": args.port}) + "\n")
    sys.stdout.flush()

    receiver.start()

    # receiver 停止后通知 Node
    sys.stdout.write(json.dumps({"type": "stopped"}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
