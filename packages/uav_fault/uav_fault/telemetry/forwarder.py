"""UDP JSON-lines 转发器。"""

import json
import socket


class UDPForwarder:

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._target = (host, port)
        self._seq = 0
        self._send_errors = 0

    def send(self, row: dict) -> None:
        row["seq"] = self._seq
        self._seq += 1
        data = json.dumps(row, separators=(",", ":")).encode("utf-8")
        try:
            self._sock.sendto(data, self._target)
        except OSError as e:
            self._send_errors += 1
            if self._send_errors <= 3:
                print(f"[forwarder] 发送失败: {e}")
            elif self._send_errors == 4:
                print(f"[forwarder] 后续发送错误将不再打印")

    def close(self) -> None:
        self._sock.close()
        if self._send_errors > 0:
            print(f"[forwarder] 累计发送失败 {self._send_errors} 次")
