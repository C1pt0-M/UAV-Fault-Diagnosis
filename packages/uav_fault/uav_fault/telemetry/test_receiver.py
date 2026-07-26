from __future__ import annotations

import csv
import json
from pathlib import Path

from uav_fault.telemetry.receiver import _coerce_seq


def test_coerce_seq_accepts_numeric_string() -> None:
    assert _coerce_seq("42", default=0) == 42


def test_coerce_seq_falls_back_on_invalid_value() -> None:
    assert _coerce_seq("not-a-number", default=7) == 7


def test_telemetry_receiver_persists_live_baro_temp_and_actuator_controls(
    tmp_path: Path, monkeypatch
):
    from uav_fault.telemetry.receiver import TelemetryReceiver

    payload = {
        "seq": 12,
        "time_s": 1.25,
        "motor1": 1200,
        "motor2": 1190,
        "motor3": 1210,
        "motor4": 1180,
        "baro_temp": 24.5,
        "actuator_ctrl_1": 0.11,
        "actuator_ctrl_2": -0.22,
        "actuator_ctrl_3": 0.33,
        "actuator_ctrl_4": 0.44,
    }

    class _FakeSocket:
        def __init__(self, packets):
            self._packets = list(packets)

        def settimeout(self, timeout):
            self.timeout = timeout

        def bind(self, addr):
            self.addr = addr

        def recvfrom(self, size):
            if self._packets:
                return self._packets.pop(0), ("127.0.0.1", 15550)
            raise KeyboardInterrupt

        def close(self):
            return None

    monkeypatch.setattr(
        "uav_fault.telemetry.receiver.socket.socket",
        lambda *args, **kwargs: _FakeSocket([json.dumps(payload).encode("utf-8")]),
    )
    rows = []
    csv_path = tmp_path / "receiver_live.csv"
    receiver = TelemetryReceiver(
        port=15550,
        csv_path=str(csv_path),
        window_size=4,
        auto_stop=0.0,
        recv_timeout=0.1,
        trim_on_stop=False,
    )
    receiver.callbacks.append(lambda row, window: rows.append((row, list(window))))

    receiver.start()

    with csv_path.open(newline="", encoding="utf-8") as f:
        saved_rows = list(csv.DictReader(f))

    assert len(saved_rows) == 1
    assert saved_rows[0]["baro_temp"] == "24.5"
    assert saved_rows[0]["actuator_ctrl_1"] == "0.11"
    assert saved_rows[0]["actuator_ctrl_2"] == "-0.22"
    assert saved_rows[0]["actuator_ctrl_3"] == "0.33"
    assert saved_rows[0]["actuator_ctrl_4"] == "0.44"
    assert rows[0][0]["baro_temp"] == 24.5
    assert rows[0][0]["actuator_ctrl_1"] == 0.11
    assert rows[0][1][0]["baro_temp"] == 24.5
    assert rows[0][1][0]["actuator_ctrl_4"] == 0.44
