#!/usr/bin/env python3
"""
按场景表顺序触发 Fault Dataset 在线采集（withTelemetry=true）。

用途：
- 复用 scenarios/*.csv 的参数范围；
- 通过 dashboard backend 的 /api/inject 串行触发采集；
- 每条任务写入 JSONL 日志，支持断点续跑；
- 默认自动跳过已产出 telemetry_live/csv/<scene_id>.csv 的场景。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCENARIO_FILES = [
    ("normal_flight", "normal_flight.csv"),
    ("motor_failure", "motor_failure.csv"),
    ("battery_failure", "battery_failure.csv"),
    ("prop_imbalance", "prop_imbalance.csv"),
    ("thrust_loss", "thrust_loss.csv"),
]


def _to_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    s = str(v).strip()
    if s == "":
        return default
    return float(s)


def _to_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    s = str(v).strip()
    if s == "":
        return default
    return int(float(s))


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    token: str = "",
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return e.code, raw


def _build_inject_payload(fault_type: str, row: dict[str, str]) -> tuple[dict[str, Any], float]:
    sim_time_s = _to_float(row.get("sim_time_s"), 60.0)
    inject_mode = row.get("inject_mode", "step") or "step"
    curve = row.get("curve", "none") or "none"
    ramp_duration = _to_float(row.get("ramp_duration"), 0.0)

    if fault_type == "normal_flight":
        params = {
            "sim_time_s": sim_time_s,
        }
    elif fault_type == "motor_failure":
        params = {
            "motor_index": _to_int(row.get("motor_index"), 1),
            "scale": _to_float(row.get("scale"), 0.8),
            "fault_ratio": _to_float(row.get("fault_ratio"), 0.3),
            "sim_time_s": sim_time_s,
            "inject_mode": inject_mode,
            "ramp_duration": ramp_duration,
            "curve": curve,
        }
    elif fault_type == "battery_failure":
        params = {
            "drain_time": _to_float(row.get("drain_time"), 20.0),
            "fault_ratio": _to_float(row.get("fault_ratio"), 0.3),
            "sim_time_s": sim_time_s,
            "inject_mode": inject_mode,
            "ramp_duration": ramp_duration,
            "curve": curve,
        }
    elif fault_type == "prop_imbalance":
        params = {
            "motor_index": _to_int(row.get("motor_index"), 1),
            "amp": _to_float(row.get("amp"), 0.2),
            "frequency_hz": _to_float(row.get("freq_hz"), 12.0),
            "fault_ratio": _to_float(row.get("fault_ratio"), 0.3),
            "sim_time_s": sim_time_s,
            "inject_mode": inject_mode,
            "ramp_duration": ramp_duration,
            "curve": curve,
        }
    elif fault_type == "thrust_loss":
        params = {
            "scale": _to_float(row.get("scale"), 0.8),
            "fault_ratio": _to_float(row.get("fault_ratio"), 0.3),
            "sim_time_s": sim_time_s,
            "inject_mode": inject_mode,
            "ramp_duration": ramp_duration,
            "curve": curve,
        }
    else:
        raise ValueError(f"unsupported fault type: {fault_type}")

    payload = {
        "faultType": fault_type,
        "withTelemetry": True,
        "params": params,
    }
    return payload, sim_time_s


def _load_rows(scenarios_dir: Path) -> list[tuple[str, dict[str, str]]]:
    rows: list[tuple[str, dict[str, str]]] = []
    for fault_type, fname in SCENARIO_FILES:
        path = scenarios_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"missing scenario file: {path}")
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((fault_type, row))
    return rows


def _load_done_scene_ids(log_path: Path) -> set[str]:
    done = set()
    if not log_path.exists():
        return done
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            scene_id = obj.get("scene_id")
            if not scene_id:
                continue
            # 1) 已明确完成
            if obj.get("ok") is True:
                done.add(str(scene_id))
                continue
            # 2) 已成功提交（用于中断后避免重复提交同一 scene）
            submit_code = obj.get("submit_code")
            event = obj.get("event")
            if event == "submitted" and submit_code in (200, 202):
                done.add(str(obj["scene_id"]))
    return done


def _load_done_scene_ids_from_outputs(fault_root: Path) -> set[str]:
    """
    读取已完成在线 CSV 的 scene_id（用于自动跳过已跑过场景）。
    """
    done = set()
    csv_dir = fault_root / "data" / "telemetry_live" / "csv"
    if not csv_dir.exists():
        return done
    for p in csv_dir.glob("*.csv"):
        done.add(p.stem)
    return done


def _append_log(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _wait_after_submit(
    api_base: str,
    *,
    sim_time_s: float,
    floor_extra_s: float,
    poll_timeout_s: float,
    poll_interval_s: float,
    idle_grace_s: float,
) -> dict[str, Any]:
    start = time.time()
    # 给当前场景最基本的运行时间，避免过早提交下一条
    floor_wait = max(30.0, sim_time_s + floor_extra_s)
    time.sleep(floor_wait)

    idle_since: float | None = None
    deadline = time.time() + poll_timeout_s
    last_status: dict[str, Any] = {}
    while time.time() < deadline:
        code, body = _http_json(f"{api_base}/api/telemetry/status", timeout=10.0)
        if code == 200 and isinstance(body, dict):
            last_status = body
            running = bool(body.get("running", False))
            if running:
                idle_since = None
            else:
                if idle_since is None:
                    idle_since = time.time()
                elif time.time() - idle_since >= idle_grace_s:
                    return {
                        "wait_ok": True,
                        "elapsed_s": round(time.time() - start, 1),
                        "telemetry_status": last_status,
                    }
        time.sleep(poll_interval_s)

    return {
        "wait_ok": False,
        "elapsed_s": round(time.time() - start, 1),
        "telemetry_status": last_status,
        "note": "poll_timeout",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按场景表批量触发在线采集（withTelemetry=true）")
    p.add_argument("--fault-root", default="/home/jack/Fault Dataset", help="Fault Dataset 根目录")
    p.add_argument("--api-base", default="http://127.0.0.1:3001", help="dashboard backend 地址")
    p.add_argument("--api-token", default="", help="可选 Bearer token")
    p.add_argument("--max-jobs", type=int, default=0, help="仅跑前 N 条（0=全部）")
    p.add_argument("--resume", action="store_true", help="根据日志跳过已成功 scene_id")
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否自动跳过 data/telemetry_live/csv 中已存在的 scene_id（默认开启）",
    )
    p.add_argument("--dry-run", action="store_true", help="仅打印，不调用接口")
    p.add_argument("--floor-extra-s", type=float, default=35.0, help="每条任务额外最短等待秒数")
    p.add_argument("--poll-timeout-s", type=float, default=180.0, help="最短等待后轮询状态的超时时间")
    p.add_argument("--poll-interval-s", type=float, default=3.0, help="状态轮询间隔")
    p.add_argument("--idle-grace-s", type=float, default=8.0, help="running=false 持续该秒数后判定完成")
    p.add_argument("--http-timeout-s", type=float, default=30.0, help="HTTP 超时")
    p.add_argument("--log", default="", help="JSONL 日志路径（默认写到 fault_root/data/telemetry_live）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    fault_root = Path(args.fault_root).resolve()
    scenarios_dir = fault_root / "scenarios"
    if not fault_root.exists():
        raise SystemExit(f"[ERROR] fault root not found: {fault_root}")
    if not scenarios_dir.exists():
        raise SystemExit(f"[ERROR] scenarios dir not found: {scenarios_dir}")

    if args.log:
        log_path = Path(args.log).resolve()
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = fault_root / "data" / "telemetry_live" / f"online_batch_{ts}.jsonl"

    rows = _load_rows(scenarios_dir)
    if args.max_jobs > 0:
        rows = rows[: args.max_jobs]

    done_scene_ids_log = _load_done_scene_ids(log_path) if args.resume else set()
    done_scene_ids_outputs = _load_done_scene_ids_from_outputs(fault_root) if args.skip_existing else set()
    all_scene_ids = {r.get("scene_id", "") for _, r in rows if r.get("scene_id", "")}
    skip_by_log = all_scene_ids & done_scene_ids_log
    skip_by_outputs = all_scene_ids & done_scene_ids_outputs
    done_scene_ids = skip_by_log | skip_by_outputs
    pre_total = len(rows)
    if done_scene_ids:
        rows = [(ft, r) for ft, r in rows if r.get("scene_id", "") not in done_scene_ids]
    skipped = pre_total - len(rows)

    print(f"[INFO] total rows to run: {len(rows)}")
    print(
        "[INFO] skipped existing rows: "
        f"{skipped} (from_log_hit={len(skip_by_log)}, from_outputs_hit={len(skip_by_outputs)})"
    )
    print(f"[INFO] log path: {log_path}")
    print(f"[INFO] api base: {args.api_base}")

    # 预检查接口
    if not args.dry_run:
        code, body = _http_json(f"{args.api_base}/api/telemetry/status", timeout=args.http_timeout_s)
        if code != 200:
            raise SystemExit(f"[ERROR] backend unavailable: GET /api/telemetry/status -> {code}, body={body}")

    ok = 0
    fail = 0
    for idx, (fault_type, row) in enumerate(rows, start=1):
        scene_id = row.get("scene_id", f"unknown_{idx}")
        payload, sim_time_s = _build_inject_payload(fault_type, row)
        print(f"[{idx}/{len(rows)}] submit {scene_id} ({fault_type}, sim={sim_time_s}s)")

        rec: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "idx": idx,
            "scene_id": scene_id,
            "fault_type": fault_type,
            "sim_time_s": sim_time_s,
            "payload": payload,
        }

        if args.dry_run:
            rec["ok"] = True
            rec["note"] = "dry_run"
            _append_log(log_path, rec)
            ok += 1
            continue

        code, body = _http_json(
            f"{args.api_base}/api/inject",
            method="POST",
            payload=payload,
            token=args.api_token,
            timeout=args.http_timeout_s,
        )
        rec["submit_code"] = code
        rec["submit_resp"] = body
        if code not in (200, 202):
            rec["ok"] = False
            rec["note"] = "submit_failed"
            rec["event"] = "final"
            _append_log(log_path, rec)
            print(f"  [FAIL] submit code={code}")
            fail += 1
            continue

        # 先记录“已提交”，确保中断后 --resume 可跳过当前 scene_id
        submit_rec = {
            "ts": rec["ts"],
            "idx": rec["idx"],
            "scene_id": rec["scene_id"],
            "fault_type": rec["fault_type"],
            "sim_time_s": rec["sim_time_s"],
            "event": "submitted",
            "submit_code": code,
            "submit_resp": body,
        }
        _append_log(log_path, submit_rec)

        wait_ret = _wait_after_submit(
            args.api_base,
            sim_time_s=sim_time_s,
            floor_extra_s=args.floor_extra_s,
            poll_timeout_s=args.poll_timeout_s,
            poll_interval_s=args.poll_interval_s,
            idle_grace_s=args.idle_grace_s,
        )
        rec.update(wait_ret)
        rec["ok"] = bool(wait_ret.get("wait_ok", False))
        rec["event"] = "final"
        _append_log(log_path, rec)
        if rec["ok"]:
            ok += 1
            print(f"  [OK] done in {wait_ret.get('elapsed_s')}s")
        else:
            fail += 1
            print(f"  [WARN] timeout after {wait_ret.get('elapsed_s')}s")

    print(f"[DONE] ok={ok}, fail={fail}, log={log_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
