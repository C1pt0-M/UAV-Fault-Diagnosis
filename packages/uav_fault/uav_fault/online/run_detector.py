"""
在线故障检测独立服务。

支持两种运行模式：
1. stdin 模式（默认）：从 stdin 读取 JSON lines（bridge 输出），实时推理后输出到 stdout
   用法: python -m uav_fault.telemetry.bridge | python -m uav_fault.online.run_detector
2. csv 模式：读取已有 CSV 文件模拟实时推理，用于离线验证
   用法: python -m uav_fault.online.run_detector --mode csv --csv path/to/telemetry.csv

输出格式 (JSON lines):
  {"type":"fault_detection","seq":1234,"data":{"label":"fault","confidence":0.95,...}}
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

from uav_fault.online.event_aggregator import FaultEventAggregator
from uav_fault.online.fault_detector import FaultDetector
from uav_fault.online.sqlite_store import EventRow, SQLiteStore
from uav_fault.telemetry.schema_adapter import canonicalize_rfly_records, normalize_time_columns


def _utc_now_iso() -> str:
    import datetime as dt

    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat(timespec="milliseconds")


def run_stdin_mode(
    detector,
    window_size,
    every_n,
    *,
    store: SQLiteStore | None = None,
    run_id: int | None = None,
    aggregator: FaultEventAggregator | None = None,
):
    """从 stdin 读取 bridge 输出的 JSON lines，执行推理。

    Stdout 行为保持不变：透传原始 frame，并追加 fault_detection JSON lines。
    若提供 store/run_id/aggregator，则额外写入事件到 SQLite（副作用，不影响 stdout）。
    """
    window = deque(maxlen=window_size)
    frame_count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        # 只处理 frame 类型消息
        if msg.get('type') != 'frame':
            # 透传非 frame 消息
            sys.stdout.write(line + '\n')
            sys.stdout.flush()
            continue

        row = msg.get('data', {})
        window.append(row)
        frame_count += 1

        # 透传原始 frame
        sys.stdout.write(line + '\n')
        sys.stdout.flush()

        # 每 N 帧推理一次
        if frame_count % every_n != 0:
            continue
        if len(window) < window_size:
            continue

        result = detector.predict(window)
        if result is None:
            continue

        out = {
            'type': 'fault_detection',
            'seq': int(row.get('seq', frame_count)),
            'data': result,
        }
        sys.stdout.write(json.dumps(out, separators=(',', ':')) + '\n')
        sys.stdout.flush()

        if store is None or run_id is None or aggregator is None:
            continue

        t_val = row.get("time_s", None)
        try:
            t = float(t_val) if t_val is not None else float(frame_count)
        except Exception:
            t = float(frame_count)

        finalized = aggregator.update(
            t=t,
            fault_prob=float(result.get("fault_prob", 0.0)),
            frame_idx=int(row.get("seq", frame_count)) if row.get("seq") is not None else frame_count,
        )
        if not finalized:
            continue

        store.insert_events(
            [
                EventRow(
                    run_id=int(run_id),
                    start_t=e.start_t,
                    end_t=e.end_t,
                    start_frame_idx=e.start_frame_idx,
                    end_frame_idx=e.end_frame_idx,
                    n_points=e.n_points,
                    max_fault_prob=e.max_fault_prob,
                    mean_fault_prob=e.mean_fault_prob,
                    meta=e.meta,
                )
                for e in finalized
            ]
        )


def run_csv_mode(detector, csv_path, window_size, every_n):
    """读取 CSV 文件模拟实时推理，用于离线验证。"""
    import pandas as pd

    df = normalize_time_columns(pd.read_csv(csv_path))
    records = canonicalize_rfly_records(df)
    print(f"读取 CSV: {csv_path} ({len(df)} 行)", file=sys.stderr)

    window = deque(maxlen=window_size)
    results = []
    t_start = time.perf_counter()

    for idx, row in enumerate(records):
        window.append(row)

        if (idx + 1) % every_n != 0:
            continue
        if len(window) < window_size:
            continue

        result = detector.predict(window)
        if result is not None:
            result['frame_idx'] = idx
            result['time_s'] = row.get('time_s', 0.0)
            results.append(result)

    elapsed = time.perf_counter() - t_start
    n_inferences = len(results)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"推理完成: {n_inferences} 次推理, 耗时 {elapsed:.2f}s", file=sys.stderr)
    if n_inferences > 0:
        print(f"平均推理间隔: {elapsed / n_inferences * 1000:.1f}ms", file=sys.stderr)

        # 统计结果
        n_fault = sum(1 for r in results if r['label'] == 'fault')
        n_normal = n_inferences - n_fault
        print(f"检测结果: normal={n_normal}, fault={n_fault}", file=sys.stderr)

        # 输出详细结果
        for r in results:
            print(json.dumps(r, separators=(',', ':')))
    return df, results


def main():
    parser = argparse.ArgumentParser(description='在线故障检测独立服务')
    parser.add_argument('--model-dir', required=True,
                        help='模型目录 (包含 best.pt, config.json, norm_stats.json)')
    parser.add_argument('--mode', choices=['stdin', 'csv'], default='stdin',
                        help='运行模式: stdin (读取 bridge 输出) 或 csv (离线验证)')
    parser.add_argument('--csv', default=None, help='CSV 文件路径 (csv 模式)')
    parser.add_argument('--window', type=int, default=0, help='滑动窗口大小 (0=使用模型/缓存配置)')
    parser.add_argument('--every', type=int, default=10, help='每 N 帧推理一次')
    parser.add_argument('--device', default='cpu', help='推理设备 (cpu/cuda)')
    parser.add_argument('--db', default='', help='SQLite 数据库路径 (留空则不写入)')
    parser.add_argument('--run-name', default='', help='可选：run 名称（便于查询）')
    # Event aggregation (event-level delivery)
    parser.add_argument('--onset-th', type=float, default=0.6, help='进入故障事件阈值 (fault_prob)')
    parser.add_argument('--clear-th', type=float, default=0.4, help='退出故障事件阈值 (fault_prob)')
    parser.add_argument('--k-on', type=int, default=3, help='进入阈值连续满足次数')
    parser.add_argument('--k-off', type=int, default=3, help='退出阈值连续满足次数')
    parser.add_argument('--min-event-s', type=float, default=0.5, help='最短事件时长（秒），短于此过滤')
    parser.add_argument('--merge-gap-s', type=float, default=0.5, help='事件间隔小于该值则合并（秒）')
    args = parser.parse_args()

    detector = FaultDetector(
        model_dir=args.model_dir,
        device=args.device,
        window_size=(args.window if args.window > 0 else None),
    )

    store = None
    run_id = None

    # Optional: DB logging (event-level).
    if args.db:
        try:
            store = SQLiteStore(args.db)

            model_dir = Path(args.model_dir)
            name = model_dir.name
            task = "online_binary"

            config = {}
            norm_stats = {}
            metrics = None
            try:
                with open(model_dir / "config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                config = {}
            try:
                with open(model_dir / "norm_stats.json", "r", encoding="utf-8") as f:
                    norm_stats = json.load(f)
            except Exception:
                norm_stats = {}
            m_path = model_dir / "metrics_test.json"
            if m_path.exists():
                try:
                    with open(m_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                except Exception:
                    metrics = None

            model_id = store.upsert_model(
                name=name,
                task=task,
                model_dir=str(model_dir),
                config=config or None,
                input_columns=norm_stats.get("columns"),
                metrics=metrics,
            )

            run_params = {
                "mode": args.mode,
                "device": args.device,
                "window_size": int(detector.window_size),
                "every_n": int(args.every),
                "event_agg": {
                    "onset_th": float(args.onset_th),
                    "clear_th": float(args.clear_th),
                    "k_on": int(args.k_on),
                    "k_off": int(args.k_off),
                    "min_event_s": float(args.min_event_s),
                    "merge_gap_s": float(args.merge_gap_s),
                },
            }

            run_id = store.create_run(
                name=(args.run_name or None),
                kind=("offline_file" if args.mode == "csv" else "realtime_session"),
                status="running",
                asset_id=None,
                model_id=model_id,
                params=run_params,
                started_at=_utc_now_iso(),
            )
        except Exception as e:
            print(f"[WARN] 初始化 SQLite 失败，将继续但不写入 DB: {e}", file=sys.stderr)
            store = None
            run_id = None

    if args.mode == 'stdin':
        aggregator = FaultEventAggregator(
            onset_th=args.onset_th,
            clear_th=args.clear_th,
            k_on=args.k_on,
            k_off=args.k_off,
            min_event_s=args.min_event_s,
            merge_gap_s=args.merge_gap_s,
        )
        try:
            run_stdin_mode(
                detector,
                detector.window_size,
                args.every,
                store=store,
                run_id=run_id,
                aggregator=aggregator,
            )
        finally:
            if store is not None and run_id is not None:
                try:
                    # Flush any pending event.
                    leftover = aggregator.flush()
                    if leftover:
                        store.insert_events(
                            [
                                EventRow(
                                    run_id=int(run_id),
                                    start_t=e.start_t,
                                    end_t=e.end_t,
                                    start_frame_idx=e.start_frame_idx,
                                    end_frame_idx=e.end_frame_idx,
                                    n_points=e.n_points,
                                    max_fault_prob=e.max_fault_prob,
                                    mean_fault_prob=e.mean_fault_prob,
                                    meta=e.meta,
                                )
                                for e in leftover
                            ]
                        )
                    store.update_run(run_id, status="success", finished_at=_utc_now_iso())
                except Exception as e:
                    store.update_run(run_id, status="failed", error=str(e), finished_at=_utc_now_iso())
            if store is not None:
                store.close()
    elif args.mode == 'csv':
        if not args.csv:
            parser.error('csv 模式需要 --csv 参数')
        # DB: link the input file as an "asset".
        asset_id = None
        if store is not None and run_id is not None:
            try:
                p = Path(args.csv)
                asset_id = store.upsert_asset(
                    kind="csv",
                    path=str(p),
                    size_bytes=p.stat().st_size if p.exists() else None,
                )
                store.conn.execute("UPDATE runs SET asset_id = ? WHERE id = ?", (asset_id, int(run_id)))
                store.conn.commit()
            except Exception as e:
                print(f"[WARN] 写入 asset 失败: {e}", file=sys.stderr)

        aggregator = FaultEventAggregator(
            onset_th=args.onset_th,
            clear_th=args.clear_th,
            k_on=args.k_on,
            k_off=args.k_off,
            min_event_s=args.min_event_s,
            merge_gap_s=args.merge_gap_s,
        )

        try:
            df, results = run_csv_mode(detector, args.csv, detector.window_size, args.every)

            # Best-effort asset enrichment from time_s.
            if store is not None and asset_id is not None and "time_s" in df.columns and len(df) > 1:
                try:
                    t = df["time_s"].astype(float)
                    dt = t.diff()
                    dt = dt[(dt > 0) & dt.notna()]
                    hz = float(1.0 / dt.median()) if not dt.empty else None
                    store.upsert_asset(
                        kind="csv",
                        path=str(Path(args.csv)),
                        sample_hz=hz,
                        t_start=float(t.iloc[0]),
                        t_end=float(t.iloc[-1]),
                    )
                except Exception:
                    pass

            # Aggregate events from inference points.
            events: list[EventRow] = []
            for r in results:
                t = r.get("time_s", None)
                if t is None:
                    continue
                try:
                    t_f = float(t)
                except Exception:
                    continue
                finalized = aggregator.update(
                    t=t_f,
                    fault_prob=float(r.get("fault_prob", 0.0)),
                    frame_idx=int(r.get("frame_idx")) if r.get("frame_idx") is not None else None,
                )
                if finalized:
                    events.extend(
                        [
                            EventRow(
                                run_id=int(run_id) if run_id is not None else -1,
                                start_t=e.start_t,
                                end_t=e.end_t,
                                start_frame_idx=e.start_frame_idx,
                                end_frame_idx=e.end_frame_idx,
                                n_points=e.n_points,
                                max_fault_prob=e.max_fault_prob,
                                mean_fault_prob=e.mean_fault_prob,
                                meta=e.meta,
                            )
                            for e in finalized
                        ]
                    )
            leftover = aggregator.flush()
            if leftover:
                events.extend(
                    [
                        EventRow(
                            run_id=int(run_id) if run_id is not None else -1,
                            start_t=e.start_t,
                            end_t=e.end_t,
                            start_frame_idx=e.start_frame_idx,
                            end_frame_idx=e.end_frame_idx,
                            n_points=e.n_points,
                            max_fault_prob=e.max_fault_prob,
                            mean_fault_prob=e.mean_fault_prob,
                            meta=e.meta,
                        )
                        for e in leftover
                    ]
                )

            if store is not None and run_id is not None:
                # Ensure run_id is correct on all rows.
                for i, er in enumerate(events):
                    if er.run_id != int(run_id):
                        events[i] = EventRow(
                            run_id=int(run_id),
                            start_t=er.start_t,
                            end_t=er.end_t,
                            start_frame_idx=er.start_frame_idx,
                            end_frame_idx=er.end_frame_idx,
                            n_points=er.n_points,
                            max_fault_prob=er.max_fault_prob,
                            mean_fault_prob=er.mean_fault_prob,
                            meta=er.meta,
                        )

                store.insert_events(events)

                total_fault_s = sum(max(0.0, e.end_t - e.start_t) for e in events)
                max_p = max((e.max_fault_prob or 0.0) for e in events) if events else 0.0
                summary = {
                    "n_inferences": int(len(results)),
                    "n_events": int(len(events)),
                    "total_fault_s": float(total_fault_s),
                    "max_event_fault_prob": float(max_p),
                }
                store.update_run(run_id, status="success", summary=summary, finished_at=_utc_now_iso())
                print(f"[DB] wrote {len(events)} events to {store.db_path} (run_id={run_id})", file=sys.stderr)
        except Exception as e:
            if store is not None and run_id is not None:
                store.update_run(run_id, status="failed", error=str(e), finished_at=_utc_now_iso())
            raise
        finally:
            if store is not None:
                store.close()


if __name__ == '__main__':
    main()
