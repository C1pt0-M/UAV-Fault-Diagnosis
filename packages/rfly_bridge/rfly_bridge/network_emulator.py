#!/usr/bin/env python3
"""
UDP 5G 链路仿真 Adapter。

作用：
- 监听一个 UDP 端口接收遥测 JSON 数据帧
- 按配置模拟链路时延/抖动/丢包/带宽限制
- 转发到下游 UDP 端口（例如 receiver/bridge）
- 可选输出网络 KPI（吞吐、丢包、时延分布）到 CSV

推荐用法：
  python -m rfly_bridge.network_emulator \\
    --config-csv packages/rfly_bridge/rfly_bridge/configs/network_profiles.csv \\
    --profile baseline \\
    --listen-port 15551 \\
    --forward-port 15550 \\
    --metrics-csv runtime/rflysim/net_baseline.csv

说明：
- 本脚本不依赖第三方库，仅用标准库。
- 转发的每帧会新增字段：
  - net_profile: 场景名
  - net_tx_ms: 链路入口时间戳（ms, time.time）
  - net_delay_ms: 本帧实际施加的延迟（ms）
  用于下游计算端到端时延（now_ms - net_tx_ms）。

- metrics CSV 中的 `delay_avg_ms / delay_p95_ms` 统计的是“实际链路时延”（从入队到发出），
  会包含排队与带宽限制带来的额外等待，适合直接用于对比实验。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class LinkProfile:
    name: str
    delay_ms: float
    jitter_ms: float
    loss_rate: float
    bandwidth_kbps: float
    description: str = ""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def load_profile(config_csv: str, profile: str) -> LinkProfile:
    with open(config_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        if (r.get("profile") or "").strip() == profile:
            desc = (r.get("description") or "").strip()
            delay_ms = float(r.get("latency_mean_ms") or 0.0)
            jitter_ms = float(r.get("jitter_ms") or 0.0)
            loss_pct = float(r.get("packet_loss_pct") or 0.0)
            bw_mbps = float(r.get("bandwidth_mbps") or 0.0)
            return LinkProfile(
                name=profile,
                delay_ms=delay_ms,
                jitter_ms=jitter_ms,
                loss_rate=_clamp(loss_pct / 100.0, 0.0, 1.0),
                bandwidth_kbps=max(0.0, bw_mbps * 1000.0),
                description=desc,
            )
    raise SystemExit(f"[ERROR] 未在配置表中找到 profile={profile!r}: {config_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UDP 5G 链路仿真器（延迟/抖动/丢包/带宽）")

    p.add_argument("--listen-host", default="0.0.0.0", help="UDP 监听地址")
    p.add_argument("--listen-port", type=int, default=15551, help="UDP 监听端口（上游发送到这里）")
    p.add_argument("--forward-host", default="127.0.0.1", help="UDP 转发目标地址")
    p.add_argument("--forward-port", type=int, default=15550, help="UDP 转发目标端口（下游监听）")

    p.add_argument("--delay-ms", type=float, default=20.0, help="平均时延（ms）")
    p.add_argument("--jitter-ms", type=float, default=5.0, help="抖动（ms，作为正态分布 sigma）")
    p.add_argument("--loss-rate", type=float, default=0.005, help="丢包率（0-1）")
    p.add_argument("--bandwidth-kbps", type=float, default=50_000.0, help="带宽上限（kbps，0=不限制）")
    p.add_argument("--queue-limit", type=int, default=5000, help="转发队列上限（超过则丢弃）")
    p.add_argument("--profile", default=None, help="从配置表选择网络场景 profile（优先级高于手动参数）")
    p.add_argument("--config-csv", default=None, help="网络场景配置 CSV")

    p.add_argument("--metrics-interval-s", type=float, default=1.0, help="KPI 输出间隔（秒）")
    p.add_argument("--metrics-csv", default=None, help="KPI CSV 输出路径（可选）")
    p.add_argument("--seed", type=int, default=0, help="随机种子（0=使用当前时间）")

    return p.parse_args()


class TokenBucket:
    def __init__(self, rate_bytes_per_s: float, burst_bytes: float):
        self._rate = max(0.0, rate_bytes_per_s)
        self._burst = max(1.0, burst_bytes)
        self._tokens = self._burst
        self._last = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        dt = now - self._last
        self._last = now
        if self._rate <= 0:
            self._tokens = self._burst
            return
        self._tokens = min(self._burst, self._tokens + dt * self._rate)

    def can_take(self, n_bytes: int) -> bool:
        self._refill()
        return self._tokens >= n_bytes

    def take(self, n_bytes: int) -> None:
        self._refill()
        self._tokens = max(0.0, self._tokens - n_bytes)

    def seconds_until_available(self, n_bytes: int) -> float:
        self._refill()
        if self._rate <= 0:
            return 0.0
        if self._tokens >= n_bytes:
            return 0.0
        return (n_bytes - self._tokens) / self._rate


@dataclass
class Packet:
    enq_s: float
    due_s: float
    payload: bytes
    delay_ms: float


class LinkEmulator:
    def __init__(self, profile: LinkProfile, listen: Tuple[str, int], forward: Tuple[str, int], queue_limit: int):
        self._profile = profile
        self._listen = listen
        self._forward = forward
        self._queue: Deque[Packet] = deque()
        self._queue_limit = max(1, queue_limit)

        bw_kbps = max(0.0, profile.bandwidth_kbps)
        # token bucket: burst 200ms 的带宽
        rate_bytes_per_s = bw_kbps * 1000.0 / 8.0 if bw_kbps > 0 else 0.0
        burst_bytes = max(rate_bytes_per_s * 0.2, 1500.0)
        self._bucket = TokenBucket(rate_bytes_per_s=rate_bytes_per_s, burst_bytes=burst_bytes)

        # metrics
        self._rx_packets = 0
        self._tx_packets = 0
        self._drop_loss = 0
        self._drop_queue = 0
        self._bad_json = 0
        self._rx_bytes = 0
        self._tx_bytes = 0
        self._delay_samples_ms: Deque[float] = deque(maxlen=10_000)

        self._last_report_s = time.monotonic()
        self._last_report_rx_bytes = 0
        self._last_report_tx_bytes = 0
        self._last_report_rx_packets = 0
        self._last_report_tx_packets = 0
        self._last_report_drop = 0

    @property
    def profile(self) -> LinkProfile:
        return self._profile

    def _sample_delay_ms(self) -> float:
        base = self._profile.delay_ms
        jitter = max(0.0, self._profile.jitter_ms)
        if jitter <= 1e-9:
            return max(0.0, base)
        # 正态分布抖动
        return max(0.0, random.gauss(base, jitter))

    def _should_drop(self) -> bool:
        return random.random() < _clamp(self._profile.loss_rate, 0.0, 1.0)

    def _encode_payload(
        self, raw: bytes, delay_ms: float, net_seq: int
    ) -> Optional[bytes]:
        # 只对 JSON 进行增强；坏包直接丢弃（统计）
        try:
            row = json.loads(raw.decode("utf-8"))
        except Exception:
            self._bad_json += 1
            return None

        # Add transport metadata consumed by the monitoring runtime.
        row["net_profile"] = self._profile.name
        # 链路层序号在随机丢包前递增，下游据此统计 emulator 的真实丢包。
        # 上游业务 seq 可能因采样/融合策略跳变，不能用于链路 KPI。
        row["net_seq"] = int(net_seq)
        row["net_tx_ms"] = _now_ms()
        row["net_delay_ms"] = float(round(delay_ms, 3))
        return json.dumps(row, separators=(",", ":")).encode("utf-8")

    async def run(self, metrics_interval_s: float, metrics_csv: Optional[str]) -> None:
        loop = asyncio.get_running_loop()

        # use low-level sockets for asyncio sock_* APIs
        import socket as _socket

        sock_in = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        sock_in.bind(self._listen)
        sock_in.setblocking(False)

        sock_out = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        sock_out.connect(self._forward)
        sock_out.setblocking(False)

        writer = None
        csv_file = None
        if metrics_csv:
            os.makedirs(os.path.dirname(metrics_csv) or ".", exist_ok=True)
            csv_file = open(metrics_csv, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "t_s",
                    "profile",
                    "delay_ms",
                    "jitter_ms",
                    "loss_rate",
                    "bandwidth_kbps",
                    "rx_packets",
                    "tx_packets",
                    "rx_bytes",
                    "tx_bytes",
                    "drop_packets",
                    "drop_loss",
                    "drop_queue",
                    "drop_pct",
                    "bad_json",
                    "rx_kbps",
                    "tx_kbps",
                    "delay_avg_ms",
                    "delay_p95_ms",
                    "queue_len",
                ],
            )
            writer.writeheader()

        async def recv_loop() -> None:
            while True:
                data = await loop.sock_recv(sock_in, 65535)
                self._rx_packets += 1
                net_seq = self._rx_packets
                self._rx_bytes += len(data)

                if self._should_drop():
                    self._drop_loss += 1
                    continue

                delay_ms = self._sample_delay_ms()
                payload = self._encode_payload(data, delay_ms, net_seq)
                if payload is None:
                    continue

                if len(self._queue) >= self._queue_limit:
                    self._drop_queue += 1
                    continue

                enq_s = loop.time()
                due_s = enq_s + delay_ms / 1000.0
                self._queue.append(Packet(enq_s=enq_s, due_s=due_s, payload=payload, delay_ms=delay_ms))

        async def send_loop() -> None:
            while True:
                if not self._queue:
                    await asyncio.sleep(0.001)
                    continue

                pkt = self._queue[0]
                now_s = loop.time()
                if now_s < pkt.due_s:
                    await asyncio.sleep(min(0.05, pkt.due_s - now_s))
                    continue

                # bandwidth limit
                n = len(pkt.payload)
                wait_s = self._bucket.seconds_until_available(n) if self._profile.bandwidth_kbps > 0 else 0.0
                if wait_s > 0:
                    await asyncio.sleep(min(wait_s, 0.1))
                    continue

                if self._profile.bandwidth_kbps > 0:
                    self._bucket.take(n)

                self._queue.popleft()
                try:
                    actual_delay_ms = (loop.time() - pkt.enq_s) * 1000.0
                    row = json.loads(pkt.payload.decode("utf-8"))
                    row["net_delay_ms"] = round(actual_delay_ms, 3)
                    payload = json.dumps(row, separators=(",", ":")).encode("utf-8")
                    await loop.sock_sendall(sock_out, payload)
                    self._tx_packets += 1
                    self._tx_bytes += len(payload)
                    # 统计“实际链路时延”（从入队到发出），包含抖动/排队/带宽等待等影响
                    self._delay_samples_ms.append(actual_delay_ms)
                except OSError:
                    # 发送失败当作队列丢弃，不重试，避免阻塞
                    self._drop_queue += 1

        async def report_loop() -> None:
            while True:
                await asyncio.sleep(max(0.2, metrics_interval_s))

                now = time.monotonic()
                dt = max(1e-6, now - self._last_report_s)
                self._last_report_s = now

                rx_packets = self._rx_packets
                tx_packets = self._tx_packets
                drop_packets = self._drop_loss + self._drop_queue

                d_rx_packets = rx_packets - self._last_report_rx_packets
                d_tx_packets = tx_packets - self._last_report_tx_packets
                d_drop = drop_packets - self._last_report_drop
                d_rx_bytes = self._rx_bytes - self._last_report_rx_bytes
                d_tx_bytes = self._tx_bytes - self._last_report_tx_bytes

                self._last_report_rx_packets = rx_packets
                self._last_report_tx_packets = tx_packets
                self._last_report_drop = drop_packets
                self._last_report_rx_bytes = self._rx_bytes
                self._last_report_tx_bytes = self._tx_bytes

                rx_kbps = (d_rx_bytes * 8.0) / dt / 1000.0
                tx_kbps = (d_tx_bytes * 8.0) / dt / 1000.0

                delay_avg = 0.0
                delay_p95 = 0.0
                if self._delay_samples_ms:
                    samples = list(self._delay_samples_ms)
                    samples.sort()
                    delay_avg = sum(samples) / len(samples)
                    delay_p95 = samples[int(0.95 * (len(samples) - 1))]

                line = (
                    f"[net] profile={self._profile.name} "
                    f"tx={d_tx_packets}/s rx={d_rx_packets}/s drop={d_drop}/s "
                    f"rx_kbps={rx_kbps:.1f} tx_kbps={tx_kbps:.1f} "
                    f"delay_avg={delay_avg:.1f}ms p95={delay_p95:.1f}ms "
                    f"queue={len(self._queue)}"
                )
                print(line)

                if writer:
                    drop_pct = (drop_packets / rx_packets * 100.0) if rx_packets > 0 else 0.0
                    writer.writerow({
                        "t_s": round(now, 3),
                        "profile": self._profile.name,
                        "delay_ms": self._profile.delay_ms,
                        "jitter_ms": self._profile.jitter_ms,
                        "loss_rate": self._profile.loss_rate,
                        "bandwidth_kbps": self._profile.bandwidth_kbps,
                        "rx_packets": rx_packets,
                        "tx_packets": tx_packets,
                        "rx_bytes": self._rx_bytes,
                        "tx_bytes": self._tx_bytes,
                        "drop_packets": drop_packets,
                        "drop_loss": self._drop_loss,
                        "drop_queue": self._drop_queue,
                        "drop_pct": round(drop_pct, 3),
                        "bad_json": self._bad_json,
                        "rx_kbps": round(rx_kbps, 3),
                        "tx_kbps": round(tx_kbps, 3),
                        "delay_avg_ms": round(delay_avg, 3),
                        "delay_p95_ms": round(delay_p95, 3),
                        "queue_len": len(self._queue),
                    })
                    csv_file.flush()  # type: ignore[union-attr]

        try:
            await asyncio.gather(recv_loop(), send_loop(), report_loop())
        finally:
            try:
                sock_in.close()
            except Exception:
                pass
            try:
                sock_out.close()
            except Exception:
                pass
            if csv_file:
                csv_file.close()


def main() -> None:
    args = parse_args()

    if args.seed == 0:
        args.seed = int(time.time())
    random.seed(args.seed)

    if args.profile:
        if not args.config_csv:
            raise SystemExit("[ERROR] 使用 --profile 时必须同时提供 --config-csv")
        profile = load_profile(args.config_csv, args.profile)
    else:
        profile = LinkProfile(
            name="custom",
            delay_ms=float(args.delay_ms),
            jitter_ms=float(args.jitter_ms),
            loss_rate=_clamp(float(args.loss_rate), 0.0, 1.0),
            bandwidth_kbps=max(0.0, float(args.bandwidth_kbps)),
        )

    listen = (args.listen_host, int(args.listen_port))
    forward = (args.forward_host, int(args.forward_port))

    print("[net] UDP 5G 链路仿真器启动")
    print(f"[net] listen  : udp://{listen[0]}:{listen[1]}")
    print(f"[net] forward : udp://{forward[0]}:{forward[1]}")
    if profile.description:
        print(f"[net] desc    : {profile.description}")
    print(
        f"[net] profile={profile.name} delay={profile.delay_ms}ms jitter={profile.jitter_ms}ms "
        f"loss={profile.loss_rate*100:.3f}% bw={profile.bandwidth_kbps:.1f}kbps seed={args.seed}"
    )

    metrics_csv = args.metrics_csv
    if metrics_csv and metrics_csv.endswith(os.sep):
        metrics_csv = os.path.join(metrics_csv, f"net_{profile.name}_{int(time.time())}.csv")

    emulator = LinkEmulator(profile=profile, listen=listen, forward=forward, queue_limit=args.queue_limit)
    try:
        asyncio.run(emulator.run(metrics_interval_s=float(args.metrics_interval_s), metrics_csv=metrics_csv))
    except KeyboardInterrupt:
        print("\n[net] 已停止")


if __name__ == "__main__":
    main()
