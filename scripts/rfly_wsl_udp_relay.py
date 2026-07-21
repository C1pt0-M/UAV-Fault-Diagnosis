from __future__ import annotations

import argparse
import selectors
import signal
import socket
import threading
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relay the RflySim MAVLink loopback link across WSL2 NAT."
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=16539)
    parser.add_argument("--copter-host", default="127.0.0.1")
    parser.add_argument("--copter-port", type=int, default=16540)
    parser.add_argument("--copter-source-host", default="127.0.0.1")
    parser.add_argument("--copter-source-port", type=int, default=17540)
    parser.add_argument("--wsl-host", required=True)
    parser.add_argument("--wsl-port", type=int, default=17540)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stop_event = threading.Event()

    def request_stop(*_args) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    px4_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    px4_socket.bind((args.listen_host, args.listen_port))
    px4_socket.setblocking(False)

    copter_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    copter_socket.bind((args.copter_source_host, args.copter_source_port))
    copter_socket.setblocking(False)

    selector = selectors.DefaultSelector()
    selector.register(px4_socket, selectors.EVENT_READ, "px4")
    selector.register(copter_socket, selectors.EVENT_READ, "copter")

    print(
        "[rfly-relay] "
        f"WSL->{args.listen_host}:{args.listen_port} "
        f"<-> {args.copter_host}:{args.copter_port} "
        f"<-> {args.wsl_host}:{args.wsl_port}",
        flush=True,
    )

    px4_packets = 0
    copter_packets = 0
    last_report = time.monotonic()
    try:
        while not stop_event.is_set():
            for key, _mask in selector.select(timeout=0.5):
                data, _source = key.fileobj.recvfrom(65535)
                if key.data == "px4":
                    copter_socket.sendto(data, (args.copter_host, args.copter_port))
                    px4_packets += 1
                else:
                    copter_socket.sendto(data, (args.wsl_host, args.wsl_port))
                    copter_packets += 1

            now = time.monotonic()
            if now - last_report >= 5.0:
                print(
                    f"[rfly-relay] px4_to_copter={px4_packets} "
                    f"copter_to_px4={copter_packets}",
                    flush=True,
                )
                last_report = now
    finally:
        selector.close()
        px4_socket.close()
        copter_socket.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
