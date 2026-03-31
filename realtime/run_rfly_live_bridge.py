from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from rfly_live_attach import OfficialRflyAttachSession, load_px4mavctrl_module
from rfly_live_bridge import PymavStateSource, RflyLiveBridge, RflyLiveEmitter


class RflyBridgeRuntime:
    def __init__(
        self,
        *,
        attach_session: OfficialRflyAttachSession,
        emitter: RflyLiveEmitter,
        pymav_source: PymavStateSource | None = None,
    ) -> None:
        self.attach_session = attach_session
        self.emitter = emitter
        self.pymav_source = pymav_source
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.attach_session.start()
        if self.pymav_source is not None:
            self.pymav_source.start()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.emitter.run,
            args=(self._stop_event,),
            daemon=True,
            name="rfly-live-emitter",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self.pymav_source is not None:
            self.pymav_source.stop()
        self.attach_session.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attached Rfly live bridge")
    parser.add_argument("--px4mavctrl", required=True, help="Path to official PX4MavCtrlV4.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15550)
    parser.add_argument("--emit-rate", type=int, default=50)
    parser.add_argument("--rpm-scale", type=float, default=10000.0)
    parser.add_argument("--test-id", type=int, default=1)
    parser.add_argument("--udp-port", type=int, default=20100)
    parser.add_argument("--udp-mode", type=int, default=2)
    parser.add_argument("--mav-ip", default="127.0.0.1")
    parser.add_argument("--pymav-address", default="udpin:0.0.0.0:14030")
    parser.add_argument("--no-pymav", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_px4mavctrl_module(Path(args.px4mavctrl))
    attach_session = OfficialRflyAttachSession(
        px4mavctrl_module=module,
        test_id=args.test_id,
        port=args.udp_port,
        ip=args.mav_ip,
        udp_mode=args.udp_mode,
    )
    bridge = RflyLiveBridge(
        forward_host=args.host,
        forward_port=args.port,
        emit_rate_hz=args.emit_rate,
        rpm_scale=args.rpm_scale,
    )
    pymav_source = None
    if not args.no_pymav:
        pymav_source = PymavStateSource(pymav_address=args.pymav_address)
    emitter = RflyLiveEmitter(
        bridge=bridge,
        mav_provider=lambda: attach_session.mav,
        pymav_source=pymav_source,
    )
    runtime = RflyBridgeRuntime(
        attach_session=attach_session,
        emitter=emitter,
        pymav_source=pymav_source,
    )
    try:
        runtime.start()
        print("rfly_live_bridge_ready")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()
        bridge.close()


if __name__ == "__main__":
    main()
