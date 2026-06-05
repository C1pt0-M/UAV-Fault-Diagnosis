from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any
import sys


def load_px4mavctrl_module(path: str | Path) -> ModuleType:
    module_path = Path(path)
    parent = str(module_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(
        f"rfly_px4mavctrl_{module_path.stem}", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialRflyAttachSession:
    """Attach to a running official Rfly environment as a read-only listener."""

    def __init__(
        self,
        *,
        px4mavctrl_module: ModuleType,
        test_id: int = 1,
        port: int = 20100,
        ip: str = "127.0.0.1",
        udp_mode: int = 2,
    ) -> None:
        self.module = px4mavctrl_module
        self.test_id = int(test_id)
        self.port = int(port)
        self.ip = str(ip)
        self.udp_mode = int(udp_mode)
        self.mav: Any | None = None
        self._mav_started = False
        self._true_started = False

    def start(self):
        self.mav = self.module.PX4MavCtrler(self.test_id, self.port, self.ip)
        self.mav.InitMavLoop(self.udp_mode)
        self._mav_started = True
        self.mav.InitTrueDataLoop()
        self._true_started = True
        return self.mav

    def stop(self) -> None:
        if self.mav is None:
            return
        if self._true_started and hasattr(self.mav, "EndTrueDataLoop"):
            self.mav.EndTrueDataLoop()
            self._true_started = False
        if self._mav_started and hasattr(self.mav, "endMavLoop"):
            self.mav.endMavLoop()
            self._mav_started = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach to official Rfly PX4MavCtrlV4")
    parser.add_argument("--px4mavctrl", required=True, help="Path to official PX4MavCtrlV4.py")
    parser.add_argument("--test-id", type=int, default=1)
    parser.add_argument("--port", type=int, default=20100)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--udp-mode", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_px4mavctrl_module(args.px4mavctrl)
    session = OfficialRflyAttachSession(
        px4mavctrl_module=module,
        test_id=args.test_id,
        port=args.port,
        ip=args.ip,
        udp_mode=args.udp_mode,
    )
    try:
        session.start()
        print("official_rfly_attach_ready")
        while True:
            pass
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()


if __name__ == "__main__":
    main()
