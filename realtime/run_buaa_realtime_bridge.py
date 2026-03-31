from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from types import ModuleType

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[0]
SRC_ROOT = PROJECT_ROOT / "src"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rfly_live_bridge import PymavStateSource, RflyLiveBridge, RflyLiveEmitter


def load_official_module(autotest_dir: str | Path, filename: str, tag: str) -> ModuleType:
    autotest_dir = Path(autotest_dir)
    path = autotest_dir / filename
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    sys.modules.setdefault("imp", types.ModuleType("imp"))
    old_cwd = os.getcwd()
    old_sys0 = sys.path[0] if sys.path else None
    old_px4 = sys.modules.get("PX4MavCtrlV4")
    if sys.path:
        sys.path[0] = str(autotest_dir)
    else:
        sys.path.insert(0, str(autotest_dir))
    os.chdir(str(autotest_dir.parent))
    spec = importlib.util.spec_from_file_location(f"buaa_{tag}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load official module: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        if filename == 'PX4MavCtrlV4.py':
            sys.modules['PX4MavCtrlV4'] = module
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)
        if old_sys0 is not None and sys.path:
            sys.path[0] = old_sys0
        if filename == 'PX4MavCtrlV4.py':
            if old_px4 is not None:
                sys.modules['PX4MavCtrlV4'] = old_px4
            else:
                sys.modules.pop('PX4MavCtrlV4', None)


class BatchLaunchSession:
    def __init__(self, *, platform_root: str | Path, px4_module: ModuleType, test_id: int, port: int = 20100, ip: str = "127.0.0.1", udp_mode: int = 2, startup_wait_s: float = 25.0, enable_true_loop: bool = False) -> None:
        self.platform_root = Path(platform_root)
        self.px4_module = px4_module
        self.test_id = int(test_id)
        self.port = int(port)
        self.ip = str(ip)
        self.udp_mode = int(udp_mode)
        self.startup_wait_s = float(startup_wait_s)
        self.enable_true_loop = bool(enable_true_loop)
        self.mav = None
        self.proc = None
        self._mav_started = False
        self._true_started = False

    def _resolve_batch_path(self) -> Path:
        db_path = self.platform_root / 'AutoTestAPI' / 'db.json'
        data = json.loads(db_path.read_text(encoding='utf-8'))
        testbat = data['testbat']
        return self.platform_root / 'bat' / f'{testbat}.bat'

    def start(self):
        batch_path = self._resolve_batch_path()
        self.proc = subprocess.Popen(['cmd.exe', '/k', str(batch_path)], cwd=str(self.platform_root))
        if self.startup_wait_s > 0:
            time.sleep(self.startup_wait_s)
        self.mav = self.px4_module.PX4MavCtrler(self.test_id, self.port, self.ip)
        self.mav.InitMavLoop(self.udp_mode)
        self._mav_started = True
        if self.enable_true_loop and hasattr(self.mav, 'InitTrueDataLoop'):
            self.mav.InitTrueDataLoop()
            self._true_started = True
        return self.mav

    def stop(self) -> None:
        if self.mav is not None:
            if self._true_started and hasattr(self.mav, 'EndTrueDataLoop'):
                self.mav.EndTrueDataLoop()
                self._true_started = False
            if self._mav_started and hasattr(self.mav, 'endMavLoop'):
                self.mav.endMavLoop()
                self._mav_started = False
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.kill()
            except Exception:
                pass


class BuaaCaseController:
    def __init__(self, *, command_module: ModuleType, db_json_path: str | Path, mav, unique_index: int, loop_hz: int = 250):
        self.command_module = command_module
        self.db_json_path = Path(db_json_path)
        self.mav = mav
        self.unique_index = int(unique_index)
        self.loop_hz = int(loop_hz)
        self.control_sequence = self._load_control_sequence()
        self.CFID = None
        self._stop_event = threading.Event()
        self._thread = None

    def _load_control_sequence(self) -> list[str]:
        data = json.loads(self.db_json_path.read_text(encoding='utf-8'))
        for case in data.get('faultcase', []):
            if int(case.get('UniqueIndex', -1)) == self.unique_index:
                seq = str(case.get('ControlSequence', ''))
                return [item for item in seq.split(';') if item]
        raise RuntimeError(f'UniqueIndex not found in db.json: {self.unique_index}')

    @staticmethod
    def _parse_numbers(ctrlseq: str) -> list[str]:
        return re.findall(r'-?\d+\.?[0-9]*', ctrlseq)

    def _do_cmd(self, ctrlseq: str) -> None:
        cmdseq = self._parse_numbers(ctrlseq)
        cmdCID = cmdseq[0]
        if cmdCID not in self.CFID.CID:
            raise RuntimeError(f'Unsupported command CID: {cmdCID}')
        FID = self.CFID.FIDPro(cmdCID)
        if len(cmdseq) > 2:
            param = [float(val) for val in cmdseq[2:]]
            FID[cmdseq[1]](param)
        else:
            FID[cmdseq[1]]()

    def _run_loop(self) -> None:
        start_time = time.time()
        self.CFID = self.command_module.CmdCtrl(self.mav, start_time)
        cid1 = self.CFID.CID1obj
        cid2 = self.CFID.CID2obj
        i = 0
        seqlen = len(self.control_sequence)
        takeoff_done = False
        first_flag = True
        last_time = time.time()
        while not self._stop_event.is_set() and i < seqlen:
            last_time = last_time + (1.0 / max(self.loop_hz, 1))
            sleep_time = last_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                last_time = time.time()

            if takeoff_done or first_flag:
                self._do_cmd(self.control_sequence[i])
                first_flag = False

            mav_height = float(getattr(self.mav, 'mav_height', 0.0) or 0.0)
            set_height = float(getattr(self.mav, 'set_height', 0.0) or 0.0)
            error = set_height - mav_height
            vz_err = 1.1 * error
            back = getattr(cid2, 'returnvalue', 0)
            if back != 0:
                if back[0] == 1:
                    param = back[1]
                    if not takeoff_done:
                        self.mav.SendVelFLU(0, 0, param[1])
                        takeoff_done = mav_height > param[2] * 0.95
                    else:
                        self.mav.SendVelFLU(0, 0, vz_err, 0)
                elif back[0] == 10:
                    param = back[1]
                    now_time = time.time()
                    if (now_time - param[2]) < param[1]:
                        vx_s = getattr(self.mav, 'set_forward_vel', 0.0) + (now_time - param[2]) * param[0]
                    else:
                        vx_s = getattr(self.mav, 'set_forward_vel', 0.0) + param[1] * param[0]
                    self.mav.SendVelFLU(vx=vx_s, vy=0, vz=vz_err, yawrate=0)

            cur = self.control_sequence[i]
            cur_type = self._parse_numbers(cur)[0]
            if (cur_type == '1' and cid1.isDone == 1) or (cur_type == '2' and cid2.isDone == 1 and (takeoff_done or self._parse_numbers(cur)[1] != '1')):
                i += 1

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name='buaa-case-controller')
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


class BuaaRealtimeBridgeRunner:
    def __init__(self, *, bridge_runtime_factory, px4_module=None, off_ctrl_module=None):
        self.bridge_runtime_factory = bridge_runtime_factory
        self.px4_module = px4_module
        self.off_ctrl_module = off_ctrl_module
        self.mav = None
        self.bridge_runtime = None
        self.controller = None

    def start(self, *, mav=None, controller=None, unique_index=None, port=20100, ip='127.0.0.1'):
        if mav is None:
            if self.px4_module is None or self.off_ctrl_module is None:
                raise RuntimeError('mav or official modules required')
            self.mav = self.px4_module.PX4MavCtrler(unique_index, port, ip)
            self.off_ctrl_module.HIL_connect(self.mav)
        else:
            self.mav = mav
        self.controller = controller
        if self.controller is not None:
            self.controller.start()
        self.bridge_runtime = self.bridge_runtime_factory(self.mav)
        self.bridge_runtime.start()
        return self.mav

    def stop(self) -> None:
        if self.bridge_runtime is not None:
            self.bridge_runtime.stop()
        if self.controller is not None:
            self.controller.stop()
        if self.off_ctrl_module is not None and self.mav is not None and self.px4_module is not None:
            self.off_ctrl_module.HIL_disconnect(self.mav)


class _BridgeRuntimeAdapter:
    def __init__(self, runtime):
        self.runtime = runtime

    def start(self) -> None:
        self.runtime.start()

    def stop(self) -> None:
        self.runtime.stop()


class _InProcessRuntime:
    def __init__(self, mav, *, host: str, port: int, emit_rate_hz: int, rpm_scale: float, pymav_address: str, no_pymav: bool):
        self.bridge = RflyLiveBridge(
            forward_host=host,
            forward_port=port,
            emit_rate_hz=emit_rate_hz,
            rpm_scale=rpm_scale,
        )
        self.pymav_source = None if no_pymav else PymavStateSource(pymav_address=pymav_address)
        self.emitter = RflyLiveEmitter(
            bridge=self.bridge,
            mav_provider=lambda: mav,
            pymav_source=self.pymav_source,
        )
        self._stop_event = None
        self._thread = None

    def start(self) -> None:
        self._stop_event = threading.Event()
        if self.pymav_source is not None:
            self.pymav_source.start()
        self._thread = threading.Thread(target=self.emitter.run, args=(self._stop_event,), daemon=True, name='buaa-live-emitter')
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self.pymav_source is not None:
            self.pymav_source.stop()
        self.bridge.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime bridge inside official BuaaAutoTest process")
    parser.add_argument("--autotest-dir", default=r"D:\BuaaAutoTestPlatForm\AutoTestAPI")
    parser.add_argument("--unique-index", type=int, default=1419000188)
    parser.add_argument("--udp-port", type=int, default=20100)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15550)
    parser.add_argument("--emit-rate", type=int, default=50)
    parser.add_argument("--rpm-scale", type=float, default=10000.0)
    parser.add_argument("--pymav-address", default="udpin:0.0.0.0:14030")
    parser.add_argument("--no-pymav", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    autotest_dir = Path(args.autotest_dir)
    px4_module = load_official_module(autotest_dir, 'PX4MavCtrlV4.py', 'px4')
    command_module = load_official_module(autotest_dir, 'command.py', 'command')
    platform_root = autotest_dir.parent
    session = BatchLaunchSession(
        platform_root=platform_root,
        px4_module=px4_module,
        test_id=args.unique_index,
        port=args.udp_port,
        ip=args.ip,
        udp_mode=2,
        startup_wait_s=25.0,
        enable_true_loop=False,
    )

    bridge_runtime = lambda mav: _BridgeRuntimeAdapter(
        _InProcessRuntime(
            mav,
            host=args.host,
            port=args.port,
            emit_rate_hz=args.emit_rate,
            rpm_scale=args.rpm_scale,
            pymav_address=args.pymav_address,
            no_pymav=args.no_pymav,
        )
    )
    runner = BuaaRealtimeBridgeRunner(bridge_runtime_factory=bridge_runtime)

    try:
        mav = session.start()
        controller = BuaaCaseController(
            command_module=command_module,
            db_json_path=autotest_dir / 'db.json',
            mav=mav,
            unique_index=args.unique_index,
            loop_hz=100,
        )
        runner.start(mav=mav, controller=controller)
        print('buaa_realtime_bridge_ready')
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        session.stop()


if __name__ == '__main__':
    main()
