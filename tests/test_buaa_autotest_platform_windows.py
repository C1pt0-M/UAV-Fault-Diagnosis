from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


pytestmark = pytest.mark.external


AUTOTEST_API_DIR = Path(r"D:\BuaaAutoTestPlatForm\AutoTestAPI")
MAVDB_PATH = AUTOTEST_API_DIR / "mavdb.py"
OFF_CTRL_PATH = AUTOTEST_API_DIR / "Off_ctrl_connect_copter.py"
PX4_MAVCTRL_PATH = AUTOTEST_API_DIR / "PX4MavCtrlV4.py"
BATCH_PATH = Path(r"D:\BuaaAutoTestPlatForm\bat\FaultModelSITL.bat")
SITL_MULTI_RUN_PATH = Path(r"D:\Rflysim\Firmware\Tools\sitl_multiple_run_rfly.sh")


def _load_mavdb_module():
    if not MAVDB_PATH.exists():
        pytest.skip(f"missing external platform file: {MAVDB_PATH}")

    sys.modules.setdefault("PX4MavCtrlV4", types.ModuleType("PX4MavCtrlV4"))

    spec = importlib.util.spec_from_file_location("buaa_autotest_mavdb", MAVDB_PATH)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_off_ctrl_module():
    if not OFF_CTRL_PATH.exists():
        pytest.skip(f"missing external platform file: {OFF_CTRL_PATH}")

    sys.modules.setdefault("PX4MavCtrlV4", types.ModuleType("PX4MavCtrlV4"))

    spec = importlib.util.spec_from_file_location("buaa_autotest_off_ctrl", OFF_CTRL_PATH)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_px4_mavctrl_module(monkeypatch):
    if not PX4_MAVCTRL_PATH.exists():
        pytest.skip(f"missing external platform file: {PX4_MAVCTRL_PATH}")

    pymavlink_module = types.ModuleType("pymavlink")
    mavutil_module = types.ModuleType("pymavlink.mavutil")
    mavutil_module.mavlink_connection = lambda *args, **kwargs: None
    pymavlink_module.mavutil = mavutil_module

    common_module = types.SimpleNamespace(
        MAV_FRAME_LOCAL_NED=1,
        MAV_FRAME_BODY_NED=8,
        MAV_CMD_COMPONENT_ARM_DISARM=400,
        MAV_PARAM_TYPE_INT32=6,
    )
    dialects_module = types.ModuleType("pymavlink.dialects")
    v20_module = types.ModuleType("pymavlink.dialects.v20")
    v20_module.common = common_module

    numpy_module = types.ModuleType("numpy")
    numpy_module.pi = 3.141592653589793
    numpy_module.array = lambda value, *args, **kwargs: value
    numpy_module.zeros = lambda shape, *args, **kwargs: [0] * (shape if isinstance(shape, int) else shape[0])

    utils_module = types.ModuleType("utils")
    utils_module.constrain_rad = lambda value: value
    utils_module.eulerAngleToMatrix = lambda angles: angles

    monkeypatch.setitem(sys.modules, "pymavlink", pymavlink_module)
    monkeypatch.setitem(sys.modules, "pymavlink.mavutil", mavutil_module)
    monkeypatch.setitem(sys.modules, "pymavlink.dialects", dialects_module)
    monkeypatch.setitem(sys.modules, "pymavlink.dialects.v20", v20_module)
    monkeypatch.setitem(sys.modules, "numpy", numpy_module)
    monkeypatch.setitem(sys.modules, "utils", utils_module)

    spec = importlib.util.spec_from_file_location("buaa_autotest_px4mavctrl", PX4_MAVCTRL_PATH)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_get_fault_case_id_supports_explicit_case_ids(tmp_path, monkeypatch):
    mavdb_module = _load_mavdb_module()

    config = {
        "testcase": "189,190",
        "faultcase": [
            {"CaseID": 189, "UniqueIndex": 1419000188},
            {"CaseID": 190, "UniqueIndex": 1419000189},
        ],
    }
    (tmp_path / "db.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])

    db_rows = [
        {"CaseID": 189, "UniqueIndex": 1419000188},
        {"CaseID": 190, "UniqueIndex": 1419000189},
    ]
    monkeypatch.setattr(mavdb_module.mavdb, "get_fault_case", lambda self: db_rows)

    instance = mavdb_module.mavdb.__new__(mavdb_module.mavdb)
    case_ids, unique_indices = mavdb_module.mavdb.get_fault_case_ID(instance)

    assert case_ids == [189, 190]
    assert unique_indices == [1419000188, 1419000189]


def test_data_api_init_tolerates_empty_px4_log_root(tmp_path, monkeypatch):
    mavdb_module = _load_mavdb_module()

    platform_path = tmp_path / "Rflysim"
    (platform_path / "Firmware" / "build" / "px4_sitl_default" / "instance_1" / "log").mkdir(
        parents=True
    )

    config = {"PlatFormPath": str(platform_path).replace("\\", "/")}
    (tmp_path / "db.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])
    monkeypatch.setattr(mavdb_module.DataAPI, "FolderCreat", lambda self: None)

    data_api = mavdb_module.DataAPI(189, 1419000188)

    assert data_api.CaseID == 189
    assert data_api.UniqueIndex == 1419000188


def test_log_data_extra_invokes_ulog2csv_via_subprocess(tmp_path, monkeypatch):
    mavdb_module = _load_mavdb_module()

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "case.ulg").write_text("ulg", encoding="utf-8")

    target_log_dir = tmp_path / "target" / "Log"
    target_log_dir.mkdir(parents=True)

    config = {"testdata": {"*vehicle_command_0.csv": []}}
    config_path = tmp_path / "db.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    invoked = []
    monkeypatch.setattr(
        mavdb_module,
        "subprocess",
        types.SimpleNamespace(run=lambda args, cwd, check: invoked.append((args, cwd, check))),
        raising=False,
    )
    monkeypatch.setattr(mavdb_module.os, "chdir", lambda _: None)

    data_api = mavdb_module.DataAPI.__new__(mavdb_module.DataAPI)
    data_api.log_path = str(tmp_path)
    data_api.TargetFilder_log = str(target_log_dir)
    data_api.jsonpath = str(config_path)

    monkeypatch.setattr(mavdb_module.DataAPI, "GetLatestLogDir", lambda self: str(source_dir))
    mavdb_module.DataAPI.LogDataExtra(data_api)

    assert invoked == [(["ulog2csv", "case.ulg"], str(target_log_dir), True)]


def test_log_data_extra_never_deletes_the_copied_ulg(tmp_path, monkeypatch):
    mavdb_module = _load_mavdb_module()

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "case.ulg").write_text("ulg", encoding="utf-8")

    target_log_dir = tmp_path / "target" / "Log"
    target_log_dir.mkdir(parents=True)

    config = {"testdata": {"*vehicle_command_0.csv": []}}
    config_path = tmp_path / "db.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    copied_files = []
    removed_files = []

    def fake_copyfile(src, dst):
        copied_files.append((src, dst))

    def fake_listdir(path):
        if Path(path) == source_dir:
            return ["case.ulg"]
        if Path(path) == target_log_dir:
            return ["case_vehicle_command_0.csv", "case.ulg"]
        return []

    monkeypatch.setattr(mavdb_module.shutil, "copyfile", fake_copyfile)
    monkeypatch.setattr(
        mavdb_module,
        "subprocess",
        types.SimpleNamespace(run=lambda args, cwd, check: None),
        raising=False,
    )
    monkeypatch.setattr(mavdb_module.os, "listdir", fake_listdir)
    monkeypatch.setattr(mavdb_module.os, "remove", lambda path: removed_files.append(path))

    data_api = mavdb_module.DataAPI.__new__(mavdb_module.DataAPI)
    data_api.log_path = str(tmp_path)
    data_api.TargetFilder_log = str(target_log_dir)
    data_api.jsonpath = str(config_path)

    monkeypatch.setattr(mavdb_module.DataAPI, "GetLatestLogDir", lambda self: str(source_dir))
    mavdb_module.DataAPI.LogDataExtra(data_api)

    assert [(Path(src), Path(dst)) for src, dst in copied_files] == [
        (source_dir / "case.ulg", target_log_dir / "case.ulg")
    ]
    assert not any(str(path).endswith(".ulg") for path in removed_files)


def test_hil_connect_launches_batch_with_startup_output_capture(tmp_path, monkeypatch):
    (tmp_path / "db.json").write_text(
        json.dumps({"testbat": "FaultModelSITL", "wsl_distro": "RflySim-20.04"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])

    off_ctrl_module = _load_off_ctrl_module()

    popen_calls = []

    class DummyChild:
        def __init__(self):
            self.stdout = None

        def terminate(self):
            return None

        def kill(self):
            return None

    def fake_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return DummyChild()

    class DummyMav:
        def __init__(self):
            self.the_connection = types.SimpleNamespace(target_system=1, target_component=1)

        def InitMavLoop(self):
            return None

        def initOffboard(self):
            return None

        def InitTrueDataLoop(self):
            return None

    monkeypatch.setattr(off_ctrl_module.os, "getcwd", lambda: str(tmp_path))
    monkeypatch.setattr(off_ctrl_module.time, "sleep", lambda *_: None)
    monkeypatch.setattr(off_ctrl_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        off_ctrl_module,
        "_wait_for_px4_ready",
        lambda mav, startup_event, timeout_s=90.0, poll_interval=0.5, px4_log_path=None: None,
        raising=False,
    )
    monkeypatch.setattr(
        off_ctrl_module,
        "_configure_direct_mavlink_endpoint",
        lambda mav: None,
        raising=False,
    )

    off_ctrl_module.path = str(tmp_path)
    off_ctrl_module.HIL_connect(DummyMav())

    assert len(popen_calls) == 1
    args, kwargs = popen_calls[0]
    assert args == (["cmd.exe", "/c", str(tmp_path / "bat" / "FaultModelSITL.bat")],)
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["shell"] is False
    assert kwargs["stdout"] == off_ctrl_module.subprocess.PIPE
    assert kwargs["stderr"] == off_ctrl_module.subprocess.STDOUT
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "ignore"
    assert kwargs["env"]["RFLY_WSL_DISTRO"] == "RflySim-20.04"


def test_is_px4_running_uses_configured_wsl_distro(tmp_path, monkeypatch):
    (tmp_path / "db.json").write_text(
        json.dumps({"testbat": "FaultModelSITL", "wsl_distro": "RflySim-20.04"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])

    off_ctrl_module = _load_off_ctrl_module()
    run_calls = []

    def fake_run(*args, **kwargs):
        run_calls.append((args, kwargs))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(off_ctrl_module.subprocess, "run", fake_run)

    assert off_ctrl_module._is_px4_running() is True
    assert run_calls == [
        (
            (
                ["wsl", "-d", "RflySim-20.04", "bash", "-lc", "pgrep -x px4 >/dev/null"],
            ),
            {
                "stdout": off_ctrl_module.subprocess.DEVNULL,
                "stderr": off_ctrl_module.subprocess.DEVNULL,
                "check": False,
            },
        )
    ]


def test_wait_for_px4_ready_polls_until_wsl_px4_and_true_data_arrive(tmp_path, monkeypatch):
    (tmp_path / "db.json").write_text(json.dumps({"testbat": "FaultModelSITL"}), encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])

    off_ctrl_module = _load_off_ctrl_module()

    class DummyMav:
        def __init__(self):
            self.mav_pos = None
            self.trueTimeStmp = 0
            self.heartbeatCount = 0

    mav = DummyMav()
    sleep_calls = []
    px4_running = {"value": False}

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) == 2:
            mav.mav_pos = [0.0, 0.0, 0.0]
        if len(sleep_calls) == 3:
            mav.trueTimeStmp = 456.0
            mav.heartbeatCount = 1
        if len(sleep_calls) == 4:
            px4_running["value"] = True

    time_values = iter([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    monkeypatch.setattr(off_ctrl_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(off_ctrl_module.time, "time", lambda: next(time_values))
    monkeypatch.setattr(
        off_ctrl_module,
        "_is_px4_running",
        lambda: px4_running["value"],
        raising=False,
    )

    class DummyEvent:
        def __init__(self):
            self._set = False

        def is_set(self):
            return self._set

        def set(self):
            self._set = True

    ready_event = DummyEvent()

    def fake_sleep_with_event(seconds):
        fake_sleep(seconds)
        if len(sleep_calls) == 4:
            ready_event.set()

    monkeypatch.setattr(off_ctrl_module.time, "sleep", fake_sleep_with_event)

    off_ctrl_module._wait_for_px4_ready(mav, ready_event, timeout_s=1.2, poll_interval=0.2)

    assert sleep_calls == [0.2, 0.2, 0.2, 0.2]


def test_wait_for_mavlink_target_requires_nonzero_target_ids(tmp_path, monkeypatch):
    (tmp_path / "db.json").write_text(json.dumps({"testbat": "FaultModelSITL"}), encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])

    off_ctrl_module = _load_off_ctrl_module()

    class DummyMav:
        def __init__(self):
            self.the_connection = types.SimpleNamespace(target_system=0, target_component=0)

    mav = DummyMav()
    sleep_calls = []
    time_values = iter([0.0, 0.2, 0.4, 0.6, 0.8])

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) == 2:
            mav.the_connection.target_system = 1
            mav.the_connection.target_component = 1

    monkeypatch.setattr(off_ctrl_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(off_ctrl_module.time, "time", lambda: next(time_values))

    off_ctrl_module._wait_for_mavlink_target(mav, timeout_s=0.8, poll_interval=0.2)

    assert sleep_calls == [0.2, 0.2]


def test_wait_for_mavlink_target_falls_back_to_single_vehicle_defaults(tmp_path, monkeypatch):
    (tmp_path / "db.json").write_text(json.dumps({"testbat": "FaultModelSITL"}), encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])

    off_ctrl_module = _load_off_ctrl_module()

    class DummyMav:
        def __init__(self):
            self.CopterID = 1
            self.the_connection = types.SimpleNamespace(target_system=0, target_component=0)

    mav = DummyMav()
    sleep_calls = []
    time_values = iter([0.0, 0.2, 0.4, 0.6])

    monkeypatch.setattr(off_ctrl_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(off_ctrl_module.time, "time", lambda: next(time_values))

    off_ctrl_module._wait_for_mavlink_target(mav, timeout_s=0.4, poll_interval=0.2)

    assert mav.the_connection.target_system == 1
    assert mav.the_connection.target_component == 1
    assert sleep_calls == [0.2]


def test_configure_direct_mavlink_endpoint_uses_wsl_ip_and_rfly_ports(tmp_path, monkeypatch):
    (tmp_path / "db.json").write_text(json.dumps({"testbat": "FaultModelSITL"}), encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])

    off_ctrl_module = _load_off_ctrl_module()

    class DummyMav:
        def __init__(self):
            self.CopterID = 1

    monkeypatch.setattr(
        off_ctrl_module,
        "_get_wsl_ipv4",
        lambda: "172.20.114.166",
        raising=False,
    )

    mav = DummyMav()
    off_ctrl_module._configure_direct_mavlink_endpoint(mav)

    assert mav.mavlink_listen_port == 14540
    assert mav.mavlink_send_port == 14580
    assert mav.mavlink_send_ip == "172.20.114.166"


def test_wait_for_px4_ready_requires_startup_marker(tmp_path, monkeypatch):
    (tmp_path / "db.json").write_text(json.dumps({"testbat": "FaultModelSITL"}), encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])

    off_ctrl_module = _load_off_ctrl_module()

    class DummyMav:
        def __init__(self):
            self.mav_pos = [0.0, 0.0, 0.0]
            self.trueTimeStmp = 123.0
            self.heartbeatCount = 0

    class DummyEvent:
        def __init__(self):
            self._set = False

        def is_set(self):
            return self._set

        def set(self):
            self._set = True

    ready_event = DummyEvent()
    sleep_calls = []
    px4_running = {"value": True}

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) == 2:
            ready_event.set()

    mav = DummyMav()
    time_values = iter([0.0, 0.2, 0.4, 0.6, 0.8])
    monkeypatch.setattr(off_ctrl_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(off_ctrl_module.time, "time", lambda: next(time_values))
    monkeypatch.setattr(
        off_ctrl_module,
        "_is_px4_running",
        lambda: px4_running["value"],
        raising=False,
    )

    off_ctrl_module._wait_for_px4_ready(mav, ready_event, timeout_s=1.0, poll_interval=0.2)

    assert sleep_calls == [0.2, 0.2]


def test_wait_for_px4_ready_requires_startup_complete_log(tmp_path, monkeypatch):
    (tmp_path / "db.json").write_text(json.dumps({"testbat": "FaultModelSITL"}), encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])

    off_ctrl_module = _load_off_ctrl_module()

    class DummyMav:
        def __init__(self):
            self.mav_pos = [0.0, 0.0, 0.0]
            self.trueTimeStmp = 123.0
            self.heartbeatCount = 0

    class DummyEvent:
        def is_set(self):
            return True

    px4_log_path = tmp_path / "out.log"
    px4_log_path.write_text(
        "PX4 SIM HOST: 172.20.112.1\nINFO  [simulator] Waiting for simulator to accept connection on TCP port 4560\n",
        encoding="utf-8",
    )

    sleep_calls = []
    time_values = iter([0.0, 0.2, 0.4, 0.6, 0.8])

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) == 2:
            px4_log_path.write_text(
                "PX4 SIM HOST: 172.20.112.1\n"
                "INFO  [simulator] Waiting for simulator to accept connection on TCP port 4560\n"
                "INFO  [simulator] Simulator connected on TCP port 4560\n"
                "INFO  [px4] Startup script returned successfully\n",
                encoding="utf-8",
            )

    mav = DummyMav()
    monkeypatch.setattr(off_ctrl_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(off_ctrl_module.time, "time", lambda: next(time_values))
    monkeypatch.setattr(off_ctrl_module, "_is_px4_running", lambda: True, raising=False)

    off_ctrl_module._wait_for_px4_ready(
        mav,
        DummyEvent(),
        timeout_s=0.8,
        poll_interval=0.2,
        px4_log_path=str(px4_log_path),
    )

    assert sleep_calls == [0.2, 0.2]


def test_wait_for_px4_ready_tolerates_missing_heartbeat(tmp_path, monkeypatch, capsys):
    (tmp_path / "db.json").write_text(json.dumps({"testbat": "FaultModelSITL"}), encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])

    off_ctrl_module = _load_off_ctrl_module()

    class DummyMav:
        def __init__(self):
            self.mav_pos = [0.0, 0.0, 0.0]
            self.trueTimeStmp = 123.0
            self.heartbeatCount = 0

    class DummyEvent:
        def is_set(self):
            return True

    mav = DummyMav()
    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(off_ctrl_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(off_ctrl_module, "_is_px4_running", lambda: True, raising=False)
    monkeypatch.setattr(
        off_ctrl_module,
        "_is_simulator_connected",
        lambda px4_log_path: True,
        raising=False,
    )
    monkeypatch.setattr(
        off_ctrl_module,
        "_is_px4_startup_complete",
        lambda px4_log_path: True,
        raising=False,
    )

    off_ctrl_module._wait_for_px4_ready(
        mav,
        DummyEvent(),
        timeout_s=0.8,
        poll_interval=0.2,
        px4_log_path=str(tmp_path / "out.log"),
    )

    assert sleep_calls == []
    assert 'no HEARTBEAT observed yet' in capsys.readouterr().out


def test_hil_connect_waits_for_px4_ready_before_returning(tmp_path, monkeypatch):
    (tmp_path / "db.json").write_text(json.dumps({"testbat": "FaultModelSITL"}), encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])

    off_ctrl_module = _load_off_ctrl_module()
    events = []

    class DummyChild:
        def __init__(self):
            self.stdout = None

        def terminate(self):
            return None

        def kill(self):
            return None

    class DummyMav:
        def __init__(self):
            self.the_connection = types.SimpleNamespace(target_system=1, target_component=1)

        def InitMavLoop(self):
            events.append("InitMavLoop")

        def initOffboard(self):
            events.append("initOffboard")

        def InitTrueDataLoop(self):
            events.append("InitTrueDataLoop")

    monkeypatch.setattr(off_ctrl_module.os, "getcwd", lambda: str(tmp_path))
    monkeypatch.setattr(off_ctrl_module.time, "sleep", lambda *_: None)
    monkeypatch.setattr(off_ctrl_module.subprocess, "Popen", lambda *args, **kwargs: DummyChild())
    monkeypatch.setattr(
        off_ctrl_module,
        "_wait_for_px4_ready",
        lambda mav, startup_event, timeout_s=90.0, poll_interval=0.5, px4_log_path=None: events.append("wait_ready"),
        raising=False,
    )
    monkeypatch.setattr(
        off_ctrl_module,
        "_wait_for_mavlink_target",
        lambda mav, timeout_s=30.0, poll_interval=0.2: events.append("wait_mavlink_target"),
        raising=False,
    )
    monkeypatch.setattr(
        off_ctrl_module,
        "_configure_direct_mavlink_endpoint",
        lambda mav: events.append("configure_direct_mavlink"),
        raising=False,
    )

    off_ctrl_module.path = str(tmp_path)
    off_ctrl_module.HIL_connect(DummyMav())

    assert events == ["configure_direct_mavlink", "InitMavLoop", "InitTrueDataLoop", "wait_ready", "wait_mavlink_target", "initOffboard"]


def test_hil_connect_initializes_offboard_only_after_px4_ready(tmp_path, monkeypatch):
    (tmp_path / "db.json").write_text(json.dumps({"testbat": "FaultModelSITL"}), encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])

    off_ctrl_module = _load_off_ctrl_module()
    events = []

    class DummyChild:
        def __init__(self):
            self.stdout = None

        def terminate(self):
            return None

        def kill(self):
            return None

    class DummyMav:
        def __init__(self):
            self.the_connection = types.SimpleNamespace(target_system=1, target_component=1)

        def InitMavLoop(self):
            events.append("InitMavLoop")

        def initOffboard(self):
            events.append("initOffboard")

        def InitTrueDataLoop(self):
            events.append("InitTrueDataLoop")

    def fake_wait_ready(*args, **kwargs):
        events.append("wait_ready")
        assert "initOffboard" not in events

    monkeypatch.setattr(off_ctrl_module.os, "getcwd", lambda: str(tmp_path))
    monkeypatch.setattr(off_ctrl_module.time, "sleep", lambda *_: None)
    monkeypatch.setattr(off_ctrl_module.subprocess, "Popen", lambda *args, **kwargs: DummyChild())
    monkeypatch.setattr(
        off_ctrl_module,
        "_wait_for_px4_ready",
        fake_wait_ready,
        raising=False,
    )
    monkeypatch.setattr(
        off_ctrl_module,
        "_wait_for_mavlink_target",
        lambda mav, timeout_s=30.0, poll_interval=0.2: events.append("wait_mavlink_target"),
        raising=False,
    )
    monkeypatch.setattr(
        off_ctrl_module,
        "_configure_direct_mavlink_endpoint",
        lambda mav: events.append("configure_direct_mavlink"),
        raising=False,
    )

    off_ctrl_module.path = str(tmp_path)
    off_ctrl_module.HIL_connect(DummyMav())

    assert events.index("wait_ready") < events.index("initOffboard")


def test_init_offboard_sets_plain_int_rc_loss_params_and_disables_rc_checks(monkeypatch):
    px4_module = _load_px4_mavctrl_module(monkeypatch)

    param_calls = []

    dummy = types.SimpleNamespace(
        UDPMode=2,
        EnList=None,
        type_mask=None,
        coordinate_frame=None,
        pos=None,
        vel=None,
        acc=None,
        yaw=None,
        yawrate=None,
        isInOffboard=False,
        isArmed=False,
        hasSendDisableRTLRC=False,
        f_index=0,
        f_id=0,
        f_mode=0,
        f_controls=[0] * 16,
        f_pg_time=0,
        f_sleep_status=0,
        f_flight_status=0,
        t2=types.SimpleNamespace(start=lambda: None),
        TypeMask=lambda en_list: 123,
        sendMavSetParam=lambda param_id, value, param_type: param_calls.append((param_id, value, param_type)),
        sendMavOffboardAPI=lambda *args: None,
        sendMavFaultInfoAPI=lambda *args: None,
        SendSetMode=lambda *args: None,
        SendMavCmdLong=lambda *args: None,
        sendMavManualCtrl=lambda *args: None,
    )

    monkeypatch.setattr(px4_module.time, "sleep", lambda *_: None)

    px4_module.PX4MavCtrler.initOffboard(dummy)

    assert ("NAV_RCL_ACT".encode(), 0, px4_module.mavlink2.MAV_PARAM_TYPE_INT32) in param_calls
    assert ("NAV_DLL_ACT".encode(), 0, px4_module.mavlink2.MAV_PARAM_TYPE_INT32) in param_calls
    assert ("COM_RCL_EXCEPT".encode(), 4, px4_module.mavlink2.MAV_PARAM_TYPE_INT32) in param_calls
    assert ("COM_RC_IN_MODE".encode(), 1, px4_module.mavlink2.MAV_PARAM_TYPE_INT32) in param_calls


def test_init_offboard_warms_up_setpoints_before_switching_and_arming(monkeypatch):
    px4_module = _load_px4_mavctrl_module(monkeypatch)

    events = []
    sleep_calls = []

    dummy = types.SimpleNamespace(
        UDPMode=2,
        EnList=None,
        type_mask=None,
        coordinate_frame=None,
        pos=None,
        vel=None,
        acc=None,
        yaw=None,
        yawrate=None,
        isInOffboard=False,
        isArmed=False,
        hasSendDisableRTLRC=False,
        f_index=0,
        f_id=0,
        f_mode=0,
        f_controls=[0] * 16,
        f_pg_time=0,
        f_sleep_status=0,
        f_flight_status=0,
        t2=types.SimpleNamespace(start=lambda: events.append("thread_started")),
        TypeMask=lambda en_list: 123,
        sendMavSetParam=lambda *args: events.append(("set_param", args[0])),
        sendMavOffboardAPI=lambda *args: events.append("offboard_msg"),
        sendMavFaultInfoAPI=lambda *args: events.append("fault_info"),
        sendMavManualCtrl=lambda *args: events.append("manual_control"),
        SendSetMode=lambda mainmode, cusmode=0: events.append(("set_mode", mainmode, cusmode)),
        SendMavCmdLong=lambda command, *args: events.append(("command", command)),
    )

    monkeypatch.setattr(px4_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    px4_module.PX4MavCtrler.initOffboard(dummy)

    offboard_mode_event = (
        "set_mode",
        px4_module.PX4_CUSTOM_MAIN_MODE.PX4_CUSTOM_MAIN_MODE_OFFBOARD,
        0,
    )
    arm_event = ("command", px4_module.mavlink2.MAV_CMD_COMPONENT_ARM_DISARM)

    assert offboard_mode_event in events
    assert arm_event in events
    assert events.index(offboard_mode_event) < events.index(arm_event)
    assert any(seconds >= 1.0 for seconds in sleep_calls)
    assert "manual_control" not in events


def test_sitl_multiple_run_rfly_sets_partner_ip_for_offboard_link():
    if not SITL_MULTI_RUN_PATH.exists():
        pytest.skip(f"missing external script: {SITL_MULTI_RUN_PATH}")

    text = SITL_MULTI_RUN_PATH.read_text(encoding="utf-8", errors="ignore")

    assert (
        "sed s/'mavlink start -x -u $udp_offboard_port_local -r 4000000 -f -m onboard -o $udp_offboard_port_remote'/'mavlink start -x -u $udp_offboard_port_local -r 4000000 -f -m onboard -o $udp_offboard_port_remote -t $PX4_SIM_HOST_ADDR'/"
        in text
    )
    assert "sed s/'mavlink start -x -u $udp_offboard_port_local'/'# mavlink start -x -u $udp_offboard_port_local'/" not in text
    assert "sed s/'mavlink start -x -u $udp_gcs_port_local -r 4000000'/'mavlink start -x -u $udp_offboard_port_local -r 4000000 -o $udp_offboard_port_remote -t $PX4_SIM_HOST_ADDR'/" not in text


def test_faultmodel_sitl_batch_does_not_auto_kill_simulators_after_startup():
    if not BATCH_PATH.exists():
        pytest.skip(f"missing external batch file: {BATCH_PATH}")

    text = BATCH_PATH.read_text(encoding="utf-8", errors="ignore")
    _, tail = text.split("SET PATH=%WINDOWSPATH%", 1)

    assert 'taskkill /im "CopterSim.exe"' not in tail
    assert 'taskkill /f /im "QGroundControl.exe"' not in tail
    assert 'taskkill /f /im "RflySim3D.exe"' not in tail


def test_faultmodel_sitl_batch_sets_wsl_distro_and_px4_sim_host_addr():
    if not BATCH_PATH.exists():
        pytest.skip(f"missing external batch file: {BATCH_PATH}")

    text = BATCH_PATH.read_text(encoding="utf-8", errors="ignore")

    assert "RFLY_WSL_DISTRO" in text
    assert "wsl -d %WSL_DISTRO% bash -lc" in text
    assert "PX4_SIM_HOST_ADDR=$(ip route show default" in text
    assert 'export PATH=\\"$HOME/ninja:$HOME/gcc-arm-none-eabi-7-2017-q4-major/bin:$PATH\\"' in text
