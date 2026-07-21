from pathlib import Path
import tempfile
import types

from rfly_bridge import run_buaa_realtime_bridge

def test_load_official_module_from_autotest_dir_uses_explicit_file():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "db.json").write_text("{}", encoding="utf-8")
        (root / "PX4MavCtrlV4.py").write_text(
            "class PX4MavCtrler:\n"
            "    def __init__(self, test_id, port=20100, ip='127.0.0.1'):\n"
            "        self.test_id = test_id\n",
            encoding="utf-8",
        )

        mod = run_buaa_realtime_bridge.load_official_module(root, "PX4MavCtrlV4.py", "px4")

        assert hasattr(mod, "PX4MavCtrler")


def test_runtime_starts_bridge_after_hil_connect_and_stops_before_disconnect():
    events = []

    px4_module = types.SimpleNamespace(
        PX4MavCtrler=lambda test_id, port=20100, ip="127.0.0.1": types.SimpleNamespace(
            test_id=test_id,
            port=port,
            ip=ip,
        )
    )
    off_ctrl_module = types.SimpleNamespace(
        HIL_connect=lambda mav: events.append(("HIL_connect", mav.port)),
        HIL_disconnect=lambda mav: events.append(("HIL_disconnect", mav.port)),
    )
    bridge_runtime = types.SimpleNamespace(
        start=lambda: events.append("bridge_start"),
        stop=lambda: events.append("bridge_stop"),
    )

    runner = run_buaa_realtime_bridge.BuaaRealtimeBridgeRunner(
        px4_module=px4_module,
        off_ctrl_module=off_ctrl_module,
        bridge_runtime_factory=lambda mav: bridge_runtime,
    )

    mav = runner.start(unique_index=1419000188, port=20100)
    runner.stop()

    assert mav.port == 20100
    assert events == [
        ("HIL_connect", 20100),
        "bridge_start",
        "bridge_stop",
        ("HIL_disconnect", 20100),
    ]



def test_batch_session_reads_testbat_and_starts_only_mav_loop(tmp_path, monkeypatch):
    root = tmp_path / "platform"
    root.mkdir()
    (root / "bat").mkdir()
    (root / "AutoTestAPI").mkdir()
    (root / "AutoTestAPI" / "db.json").write_text(
        '{"testbat":"FaultModelSITL"}', encoding="utf-8"
    )
    (root / "bat" / "FaultModelSITL.bat").write_text(
        '@echo off\n'
        'SET UE4_MAP=NeighborhoodPark\n'
        'tasklist|find /i "RflySim3D.exe" || start %PSP_PATH%\\RflySim3D\\RflySim3D.exe\n',
        encoding="utf-8",
    )

    events = []

    class DummyMav:
        def __init__(self, test_id, port=20100, ip="127.0.0.1"):
            self.test_id = test_id
            self.port = port
            self.ip = ip
            self.hasHeartbeat = True

        def InitMavLoop(self, UDPMode=2):
            events.append(("InitMavLoop", UDPMode))

        def endMavLoop(self):
            events.append(("endMavLoop",))

        def initOffboard(self):
            events.append(("initOffboard",))

    px4_module = types.SimpleNamespace(PX4MavCtrler=DummyMav)

    launched = []

    def fake_popen(args, cwd=None, **kwargs):
        launched.append((args, cwd))
        return types.SimpleNamespace(
            pid=4321,
            terminate=lambda: events.append(("terminate",)),
            kill=lambda: events.append(("kill",)),
        )

    taskkill_calls = []

    def fake_run(args, **kwargs):
        taskkill_calls.append((args, kwargs))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_buaa_realtime_bridge.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_buaa_realtime_bridge.subprocess, "run", fake_run)
    monkeypatch.setattr(run_buaa_realtime_bridge.time, "sleep", lambda *_: None)

    session = run_buaa_realtime_bridge.BatchLaunchSession(
        platform_root=root,
        px4_module=px4_module,
        test_id=1419000188,
        port=20100,
        ip="127.0.0.1",
        startup_wait_s=0,
        enable_true_loop=False,
    )

    mav = session.start()
    session.stop()

    assert mav.port == 20100
    assert launched == [(["cmd.exe", "/k", str(root / "bat" / "FaultModelSITL.bat")], str(root))]
    assert events == [("InitMavLoop", 2), ("initOffboard",), ("endMavLoop",)]
    assert taskkill_calls[0][0] == ["taskkill", "/PID", "4321", "/T", "/F"]
    batch_text = (root / "bat" / "FaultModelSITL.bat").read_text(encoding="utf-8")
    assert "SET UE4_MAP=LowGPU" in batch_text
    assert "RflyChangeMapbyName-LowGPU" in batch_text


def test_batch_session_manual_control_never_initializes_offboard(tmp_path, monkeypatch):
    root = tmp_path / "platform"
    (root / "bat").mkdir(parents=True)
    (root / "AutoTestAPI").mkdir()
    (root / "AutoTestAPI" / "db.json").write_text(
        '{"testbat":"FaultModelSITL"}', encoding="utf-8"
    )
    batch_path = root / "bat" / "FaultModelSITL.bat"
    batch_path.write_text(
        "@echo off\n"
        "SET /a START_INDEX=1\n"
        "set DLLModel=FaultModel450d_new\n"
        "set PX4SitlFrame=quadrotor_x\n"
        "SET UE4_MAP=LowGPU\n"
        "SET UDPSIMMODE=2\n"
        "start /realtime CopterSim.exe 1 %cntr% %portNum% %DLLModel% %SimMode% %UE4_MAP% %IS_BROADCAST% %PosXX% %PosYY% %ORIGIN_YAW% 1 %UDPSIMMODE%\n",
        encoding="utf-8",
    )

    events = []

    class DummyMav:
        hasHeartbeat = True

        def __init__(self, *_args, **_kwargs):
            pass

        def InitMavLoop(self, udp_mode):
            events.append(("InitMavLoop", udp_mode))

        def initOffboard(self):
            events.append(("initOffboard",))

        def endMavLoop(self):
            events.append(("endMavLoop",))

    monkeypatch.setattr(
        run_buaa_realtime_bridge.subprocess,
        "Popen",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            pid=None, terminate=lambda: None, kill=lambda: None
        ),
    )

    session = run_buaa_realtime_bridge.BatchLaunchSession(
        platform_root=root,
        px4_module=types.SimpleNamespace(PX4MavCtrler=DummyMav),
        test_id=1419000188,
        startup_wait_s=0,
        init_offboard=False,
    )

    session.start()
    session.stop()

    assert events == [("InitMavLoop", 2), ("endMavLoop",)]
    batch_text = batch_path.read_text(encoding="utf-8")
    assert "SET /a CLASS_3D_ID=-1" in batch_text
    assert "set DLLModel=0" in batch_text
    assert "set PX4SitlFrame=iris" in batch_text
    assert "SET UDPSIMMODE=2" in batch_text
    assert (
        "start /realtime CopterSim.exe 1 %cntr% %CLASS_3D_ID% %DLLModel%"
        in batch_text
    )
    assert "CopterSim.exe 1 %cntr% %portNum%" not in batch_text


def test_parse_args_accepts_duration_and_startup_wait(monkeypatch):
    monkeypatch.setattr(
        run_buaa_realtime_bridge.sys,
        "argv",
        [
            "run_buaa_realtime_bridge.py",
            "--duration",
            "30",
            "--startup-wait",
            "1.5",
            "--ready-timeout",
            "12",
            "--manual-control",
            "--sim-map",
            "LowGPU",
            "--pymav-address",
            "udpin:0.0.0.0:14031",
        ],
    )

    args = run_buaa_realtime_bridge.parse_args()

    assert args.duration == 30
    assert args.startup_wait == 1.5
    assert args.ready_timeout == 12
    assert args.manual_control is True
    assert args.sim_map == "LowGPU"
    assert args.pymav_address == "udpin:0.0.0.0:14031"


def test_main_manual_control_does_not_load_or_start_case_controller(monkeypatch, tmp_path):
    autotest_dir = tmp_path / "AutoTestAPI"
    autotest_dir.mkdir()
    args = types.SimpleNamespace(
        autotest_dir=str(autotest_dir),
        unique_index=1419000188,
        udp_port=20100,
        ip="127.0.0.1",
        host="127.0.0.1",
        port=15550,
        emit_rate=50,
        rpm_scale=10000.0,
        pymav_address="udpin:0.0.0.0:14031",
        startup_wait=0.0,
        ready_timeout=15.0,
        manual_control=True,
        duration=1.0,
        sim_map="LowGPU",
        no_pymav=False,
    )
    loaded_modules = []
    session_options = []
    runner_controllers = []
    mav = object()

    def fake_load(_autotest_dir, filename, _tag):
        loaded_modules.append(filename)
        if filename == "command.py":
            raise AssertionError("manual mode must not load command.py")
        return types.SimpleNamespace()

    class FakeSession:
        def __init__(self, **kwargs):
            session_options.append(kwargs)

        def start(self):
            return mav

        def stop(self):
            pass

    class FakeRunner:
        def __init__(self, *, bridge_runtime_factory):
            self.bridge_runtime_factory = bridge_runtime_factory

        def start(self, *, mav, controller):
            runner_controllers.append(controller)

        def stop(self):
            pass

    monkeypatch.setattr(run_buaa_realtime_bridge, "parse_args", lambda: args)
    monkeypatch.setattr(run_buaa_realtime_bridge, "load_official_module", fake_load)
    monkeypatch.setattr(run_buaa_realtime_bridge, "BatchLaunchSession", FakeSession)
    monkeypatch.setattr(run_buaa_realtime_bridge, "BuaaRealtimeBridgeRunner", FakeRunner)
    monkeypatch.setattr(
        run_buaa_realtime_bridge,
        "BuaaCaseController",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("manual mode must not create a case controller")
        ),
    )
    timestamps = iter([0.0, 2.0])
    monkeypatch.setattr(run_buaa_realtime_bridge.time, "time", lambda: next(timestamps))

    run_buaa_realtime_bridge.main()

    assert loaded_modules == ["PX4MavCtrlV4.py"]
    assert session_options[0]["init_offboard"] is False
    assert runner_controllers == [None]



def test_case_controller_loads_control_sequence_by_unique_index_and_runs_once(tmp_path, monkeypatch):
    autotest_dir = tmp_path / "AutoTestAPI"
    autotest_dir.mkdir()
    (autotest_dir / "db.json").write_text(
        '{"faultcase":[{"CaseID":189,"UniqueIndex":1419000188,"ControlSequence":"2,1,0.5,1,15;1,1,5"}]}',
        encoding="utf-8",
    )

    events = []

    class DummyCmdCtrl:
        def __init__(self, mav, start_time):
            self.CID = {'1': object(), '2': object()}
            self.CID1obj = types.SimpleNamespace(isDone=1)
            self.CID2obj = types.SimpleNamespace(isDone=1, returnvalue=0)

        def FIDPro(self, cmdCID):
            return {
                '1': lambda param=None: events.append((cmdCID, '1', param)),
            }

    command_module = types.SimpleNamespace(CmdCtrl=DummyCmdCtrl)
    mav = types.SimpleNamespace(set_height=0, mav_height=0, set_forward_vel=0)

    monkeypatch.setattr(run_buaa_realtime_bridge.time, 'sleep', lambda *_: None)

    controller = run_buaa_realtime_bridge.BuaaCaseController(
        command_module=command_module,
        db_json_path=autotest_dir / 'db.json',
        mav=mav,
        unique_index=1419000188,
        loop_hz=50,
    )

    assert controller.control_sequence == ['2,1,0.5,1,15', '1,1,5']
