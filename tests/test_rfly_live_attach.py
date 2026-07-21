from pathlib import Path

from rfly_bridge import rfly_live_attach


def test_official_module_loader_imports_px4mavctrl_from_explicit_path(tmp_path):
    (tmp_path / "utils.py").write_text(
        "def constrain_rad(value):\n"
        "    return value\n"
        "def eulerAngleToMatrix(angles):\n"
        "    return angles\n",
        encoding="utf-8",
    )
    mod_path = tmp_path / "PX4MavCtrlV4.py"
    mod_path.write_text(
        "from utils import constrain_rad\n"
        "class PX4MavCtrler:\n"
        "    def __init__(self, test_id, port=20100, ip='127.0.0.1'):\n"
        "        self.test_id = test_id\n"
        "        self.port = port\n",
        encoding="utf-8",
    )

    loaded = rfly_live_attach.load_px4mavctrl_module(mod_path)

    assert hasattr(loaded, "PX4MavCtrler")
    assert loaded.constrain_rad(1.0) == 1.0


def test_attach_session_starts_mav_and_true_data_loops_without_offboard(tmp_path):
    mod_path = tmp_path / "PX4MavCtrlV4.py"
    mod_path.write_text(
        "events = []\n"
        "class PX4MavCtrler:\n"
        "    def __init__(self, test_id, port=20100, ip='127.0.0.1'):\n"
        "        self.test_id = test_id\n"
        "        self.port = port\n"
        "        self.ip = ip\n"
        "    def InitMavLoop(self, UDPMode=2):\n"
        "        events.append(('InitMavLoop', UDPMode))\n"
        "    def InitTrueDataLoop(self):\n"
        "        events.append(('InitTrueDataLoop',))\n"
        "    def initOffboard(self):\n"
        "        events.append(('initOffboard',))\n"
        "    def endMavLoop(self):\n"
        "        events.append(('endMavLoop',))\n"
        "    def EndTrueDataLoop(self):\n"
        "        events.append(('EndTrueDataLoop',))\n",
        encoding="utf-8",
    )

    loaded = rfly_live_attach.load_px4mavctrl_module(mod_path)
    session = rfly_live_attach.OfficialRflyAttachSession(
        px4mavctrl_module=loaded,
        test_id=1,
        port=20100,
    )

    mav = session.start()

    assert mav.port == 20100
    assert loaded.events == [("InitMavLoop", 2), ("InitTrueDataLoop",)]


def test_attach_session_uses_keyword_args_for_current_rfly_sdk_signature(tmp_path):
    mod_path = tmp_path / "PX4MavCtrlV4.py"
    mod_path.write_text(
        "events = []\n"
        "class PX4MavCtrler:\n"
        "    def __init__(self, ID=1, ip='127.0.0.1', Com='udp', port=0):\n"
        "        self.ID = ID\n"
        "        self.ip = ip\n"
        "        self.Com = Com\n"
        "        self.port = port\n"
        "        events.append(('ctor', ID, ip, Com, port))\n"
        "    def InitMavLoop(self, UDPMode=2):\n"
        "        events.append(('InitMavLoop', UDPMode))\n"
        "    def InitTrueDataLoop(self):\n"
        "        events.append(('InitTrueDataLoop',))\n"
        "    def endMavLoop(self):\n"
        "        events.append(('endMavLoop',))\n"
        "    def EndTrueDataLoop(self):\n"
        "        events.append(('EndTrueDataLoop',))\n",
        encoding="utf-8",
    )

    loaded = rfly_live_attach.load_px4mavctrl_module(mod_path)
    session = rfly_live_attach.OfficialRflyAttachSession(
        px4mavctrl_module=loaded,
        test_id=1,
        port=20100,
        ip="127.0.0.1",
        udp_mode=2,
    )

    mav = session.start()

    assert mav.ID == 1
    assert mav.ip == "127.0.0.1"
    assert mav.Com == "udp"
    assert mav.port == 20100
    assert loaded.events == [
        ("ctor", 1, "127.0.0.1", "udp", 20100),
        ("InitMavLoop", 2),
        ("InitTrueDataLoop",),
    ]


def test_attach_session_stop_closes_true_data_before_mav_loop(tmp_path):
    mod_path = tmp_path / "PX4MavCtrlV4.py"
    mod_path.write_text(
        "events = []\n"
        "class PX4MavCtrler:\n"
        "    def __init__(self, test_id, port=20100, ip='127.0.0.1'):\n"
        "        pass\n"
        "    def InitMavLoop(self, UDPMode=2):\n"
        "        events.append(('InitMavLoop', UDPMode))\n"
        "    def InitTrueDataLoop(self):\n"
        "        events.append(('InitTrueDataLoop',))\n"
        "    def endMavLoop(self):\n"
        "        events.append(('endMavLoop',))\n"
        "    def EndTrueDataLoop(self):\n"
        "        events.append(('EndTrueDataLoop',))\n",
        encoding="utf-8",
    )

    loaded = rfly_live_attach.load_px4mavctrl_module(mod_path)
    session = rfly_live_attach.OfficialRflyAttachSession(
        px4mavctrl_module=loaded,
        test_id=1,
        port=20100,
    )
    session.start()
    session.stop()

    assert loaded.events == [
        ("InitMavLoop", 2),
        ("InitTrueDataLoop",),
        ("EndTrueDataLoop",),
        ("endMavLoop",),
    ]
