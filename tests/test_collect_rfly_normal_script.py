from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_collect_rfly_normal_uses_direct_bridge_and_quality_gate():
    script = (PROJECT_ROOT / "scripts" / "collect_rfly_normal.ps1").read_text(
        encoding="utf-8"
    )

    assert "start_rflysim_sitl_qgc.ps1" in script
    assert "run_rfly_live_bridge.py" in script
    assert "uav_fault.telemetry.receiver" in script
    assert '"--auto-stop", "5"' in script
    assert '"--require-ready-field", "pymav_highres_imu_seen"' in script
    assert '"--require-ready-field", "pymav_actuator_control_seen"' in script
    assert "15560" in script
    assert "14031" in script
    assert "Get-NetUDPEndpoint" in script
    assert "validate_rfly_normal.py" in script
    assert "trim_rfly_prearm.py" in script
    assert "--max-prearm-seconds 30" in script
    assert "--keep-prearm-seconds 20" in script
    assert "start_dashboard" not in script
