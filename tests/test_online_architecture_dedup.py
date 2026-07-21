from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAULT_DETECTOR_PATH = (
    PROJECT_ROOT / "packages" / "uav_fault" / "uav_fault" / "online" / "fault_detector.py"
)


def test_fault_detector_reuses_training_online_architecture_source():
    source = FAULT_DETECTOR_PATH.read_text(encoding="utf-8")

    assert (
        "from uav_fault.models.cnn.lstm import ("
        in source
        or "from cnn.lstm import ("
        in source
    )
    assert "class _OnlineBiLSTM" not in source
    assert "class _OnlineCNN1D" not in source
    assert "class _OnlineTransformer" not in source
    assert "class _OnlineMSTCA" not in source
