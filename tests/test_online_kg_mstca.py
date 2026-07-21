import json
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from uav_fault.models.cnn.lstm import OnlineKGMSTCA
from uav_fault.models.train_online_detector import MODEL_REGISTRY, train
from uav_fault.online.fault_detector import FaultDetector


MODEL_COLUMNS = json.loads(
    (PROJECT_ROOT / "assets" / "models" / "online_detector_cnn1d_rfly" / "norm_stats.json").read_text(
        encoding="utf-8"
    )
)["columns"]


class Args:
    model = "kg_mstca"
    cache_dir = ""
    output_dir = ""
    epochs = 2
    batch_size = 4
    lr = 3e-4
    weight_decay = 1e-4
    gamma = 2.0
    hidden_size = 16
    num_layers = 1
    dropout = 0.2
    num_workers = 0
    patience = 1
    min_epochs = 1
    min_delta = 0.002
    warmup = 1
    seed = 42
    max_train_samples = 0


def _make_cache(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    norm_stats = {
        "mean": [0.0] * len(MODEL_COLUMNS),
        "std": [1.0] * len(MODEL_COLUMNS),
        "columns": MODEL_COLUMNS,
        "window_size": 150,
        "target_hz": 50.0,
        "stride": 10,
        "feature_schema": "rfly_seq_columns_v1",
    }
    (cache_dir / "norm_stats.json").write_text(
        json.dumps(norm_stats), encoding="utf-8"
    )

    def _save(name: str, labels: list[int]):
        x = torch.zeros((len(labels), len(MODEL_COLUMNS), 150), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.int64)
        torch.save({"x": x, "y": y}, cache_dir / f"{name}.pt")

    _save("train", [0, 0, 0, 1, 1, 1])
    _save("val", [0, 1, 0, 1])
    _save("test", [1, 0, 1, 0])


def _make_model_dir(model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    model = OnlineKGMSTCA(
        in_channels=len(MODEL_COLUMNS),
        hidden_size=16,
        num_layers=1,
        num_classes=2,
        dropout=0.2,
    )
    torch.save(model.state_dict(), model_dir / "best.pt")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model": "kg_mstca",
                "task": "online_binary",
                "feature_schema": "rfly_seq_columns_v1",
                "window_size": 150,
                "feature_count": len(MODEL_COLUMNS),
                "in_channels": len(MODEL_COLUMNS),
                "num_classes": 2,
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.2,
                "label_names": ["normal", "fault"],
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "norm_stats.json").write_text(
        json.dumps(
            {
                "mean": [0.0] * len(MODEL_COLUMNS),
                "std": [1.0] * len(MODEL_COLUMNS),
                "columns": MODEL_COLUMNS,
                "window_size": 150,
                "target_hz": 50.0,
                "stride": 10,
                "feature_schema": "rfly_seq_columns_v1",
            }
        ),
        encoding="utf-8",
    )


def test_kg_mstca_is_registered_and_outputs_binary_logits():
    assert "kg_mstca" in MODEL_REGISTRY

    model = MODEL_REGISTRY["kg_mstca"](
        in_channels=len(MODEL_COLUMNS),
        hidden_size=16,
        num_layers=1,
        num_classes=2,
        dropout=0.2,
    )
    x = torch.randn(3, len(MODEL_COLUMNS), 150)
    logits = model(x)

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_train_online_detector_supports_kg_mstca_smoke(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "model"
    _make_cache(cache_dir)

    args = Args()
    args.cache_dir = str(cache_dir)
    args.output_dir = str(output_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["model"] == "kg_mstca"
    assert config["num_classes"] == 2

    metrics_test = json.loads(
        (output_dir / "metrics_test.json").read_text(encoding="utf-8")
    )
    assert "f1_fault" in metrics_test


def test_fault_detector_can_load_kg_mstca_model_dir(tmp_path: Path):
    model_dir = tmp_path / "kg_mstca_model"
    _make_model_dir(model_dir)

    detector = FaultDetector(model_dir=str(model_dir), window_size=None)
    records = [{} for _ in range(150)]

    pred = detector.predict(records)

    assert pred is not None
    assert set(pred.keys()) == {"label", "confidence", "fault_prob", "raw_logits"}
    assert pred["label"] in {"normal", "fault"}
    assert 0.0 <= pred["confidence"] <= 1.0
    assert 0.0 <= pred["fault_prob"] <= 1.0
    assert len(pred["raw_logits"]) == 2
