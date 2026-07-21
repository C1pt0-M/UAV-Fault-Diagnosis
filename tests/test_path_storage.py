from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
from uav_dashboard_api import create_app
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.utils.path_store import (
    REPO_PREFIX,
    RUNTIME_PREFIX,
    resolve_local_path,
    serialize_local_path,
)
from uav_dashboard_api.config import TestingConfig


class PathStorageTestingConfig(TestingConfig):
    REPO_ROOT = str(PROJECT_ROOT)
    RUNTIME_DIR = PROJECT_ROOT / "apps" / "dashboard" / "backend" / "runtime_test"
    UPLOAD_DIR = RUNTIME_DIR / "uploads"
    DERIVED_DIR = RUNTIME_DIR / "derived"
    RUNS_DIR = RUNTIME_DIR / "runs"
    REPORTS_DIR = RUNTIME_DIR / "reports"
    AI_RUNTIME_DIR = RUNTIME_DIR / "ai"
    AI_KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge_base"
    EXPOSE_SERVER_PATHS = False


def test_runtime_path_roundtrip_serializes_to_runtime_scheme(tmp_path):
    runtime_dir = tmp_path / "runtime"
    target = runtime_dir / "uploads" / "csv" / "demo.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    config = {
        "RUNTIME_DIR": str(runtime_dir),
        "REPO_ROOT": str(tmp_path / "repo"),
    }

    stored = serialize_local_path(target, config)

    assert stored == f"{RUNTIME_PREFIX}uploads/csv/demo.csv"
    assert resolve_local_path(stored, config) == str(target.resolve())


def test_repo_path_roundtrip_serializes_to_repo_scheme(tmp_path):
    repo_root = tmp_path / "repo"
    target = repo_root / "assets" / "models" / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("weights", encoding="utf-8")
    config = {
        "RUNTIME_DIR": str(tmp_path / "runtime"),
        "REPO_ROOT": str(repo_root),
    }

    stored = serialize_local_path(target, config)

    assert stored == f"{REPO_PREFIX}assets/models/best.pt"
    assert resolve_local_path(stored, config) == str(target.resolve())


def test_legacy_runtime_absolute_path_can_still_be_resolved(tmp_path):
    runtime_dir = tmp_path / "runtime"
    target = runtime_dir / "uploads" / "csv" / "legacy.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    config = {
        "RUNTIME_DIR": str(runtime_dir),
        "REPO_ROOT": str(tmp_path / "repo"),
    }

    legacy_path = (
        "D:/UAV-Fault-Diagnosis/apps/dashboard/backend/runtime/uploads/csv/legacy.csv"
    )

    assert resolve_local_path(legacy_path, config) == str(target.resolve())


def test_asset_meta_serializes_prepared_csv_path():
    app = create_app(config_class=PathStorageTestingConfig)
    runtime_file = Path(app.config["UPLOAD_DIR"]) / "csv" / "prepared.csv"
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_file.write_text("a,b\n1,2\n", encoding="utf-8")

    with app.app_context():
        asset = Asset(
            filename="prepared.csv",
            kind="csv",
            stored_path=serialize_local_path(runtime_file, app.config),
            size_bytes=runtime_file.stat().st_size,
            sha256="deadbeef",
        )
        asset.set_meta({"prepared_csv_path": str(runtime_file)})

        assert asset.get_meta()["prepared_csv_path"] == "runtime://uploads/csv/prepared.csv"


def test_run_record_result_serializes_path_like_fields():
    app = create_app(config_class=PathStorageTestingConfig)
    runtime_file = Path(app.config["UPLOAD_DIR"]) / "csv" / "input.csv"
    model_dir = PROJECT_ROOT / "assets" / "models" / "online_detector_cnn1d_rfly"
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_file.write_text("a,b\n1,2\n", encoding="utf-8")

    with app.app_context():
        run = RunRecord(task="offline_multiclass", status="success", progress=1.0)
        run.set_result(
            {
                "input_csv": str(runtime_file),
                "debug": {
                    "model_dir": str(model_dir),
                    "norm_source": str(model_dir / "norm_stats.json"),
                },
            }
        )

        payload = run.get_result()
        assert payload["input_csv"] == "runtime://uploads/csv/input.csv"
        assert payload["debug"]["model_dir"].startswith("repo://assets/models/")
        assert payload["debug"]["norm_source"].startswith("repo://assets/models/")


def test_model_to_dict_hides_server_paths_by_default():
    app = create_app(config_class=PathStorageTestingConfig)

    with app.app_context():
        asset = Asset(
            filename="demo.csv",
            kind="csv",
            stored_path="runtime://uploads/csv/demo.csv",
            size_bytes=10,
            sha256="deadbeef",
        )
        report = DiagnosisReport(
            run_id=1,
            asset_id=None,
            model_name="offline-model",
            prediction_label="motor",
            confidence=0.9,
            pdf_path="runtime://reports/demo.pdf",
        )

        assert "stored_path" not in asset.to_dict()
        assert "pdf_path" not in report.to_dict()
