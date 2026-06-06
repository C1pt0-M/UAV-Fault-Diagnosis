from pathlib import Path

import bcrypt
import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.user import User
from uav_dashboard_api.services.diagnosis_report import (
    _build_data_quality,
    create_or_get_diagnosis_report,
)
from uav_dashboard_api.utils.jwt_utils import JWTUtils
from uav_dashboard_api.config import TestingConfig


@pytest.fixture
def app(tmp_path: Path):
    db_path = tmp_path / "test_diagnosis_report.db"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    class LocalTestingConfig(TestingConfig):
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_path}"
        RUNTIME_DIR = runtime_dir
        UPLOAD_DIR = runtime_dir / "uploads"
        DERIVED_DIR = runtime_dir / "derived"
        RUNS_DIR = runtime_dir / "runs"
        REPORTS_DIR = runtime_dir / "reports"

    app = create_app(LocalTestingConfig)
    app.config["TESTING"] = True

    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def auth_header(app):
    with app.app_context():
        hashed = bcrypt.hashpw("test123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        user = User(
            username="report_user",
            password=hashed,
            real_name="Report User",
            email="report@example.com",
            role="admin",
            status="active",
        )
        db.session.add(user)
        db.session.commit()
        token = JWTUtils.generate_token(user.id)
        return {"Authorization": f"Bearer {token}"}, user.id


def test_create_or_get_diagnosis_report_writes_readable_summary_without_pdf(app, auth_header):
    _, user_id = auth_header

    with app.app_context():
        asset = Asset(
            filename="mission_report.csv",
            kind="csv",
            stored_path="/tmp/mission_report.csv",
            size_bytes=128,
            sha256="abc123",
            created_by=user_id,
        )
        asset.set_meta({"prepared_csv_path": "/tmp/mission_report.csv"})
        db.session.add(asset)
        db.session.flush()

        run = RunRecord(
            task="offline_multiclass",
            status="success",
            progress=1.0,
            asset_id=asset.id,
            created_by=user_id,
        )
        run.set_params({"asset_id": asset.id, "model": "cnn_lstm"})
        run.set_result(
            {
                "prediction": {"label": "motor", "confidence": 0.93},
                "probs": {"motor": 0.93, "propeller": 0.04, "normal": 0.03},
                "labels": ["normal", "motor", "sensor"],
                "input_csv": "/tmp/mission_report.csv",
                "debug": {
                    "csv_path": "/tmp/mission_report.csv",
                    "sample_count": 800,
                    "in_channels": 24,
                },
                "severity": {
                    "level": "heavy",
                    "score": 0.88,
                    "method": "rule_v1",
                    "evidence": [
                        {
                            "metric": "motor_imbalance_ratio",
                            "status": "used",
                            "value": 0.41,
                            "score": 1.0,
                            "required": True,
                        }
                    ],
                },
                "recommendations": [
                    {
                        "title": "必要时停飞并安排动力系统复检",
                        "priority": "P0",
                        "reason_codes": ["motor_imbalance_ratio"],
                    }
                ],
            }
        )
        db.session.add(run)
        db.session.commit()

        report, created = create_or_get_diagnosis_report(run, created_by=user_id)

        assert created is True
        summary = report.get_summary()
        assert summary["readable"]["conclusion"].startswith("本次离线诊断识别为电机故障")
        assert summary["readable"]["evidence"]
        assert summary["readable"]["recommendations"]
        assert summary["data_quality"]["level"] in {"good", "warning"}
        assert "mission_report.csv" in summary["traceability"]["input_csv"]
        assert summary["report_confidence"]["level"] in {"high", "medium", "low", "unknown"}
        assert summary["review"]["status"] == "pending"
        assert report.pdf_path == ""
        assert "pdf_path" not in report.to_dict()


def test_data_quality_does_not_treat_seq_len_as_sample_count(app):
    with app.app_context():
        quality = _build_data_quality(
            {
                "input_csv": "/tmp/missing_seq_only.csv",
                "debug": {
                    "seq_len": 2048,
                    "in_channels": 31,
                },
            },
            None,
        )

    assert quality["level"] == "warning"
    assert quality["sample_count"] is None
    assert any("序列长度" in warning for warning in quality["warnings"])


def test_data_quality_counts_samples_from_resolvable_csv(app):
    with app.app_context():
        csv_path = Path(app.config["UPLOAD_DIR"]) / "sample_rows.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(
            "timestamp,roll,pitch\n"
            "0.00,0.1,0.2\n"
            "0.01,0.2,0.3\n"
            "0.02,0.3,0.4\n",
            encoding="utf-8",
        )

        quality = _build_data_quality(
            {
                "input_csv": str(csv_path),
                "debug": {
                    "feature_dim": 2,
                },
            },
            None,
        )

    assert quality["level"] == "good"
    assert quality["sample_count"] == 3
    assert quality["feature_count"] == 2


def _create_legacy_report(app, user_id: int, *, summary: dict, label: str = "motor"):
    run = RunRecord(
        task="offline_multiclass",
        status="success",
        progress=1.0,
        created_by=user_id,
    )
    run.set_result({"prediction": {"label": label, "confidence": 0.81}})
    db.session.add(run)
    db.session.flush()

    report = DiagnosisReport(
        run_id=run.id,
        asset_id=None,
        model_name="legacy-model",
        prediction_label=label,
        confidence=0.81,
        pdf_path="",
        created_by=user_id,
    )
    report.set_summary(summary)
    report.set_probs({label: 0.81, "normal": 0.19})
    db.session.add(report)
    db.session.commit()
    return report


def test_legacy_report_to_dict_tolerates_missing_enhanced_fields(app, auth_header):
    _, user_id = auth_header

    with app.app_context():
        report = _create_legacy_report(
            app,
            user_id,
            summary={
                "severity": {"level": "light", "evidence": []},
                "recommendations": [{"title": "安排检查", "priority": "P2"}],
            },
        )
        payload = report.to_dict()

        assert payload["prediction_label"] == "motor"
        assert payload["summary"]["severity"]["level"] == "light"
        assert "data_quality" not in payload["summary"]
        assert "\\u" not in report.summary_json
        assert "安排检查" in report.summary_json


def test_set_summary_preserves_chinese_without_unicode_escape(app, auth_header):
    _, user_id = auth_header

    with app.app_context():
        report = _create_legacy_report(
            app,
            user_id,
            label="sensor",
            summary={
                "data_quality": {
                    "level": "warning",
                    "warnings": ["未记录样本数量，无法完整评估输入覆盖度。"],
                },
                "review": {"status": "pending"},
            },
        )

        assert "\\u672a" not in report.summary_json
        assert "未记录样本数量" in report.summary_json
        assert report.get_summary()["data_quality"]["warnings"][0].startswith("未记录")


