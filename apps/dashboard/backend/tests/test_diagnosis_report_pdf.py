from pathlib import Path

import bcrypt
import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.user import User
from uav_dashboard_api.services.diagnosis_report import (
    _build_data_quality,
    _build_pdf_lines,
    create_or_get_diagnosis_report,
)
from uav_dashboard_api.utils.path_store import resolve_local_path
from uav_dashboard_api.utils.jwt_utils import JWTUtils
from uav_dashboard_api.config import TestingConfig


@pytest.fixture
def app(tmp_path: Path):
    db_path = tmp_path / "test_diagnosis_report_pdf.db"
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
            username="pdf_user",
            password=hashed,
            real_name="PDF User",
            email="pdf@example.com",
            role="admin",
            status="active",
        )
        db.session.add(user)
        db.session.commit()
        token = JWTUtils.generate_token(user.id)
        return {"Authorization": f"Bearer {token}"}, user.id


def test_build_data_quality_marks_low_sample_as_poor():
    quality = _build_data_quality({"debug": {"sample_count": 5}}, {})
    assert quality["level"] == "poor"
    assert quality["warnings"]


def test_build_pdf_lines_supports_missing_enhanced_fields():
    summary = {
        "basic": {"report_generated_at": "2026-03-26 21:00:00", "run_id": 1},
        "conclusion": {"label": "normal", "confidence": 0.9},
        "severity": {"level": "none", "method": "rule_v1"},
        "topk": [{"label": "normal", "prob": 0.9}],
        "recommendations": [],
    }
    lines = _build_pdf_lines(summary, {"normal": 0.9})
    content = "\n".join(lines)
    assert "数据质量" in content
    assert "人工复核记录" in content
    assert "未知" in content or "待补充" in content


def test_build_pdf_lines_uses_user_readable_chinese_summary():
    summary = {
        "basic": {
            "report_generated_at": "2026-03-26 21:00:00",
            "run_id": 7,
            "asset_id": 3,
            "asset_filename": "mission_003.csv",
            "model_name": "cnn_lstm",
        },
        "conclusion": {
            "label": "motor",
            "confidence": 0.93,
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
        "topk": [
            {"label": "motor", "prob": 0.93},
            {"label": "propeller", "prob": 0.04},
        ],
        "key_params": {
            "motor_imbalance_ratio": 0.41,
            "esc_rpm_imbalance_ratio": 0.33,
        },
        "recommendations": [
            {
                "title": "必要时停飞并安排动力系统复检",
                "priority": "P0",
                "reason_codes": ["motor_imbalance_ratio"],
            }
        ],
        "recommendation_meta": {
            "strategy": "rule_v1",
            "fallback_used": False,
            "severity_level": "heavy",
            "candidate_count": 1,
        },
        "data_quality": {
            "level": "good",
            "sample_count": 120,
            "window_count": 40,
            "feature_count": 32,
            "missing_columns": [],
            "missing_value_ratio": None,
            "invalid_value_ratio": None,
            "time_range": {"start": None, "end": None},
            "warnings": [],
        },
        "traceability": {
            "run_id": 7,
            "asset_id": 3,
            "asset_filename": "mission_003.csv",
            "input_csv": "/tmp/mission_003.csv",
            "model_name": "cnn_lstm",
            "model_version": "v1",
            "model_hash": "",
            "rules_version": "rule_v1",
            "generated_at": "2026-03-26 21:00:00",
        },
        "model_card": {
            "task": "offline_multiclass",
            "class_set": ["normal", "motor", "sensor"],
            "intended_use": "离线日志故障诊断辅助分析",
            "limitations": [],
            "metrics": {},
        },
        "report_confidence": {
            "level": "high",
            "score": 0.93,
            "reasons": ["输入数据质量可接受", "模型主分类置信度较高"],
        },
        "review": {
            "status": "pending",
            "reviewer": "",
            "reviewed_at": "",
            "conclusion": "",
            "maintenance_notes": "",
        },
    }

    lines = _build_pdf_lines(summary, {"motor": 0.93, "propeller": 0.04, "normal": 0.03})
    content = "\n".join(lines)

    assert "离线诊断报告" in content
    assert "本次离线诊断识别为电机故障" in content
    assert "高风险" in content
    assert "建议动作" in content
    assert "数据质量" in content
    assert "模型与规则追溯" in content
    assert "人工复核记录" in content
    assert "电机输出均衡性异常" in content
    assert "必要时停飞并安排动力系统复检" in content


def test_create_or_get_diagnosis_report_writes_pdf_file(app, auth_header):
    _, user_id = auth_header

    with app.app_context():
        asset = Asset(
            filename="mission_pdf.csv",
            kind="csv",
            stored_path="/tmp/mission_pdf.csv",
            size_bytes=128,
            sha256="abc123",
            created_by=user_id,
        )
        asset.set_meta({"prepared_csv_path": "/tmp/mission_pdf.csv"})
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
        assert summary["data_quality"]["level"] in {"unknown", "good", "warning", "poor"}
        assert summary["traceability"]["model_name"] == "cnn_lstm"
        assert summary["report_confidence"]["level"] in {"high", "medium", "low", "unknown"}
        assert summary["review"]["status"] == "pending"
        assert "\\u" not in report.summary_json
        pdf_path = Path(resolve_local_path(report.pdf_path, app.config))
        assert pdf_path.exists()
        assert pdf_path.read_bytes().startswith(b"%PDF")
