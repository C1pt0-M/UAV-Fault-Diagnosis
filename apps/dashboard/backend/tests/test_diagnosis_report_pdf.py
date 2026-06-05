import re
import sys
from pathlib import Path

import bcrypt
import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.user import User
from uav_dashboard_api.services.diagnosis_report import (
    REPORTLAB_FONT_CANDIDATES,
    _build_pdf_lines,
    _render_pdf_with_reportlab,
    create_or_get_diagnosis_report,
)
from uav_dashboard_api.utils.pdf_fonts import build_pdf_font_candidates, resolve_pdf_font_name
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
            "sample_count": 1200,
            "window_count": 48,
            "feature_count": 32,
            "missing_columns": [],
            "missing_value_ratio": 0.01,
            "invalid_value_ratio": 0.0,
            "time_range": {"start": "2026-03-26 20:00:00", "end": "2026-03-26 21:00:00"},
            "warnings": [],
        },
        "traceability": {
            "run_id": 7,
            "asset_id": 3,
            "asset_filename": "mission_003.csv",
            "input_csv": "/tmp/mission_003.csv",
            "model_name": "cnn_lstm",
            "model_version": "",
            "model_hash": "",
            "rules_version": "rule_v1",
            "generated_at": "2026-03-26 21:00:00",
        },
        "model_card": {
            "task": "offline_multiclass",
            "class_set": ["normal", "motor", "sensor"],
            "intended_use": "RflySim/RflyMAD 离线日志故障诊断辅助分析",
            "limitations": [],
            "metrics": {},
        },
        "report_confidence": {
            "level": "high",
            "score": 0.86,
            "reasons": ["模型主分类置信度较高（93.0%）。"],
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
    assert "二、数据质量" in content
    assert "四、风险等级与可信度" in content
    assert "六、模型与规则追溯" in content
    assert "七、人工复核记录" in content
    assert "高风险" in content
    assert "建议动作" in content
    assert "电机输出均衡性异常" in content
    assert "必要时停飞并安排动力系统复检" in content
    assert "待复核" in content


def test_build_pdf_lines_handles_missing_enhanced_fields():
    summary = {
        "basic": {
            "report_generated_at": "2026-03-26 21:00:00",
            "run_id": 1,
            "asset_id": 1,
            "asset_filename": "legacy.csv",
            "model_name": "cnn_lstm",
        },
        "conclusion": {"label": "normal", "confidence": 0.8},
        "severity": {"level": "none", "score": 0, "method": "rule_v1", "evidence": []},
        "topk": [{"label": "normal", "prob": 0.8}],
        "key_params": {},
        "recommendations": [],
        "recommendation_meta": {},
    }

    lines = _build_pdf_lines(summary, {"normal": 0.8, "motor": 0.2})
    content = "\n".join(lines)

    assert "二、数据质量" in content
    assert "七、人工复核记录" in content
    assert "待复核" in content
    assert "待补充" in content


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
                "labels": ["normal", "motor", "sensor"],
                "input_csv": "/tmp/mission_pdf.csv",
                "debug": {
                    "csv_path": "/tmp/mission_pdf.csv",
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
        assert summary["data_quality"]["level"] in {"good", "warning"}
        assert "mission_pdf.csv" in summary["traceability"]["input_csv"]
        assert summary["report_confidence"]["level"] in {"high", "medium", "low", "unknown"}
        assert summary["review"]["status"] == "pending"
        pdf_path = Path(resolve_local_path(report.pdf_path, app.config))
        assert pdf_path.exists()
        assert pdf_path.read_bytes().startswith(b"%PDF")


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
        pdf_path=str(app.config["REPORTS_DIR"] / f"legacy_{run.id}.pdf"),
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


def test_pdf_font_candidates_include_platform_defaults():
    candidates = build_pdf_font_candidates()
    assert candidates
    assert REPORTLAB_FONT_CANDIDATES == candidates
    if sys.platform.startswith("win"):
        assert any("msyh.ttc" in item.lower() for item in candidates)


def test_render_pdf_with_reportlab_preserves_chinese_when_font_available(app, tmp_path):
    if resolve_pdf_font_name(REPORTLAB_FONT_CANDIDATES) == "Helvetica":
        pytest.skip("当前环境没有可用中文字体")

    summary = {
        "basic": {
            "report_generated_at": "2026-03-26 21:00:00",
            "run_id": 1,
            "asset_id": 1,
            "asset_filename": "mission.csv",
            "model_name": "cnn_lstm",
        },
        "conclusion": {"label": "normal", "confidence": 0.8},
        "severity": {"level": "none", "score": 0, "method": "rule_v1", "evidence": []},
        "topk": [{"label": "normal", "prob": 0.8}],
        "key_params": {},
        "recommendations": [{"title": "保持例行巡检与校准", "priority": "P3", "reason_codes": []}],
        "recommendation_meta": {},
        "data_quality": {"level": "good", "sample_count": 100, "window_count": 0, "feature_count": 31, "missing_columns": [], "missing_value_ratio": None, "invalid_value_ratio": None, "time_range": {"start": None, "end": None}, "warnings": []},
        "traceability": {"run_id": 1, "asset_id": 1, "asset_filename": "mission.csv", "input_csv": "mission.csv", "model_name": "cnn_lstm", "model_version": "", "model_hash": "", "rules_version": "rule_v1", "generated_at": "2026-03-26 21:00:00"},
        "model_card": {"task": "offline_multiclass", "class_set": ["normal", "motor", "sensor"], "intended_use": "离线日志故障诊断辅助分析", "limitations": [], "metrics": {}},
        "report_confidence": {"level": "medium", "score": 0.7, "reasons": ["模型主分类置信度中等（80.0%）。"]},
        "review": {"status": "pending", "reviewer": "", "reviewed_at": "", "conclusion": "", "maintenance_notes": ""},
    }
    lines = _build_pdf_lines(summary, {"normal": 0.8, "motor": 0.2})
    pdf_path = tmp_path / "chinese_report.pdf"

    with app.app_context():
        assert _render_pdf_with_reportlab(pdf_path, lines) is True

    from pypdf import PdfReader

    text = "\n".join((page.extract_text() or "") for page in PdfReader(str(pdf_path)).pages)
    assert re.search(r"[\u4e00-\u9fff]", text)
    assert "离线诊断报告" in text
    assert "数据质量" in text
    assert "待复核" in text
