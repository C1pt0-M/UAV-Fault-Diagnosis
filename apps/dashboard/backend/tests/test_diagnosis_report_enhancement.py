import json
from pathlib import Path

import bcrypt
import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.config import TestingConfig
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.user import User
from uav_dashboard_api.services.ai_assistant import ai_assistant_service
from uav_dashboard_api.services.diagnosis_report import _build_pdf_lines
from uav_dashboard_api.utils.jwt_utils import JWTUtils


@pytest.fixture
def app(tmp_path: Path):
    db_path = tmp_path / "test_diagnosis_report_enhancement.db"
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
def client(app):
    return app.test_client()


@pytest.fixture
def auth_header(app):
    with app.app_context():
        hashed = bcrypt.hashpw("test123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        user = User(
            username="enh_user",
            password=hashed,
            real_name="Enh User",
            email="enh@example.com",
            role="admin",
            status="active",
        )
        db.session.add(user)
        db.session.commit()
        token = JWTUtils.generate_token(user.id)
        return {"Authorization": f"Bearer {token}"}, user.id


def _create_legacy_report(user_id: int) -> DiagnosisReport:
    asset = Asset(
        filename="legacy_mission.csv",
        kind="csv",
        stored_path="/tmp/legacy_mission.csv",
        size_bytes=64,
        sha256="legacy123",
        created_by=user_id,
    )
    db.session.add(asset)
    db.session.flush()

    run = RunRecord(
        task="offline_multiclass",
        status="success",
        progress=1.0,
        asset_id=asset.id,
        created_by=user_id,
    )
    run.set_params({"model": "cnn_lstm"})
    run.set_result(
        {
            "prediction": {"label": "sensor", "confidence": 0.71},
            "probs": {"sensor": 0.71, "motor": 0.2, "normal": 0.09},
            "severity": {"level": "medium", "score": 0.55, "method": "rule_v1"},
            "recommendations": [{"title": "检查 IMU 校准", "priority": "P1"}],
        }
    )
    db.session.add(run)
    db.session.flush()

    report = DiagnosisReport(
        run_id=run.id,
        asset_id=asset.id,
        model_name="cnn_lstm",
        prediction_label="sensor",
        confidence=0.71,
        pdf_path=str(Path("/tmp/legacy-report.pdf")),
        created_by=user_id,
    )
    report.set_probs({"sensor": 0.71, "motor": 0.2, "normal": 0.09})
    report.set_summary(
        {
            "basic": {
                "run_id": run.id,
                "asset_id": asset.id,
                "asset_filename": asset.filename,
                "model_name": "cnn_lstm",
                "report_generated_at": "2026-01-01 00:00:00",
            },
            "conclusion": {"label": "sensor", "confidence": 0.71},
            "severity": {"level": "medium", "score": 0.55, "method": "rule_v1"},
            "topk": [{"label": "sensor", "prob": 0.71}],
            "key_params": {},
            "recommendations": [{"title": "检查 IMU 校准", "priority": "P1"}],
        }
    )
    db.session.add(report)
    db.session.commit()
    return report


def test_legacy_report_detail_api_and_pdf_remain_compatible(app, client, auth_header):
    headers, user_id = auth_header

    with app.app_context():
        report = _create_legacy_report(user_id)

        response = client.get(
            f"/api/reports/diagnosis/{report.id}",
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.get_json()["data"]
        assert payload["prediction_label"] == "sensor"
        assert "data_quality" not in payload["summary"]

        lines = _build_pdf_lines(payload["summary"], report.get_probs())
        content = "\n".join(lines)
        assert "数据质量" in content
        assert "人工复核记录" in content
        assert "待补充" in content or "未知" in content
        assert "\\u" not in json.dumps(payload["summary"], ensure_ascii=False)


def test_legacy_report_ai_fallback_does_not_crash(app, auth_header):
    _, user_id = auth_header

    with app.app_context():
        report = _create_legacy_report(user_id)
        scene_context = ai_assistant_service._build_scene_context(
            scene="report_analysis",
            report_id=report.id,
            run_id=report.run_id,
        )
        answer = ai_assistant_service._fallback_answer(
            scene_context=scene_context,
            message="请解释这份旧报告。",
            knowledge_hits=[],
        )

        assert "结论：" in answer
        assert "传感器故障" in answer
        assert "证据：" in answer


def test_fallback_answer_does_not_invent_fault_types_outside_report(app, auth_header):
    _, user_id = auth_header

    with app.app_context():
        report = _create_legacy_report(user_id)
        scene_context = ai_assistant_service._build_scene_context(
            scene="report_analysis",
            report_id=report.id,
            run_id=None,
        )
        answer = ai_assistant_service._fallback_answer(
            scene_context=scene_context,
            message="这份报告是否说明螺旋桨故障？",
            knowledge_hits=[
                {
                    "source_type": "knowledge_base",
                    "source_id": "kb-prop",
                    "title": "螺旋桨维护",
                    "snippet": "螺旋桨失衡会导致持续振动。",
                }
            ],
        )

        assert "螺旋桨" not in answer
        assert "propeller" not in answer.lower()
        assert "传感器故障" in answer
