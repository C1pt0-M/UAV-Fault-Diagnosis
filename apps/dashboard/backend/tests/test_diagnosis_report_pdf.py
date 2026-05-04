from pathlib import Path

import bcrypt
import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.user import User
from uav_dashboard_api.services.diagnosis_report import _build_pdf_lines, create_or_get_diagnosis_report
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
                "title": "蹇呰鏃跺仠椋炲苟瀹夋帓鍔ㄥ姏绯荤粺澶嶆",
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
    }

    lines = _build_pdf_lines(summary, {"motor": 0.93, "propeller": 0.04, "normal": 0.03})
    content = "\n".join(lines)

    assert "绂荤嚎璇婃柇鎶ュ憡" in content
    assert "鏈绂荤嚎璇婃柇璇嗗埆涓虹數鏈烘晠闅? in content
    assert "楂橀闄? in content
    assert "寤鸿鍔ㄤ綔" in content
    assert "鐢垫満杈撳嚭鍧囪　鎬у紓甯? in content
    assert "蹇呰鏃跺仠椋炲苟瀹夋帓鍔ㄥ姏绯荤粺澶嶆" in content


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
                        "title": "蹇呰鏃跺仠椋炲苟瀹夋帓鍔ㄥ姏绯荤粺澶嶆",
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
        pdf_path = Path(report.pdf_path)
        assert pdf_path.exists()
        assert pdf_path.read_bytes().startswith(b"%PDF")
