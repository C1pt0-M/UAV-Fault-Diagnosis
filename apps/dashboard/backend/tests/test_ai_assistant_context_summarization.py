from dashboard.backend.tests.test_ai_assistant import app, auth_header, seeded_records  # noqa: F401

from uav_dashboard_api import db
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.services.ai_assistant import ai_assistant_service


def test_report_context_uses_summary_lines_instead_of_raw_json(app, seeded_records):
    with app.app_context():
        report = db.session.get(DiagnosisReport, seeded_records["report_id"])
        blocks, _ = ai_assistant_service._build_report_context(report)
        joined = "\n".join(blocks)

        assert "姒傜巼鍒嗗竷:" not in joined
        assert "缁熶竴鍐崇瓥:" not in joined
        assert "鎶ュ憡鎽樿:" not in joined
        assert "涓昏缁撹" in joined
        assert "涓ラ噸搴? in joined
        assert "澶勭疆寤鸿" in joined


def test_run_context_uses_summary_lines_instead_of_raw_json(app, seeded_records):
    with app.app_context():
        run = db.session.get(RunRecord, seeded_records["run_id"])
        blocks, _ = ai_assistant_service._build_run_context(run)
        joined = "\n".join(blocks)

        assert "浠诲姟鍙傛暟:" not in joined
        assert "浠诲姟缁撴灉:" not in joined
        assert "浠诲姟绫诲瀷:" in joined
        assert "鐘舵€?" in joined
        assert "浠诲姟缁撹:" in joined
