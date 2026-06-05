from tests.test_ai_assistant import app, auth_header, seeded_records  # noqa: F401

from uav_dashboard_api import db
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.services.ai_assistant import ai_assistant_service


def test_report_context_uses_summary_lines_instead_of_raw_json(app, seeded_records):
    with app.app_context():
        report = db.session.get(DiagnosisReport, seeded_records["report_id"])
        blocks, _ = ai_assistant_service._build_report_context(report)
        joined = "\n".join(blocks)

        assert "概率分布:" not in joined
        assert "统一决策:" not in joined
        assert "报告摘要:" not in joined
        assert "主要结论" in joined
        assert "严重度" in joined
        assert "处置建议" in joined


def test_run_context_uses_summary_lines_instead_of_raw_json(app, seeded_records):
    with app.app_context():
        run = db.session.get(RunRecord, seeded_records["run_id"])
        blocks, _ = ai_assistant_service._build_run_context(run)
        joined = "\n".join(blocks)

        assert "任务参数:" not in joined
        assert "任务结果:" not in joined
        assert "任务类型:" in joined
        assert "状态:" in joined
        assert "任务结论:" in joined
