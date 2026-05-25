from tests.test_ai_assistant import app, auth_header, seeded_records  # noqa: F401

from uav_dashboard_api import db
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.services.ai_assistant import ai_assistant_service


def test_report_context_keeps_unified_decision_signal_without_raw_json(app, seeded_records):
    with app.app_context():
        report = db.session.get(DiagnosisReport, seeded_records["report_id"])
        blocks, citations = ai_assistant_service._build_report_context(report)
        joined = "\n".join(blocks)

        assert "统一决策" in joined
        assert "统一决策判定为高风险" in joined
        assert "battery_warning" in joined
        assert "立即更换电池包" in joined
        assert "报告摘要:" not in joined
        assert "统一决策:" not in joined
        assert any(item["source_type"] == "report" for item in citations)
