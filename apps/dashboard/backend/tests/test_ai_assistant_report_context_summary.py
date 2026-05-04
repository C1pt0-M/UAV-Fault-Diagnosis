from dashboard.backend.tests.test_ai_assistant import app, auth_header, seeded_records  # noqa: F401

from uav_dashboard_api import db
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.services.ai_assistant import ai_assistant_service


def test_report_context_keeps_unified_decision_signal_without_raw_json(app, seeded_records):
    with app.app_context():
        report = db.session.get(DiagnosisReport, seeded_records["report_id"])
        blocks, citations = ai_assistant_service._build_report_context(report)
        joined = "\n".join(blocks)

        assert "缁熶竴鍐崇瓥" in joined
        assert "缁熶竴鍐崇瓥鍒ゅ畾涓洪珮椋庨櫓" in joined
        assert "battery_warning" in joined
        assert "绔嬪嵆鏇存崲鐢垫睜鍖? in joined
        assert "鎶ュ憡鎽樿:" not in joined
        assert "缁熶竴鍐崇瓥:" not in joined
        assert any(item["source_type"] == "report" for item in citations)
