from tests.test_ai_assistant import app, auth_header, seeded_records  # noqa: F401

from uav_dashboard_api.services.ai_assistant import ai_assistant_service


def test_report_analysis_query_includes_report_summary_terms(app, seeded_records):
    with app.app_context():
        scene_context = ai_assistant_service._build_scene_context(
            scene="report_analysis",
            report_id=seeded_records["report_id"],
            run_id=seeded_records["run_id"],
        )

        query = ai_assistant_service._knowledge_query(
            scene_context,
            "请结合当前报告分析可能原因。",
        )

        assert "battery_failure" in query
        assert "heavy" in query
        assert "battery_warning" in query
        assert "battery_heavy_replace" in query
