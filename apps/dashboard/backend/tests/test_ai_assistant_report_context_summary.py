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
        assert "人工复核" in joined
        assert "报告摘要:" not in joined
        assert "统一决策:" not in joined
        assert any(item["source_type"] == "report" for item in citations)


def test_report_context_includes_enhanced_summary_fields(app, seeded_records):
    with app.app_context():
        report = db.session.get(DiagnosisReport, seeded_records["report_id"])
        summary = report.get_summary()
        summary.update(
            {
                "data_quality": {
                    "level": "warning",
                    "sample_count": 1200,
                    "feature_count": 31,
                    "warnings": ["未记录时间范围，建议结合原始日志复核。"],
                },
                "traceability": {
                    "run_id": report.run_id,
                    "input_csv": "/tmp/mission_001.csv",
                    "rules_version": "rule_v1",
                },
                "report_confidence": {
                    "level": "medium",
                    "score": 0.72,
                    "reasons": ["模型主分类置信度中等（91.0%）。"],
                },
                "review": {
                    "status": "pending",
                    "reviewer": "",
                    "reviewed_at": "",
                    "conclusion": "",
                    "maintenance_notes": "",
                },
            }
        )
        report.set_summary(summary)
        db.session.commit()

        blocks, _ = ai_assistant_service._build_report_context(report)
        joined = "\n".join(blocks)

        assert "数据质量:" in joined
        assert "level=warning" in joined
        assert "报告可信度:" in joined
        assert "level=medium" in joined
        assert "追溯信息:" in joined
        assert "rules_version=rule_v1" in joined
        assert "人工复核:" in joined
        assert "status=pending" in joined
        assert "未记录时间范围" in joined


def test_collect_report_terms_includes_enhanced_fields(app, seeded_records):
    with app.app_context():
        report = db.session.get(DiagnosisReport, seeded_records["report_id"])
        summary = report.get_summary()
        summary["data_quality"] = {
            "level": "poor",
            "warnings": ["缺失值占比较高（约 18.0%）。"],
        }
        summary["report_confidence"] = {
            "level": "low",
            "reasons": ["输入数据质量较差，结论支撑度不足。"],
        }
        summary["review"] = {"status": "pending", "conclusion": ""}
        report.set_summary(summary)

        terms = ai_assistant_service._collect_report_terms(report)

        assert "poor" in terms
        assert "low" in terms
        assert "pending" in terms
        assert "缺失值占比较高（约 18.0%）。" in terms
        assert "输入数据质量较差，结论支撑度不足。" in terms
