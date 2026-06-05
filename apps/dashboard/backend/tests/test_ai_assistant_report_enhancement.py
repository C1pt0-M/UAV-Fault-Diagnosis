from tests.test_ai_assistant import app, auth_header, seeded_records  # noqa: F401

from uav_dashboard_api import db
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.services.ai_assistant import ai_assistant_service


def test_report_context_includes_enhanced_summary_fields(app, seeded_records):
    with app.app_context():
        report = db.session.get(DiagnosisReport, seeded_records["report_id"])
        summary = report.get_summary()
        summary.update(
            {
                "data_quality": {
                    "level": "warning",
                    "sample_count": 12,
                    "window_count": 4,
                    "feature_count": 16,
                    "warnings": ["样本数量偏少，建议结合人工复核"],
                },
                "report_confidence": {
                    "level": "medium",
                    "score": 0.72,
                    "reasons": ["模型主分类置信度中等"],
                },
                "traceability": {
                    "model_name": report.model_name,
                    "rules_version": "rule_v1",
                    "input_csv": "/tmp/demo.csv",
                },
                "review": {"status": "pending"},
            }
        )
        report.set_summary(summary)
        db.session.commit()

        blocks, _ = ai_assistant_service._build_report_context(report)
        joined = "\n".join(blocks)

        assert "数据质量" in joined
        assert "报告可信度" in joined
        assert "追溯信息" in joined
        assert "人工复核状态" in joined
        assert "样本数量偏少" in joined
