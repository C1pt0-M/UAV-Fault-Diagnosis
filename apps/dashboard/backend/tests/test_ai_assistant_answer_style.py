from tests.test_ai_assistant import app, auth_header, seeded_records  # noqa: F401

from uav_dashboard_api import db
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.services.ai_assistant import SYSTEM_PROMPT, ai_assistant_service


def test_system_prompt_requires_report_interpreter_structure():
    assert "报告解释器" in SYSTEM_PROMPT
    assert "结论：" in SYSTEM_PROMPT
    assert "证据：" in SYSTEM_PROMPT
    assert "不确定性与限制：" in SYSTEM_PROMPT
    assert "不足以支持强结论" in SYSTEM_PROMPT
    assert "直接结论" in SYSTEM_PROMPT


def test_fallback_answer_uses_report_analysis_structure(app, seeded_records):
    with app.app_context():
        scene_context = ai_assistant_service._build_scene_context(
            scene="report_analysis",
            report_id=seeded_records["report_id"],
            run_id=seeded_records["run_id"],
        )
        answer = ai_assistant_service._fallback_answer(
            scene_context=scene_context,
            message="请解释当前报告中的风险重点。",
            knowledge_hits=[],
        )

        assert "结论：" in answer
        assert "证据：" in answer
        assert "风险：" in answer
        assert "建议动作：" in answer
        assert "不确定性与限制：" in answer
        assert "需要人工确认：" in answer
        assert "直接结论：" not in answer


def test_fallback_answer_warns_when_data_quality_is_poor(app, seeded_records):
    with app.app_context():
        report = db.session.get(DiagnosisReport, seeded_records["report_id"])
        summary = report.get_summary()
        summary.update(
            {
                "data_quality": {
                    "level": "poor",
                    "sample_count": 0,
                    "feature_count": 0,
                    "warnings": ["未记录输入 CSV 路径，数据追溯能力受限。"],
                },
                "report_confidence": {
                    "level": "low",
                    "score": 0.31,
                    "reasons": ["输入数据质量较差，结论支撑度不足。"],
                },
                "review": {"status": "pending"},
            }
        )
        report.set_summary(summary)
        db.session.commit()

        scene_context = ai_assistant_service._build_scene_context(
            scene="report_analysis",
            report_id=report.id,
            run_id=seeded_records["run_id"],
        )
        answer = ai_assistant_service._fallback_answer(
            scene_context=scene_context,
            message="这份报告可以直接下结论吗？",
            knowledge_hits=[],
        )

        assert "不足以支持强结论" in answer
        assert "数据质量" in answer or "poor" in answer
        assert "未记录输入 CSV 路径" in answer
        assert "螺旋桨" not in answer
        assert "已更换" not in answer
