from tests.test_ai_assistant import app, auth_header, seeded_records  # noqa: F401

from uav_dashboard_api.services.ai_assistant import SYSTEM_PROMPT, ai_assistant_service


def test_system_prompt_requires_report_interpreter_structure():
    assert "结论：" in SYSTEM_PROMPT
    assert "证据：" in SYSTEM_PROMPT
    assert "不确定性与限制：" in SYSTEM_PROMPT
    assert "报告解释器" in SYSTEM_PROMPT


def test_fallback_answer_uses_report_interpreter_structure(app, seeded_records):
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
