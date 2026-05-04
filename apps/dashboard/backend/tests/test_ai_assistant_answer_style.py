from dashboard.backend.tests.test_ai_assistant import app, auth_header, seeded_records  # noqa: F401

from uav_dashboard_api.services.ai_assistant import SYSTEM_PROMPT, ai_assistant_service


def test_system_prompt_requires_conclusion_reason_and_action_flow():
    assert "鐩存帴缁撹" in SYSTEM_PROMPT
    assert "鍘熷洜瑙ｉ噴" in SYSTEM_PROMPT
    assert "椋庨櫓涓庡缓璁? in SYSTEM_PROMPT


def test_fallback_answer_uses_three_stage_natural_structure(app, seeded_records):
    with app.app_context():
        scene_context = ai_assistant_service._build_scene_context(
            scene="report_analysis",
            report_id=seeded_records["report_id"],
            run_id=seeded_records["run_id"],
        )
        answer = ai_assistant_service._fallback_answer(
            scene_context=scene_context,
            message="璇疯В閲婂綋鍓嶆姤鍛婁腑鐨勯闄╅噸鐐?,
            knowledge_hits=[],
        )

        assert "鐩存帴缁撹锛? in answer
        assert "鍘熷洜瑙ｉ噴锛? in answer
        assert "椋庨櫓涓庡缓璁細" in answer
