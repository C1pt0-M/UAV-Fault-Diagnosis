from tests.test_ai_assistant import app, auth_header, seeded_records  # noqa: F401

from uav_dashboard_api.models.ai_conversation import AIConversation
from uav_dashboard_api.services.ai_assistant import ai_assistant_service


def test_fallback_answer_for_general_question_does_not_force_report_binding(app):
    with app.app_context():
        scene_context = ai_assistant_service._build_scene_context(
            scene="general",
            report_id=None,
            run_id=None,
        )
        answer = ai_assistant_service._fallback_answer(
            scene_context=scene_context,
            message="电机故障的主要表现有哪些？",
            knowledge_hits=[],
        )

        assert "建议绑定具体 report_id" not in answer
        assert "一般性判断" not in answer
        assert "直接结论：" in answer


def test_report_analysis_prompt_supports_actionable_follow_up(app, seeded_records, auth_header):
    _, user_id = auth_header
    with app.app_context():
        conversation = ai_assistant_service._create_conversation(
            scene="report_analysis",
            user_id=user_id,
            report_id=seeded_records["report_id"],
            run_id=seeded_records["run_id"],
            title="报告分析追问",
        )
        prompt_messages = ai_assistant_service._build_prompt_messages(
            scene_context=ai_assistant_service._build_scene_context(
                scene="report_analysis",
                report_id=seeded_records["report_id"],
                run_id=seeded_records["run_id"],
            ),
            knowledge_hits=[],
            user_message="具体要怎么修复？",
            conversation=conversation,
        )

        system_messages = [item["content"] for item in prompt_messages if item["role"] == "system"]
        merged = "\n".join(system_messages)
        assert "分步骤排查" in merged
        assert "优先给出可执行的检查与修复建议" in merged

        AIConversation.query.filter_by(id=conversation.id).delete()


def test_report_analysis_scene_context_uses_compact_report_summary(app, seeded_records):
    with app.app_context():
        scene_context = ai_assistant_service._build_scene_context(
            scene="report_analysis",
            report_id=seeded_records["report_id"],
            run_id=seeded_records["run_id"],
        )
        joined = "\n".join(scene_context.blocks)

        assert "【任务记录】" not in joined
        assert "warning_stage=" not in joined
        assert "统一决策说明" in joined
