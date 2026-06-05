import time

from tests.test_ai_assistant import app, auth_header, seeded_records

from uav_dashboard_api.services.ai_assistant import ai_assistant_service


class HangingProvider:
    provider_name = "mock-provider"
    model = "mock-model"

    def chat(self, messages):
        time.sleep(0.05)
        return {
            "provider": self.provider_name,
            "model": self.model,
            "content": "this response should time out",
            "usage": {},
        }


def test_call_model_falls_back_when_chat_exceeds_hard_timeout(
    app, seeded_records, monkeypatch
):
    with app.app_context():
        app.config["AI_CHAT_HARD_TIMEOUT"] = 0.01
        scene_context = ai_assistant_service._build_scene_context(
            scene="report_analysis",
            report_id=seeded_records["report_id"],
            run_id=seeded_records["run_id"],
        )
        monkeypatch.setattr(
            ai_assistant_service,
            "_provider_client",
            lambda: HangingProvider(),
        )

        started_at = time.perf_counter()
        answer, provider_name, model_name, fallback_used = ai_assistant_service._call_model(
            [{"role": "user", "content": "请解释当前报告的风险重点。"}],
            scene_context=scene_context,
            message="请解释当前报告的风险重点。",
            knowledge_hits=[],
        )
        elapsed = time.perf_counter() - started_at

        assert elapsed < 0.08
        assert fallback_used is True
        assert provider_name == "offline_fallback"
        assert model_name == "context_only"
        assert "结论：" in answer
        assert "需要人工确认：" in answer
