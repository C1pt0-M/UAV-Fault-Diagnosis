from pathlib import Path

import bcrypt
import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.models.ai_conversation import AIConversation, AIMessage
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.user import User
from uav_dashboard_api.services.ai_assistant import SYSTEM_PROMPT, ai_assistant_service
from uav_dashboard_api.utils.jwt_utils import JWTUtils
from uav_dashboard_api.config import TestingConfig


@pytest.fixture
def app(tmp_path: Path):
    db_path = tmp_path / "test_ai.db"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    knowledge_dir = runtime_dir / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    (knowledge_dir / "uav_ops.md").write_text(
        "无人机电池故障通常表现为电压下降，battery_warning 增大。",
        encoding="utf-8",
    )

    class LocalTestingConfig(TestingConfig):
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_path}"
        RUNTIME_DIR = runtime_dir
        UPLOAD_DIR = runtime_dir / "uploads"
        DERIVED_DIR = runtime_dir / "derived"
        RUNS_DIR = runtime_dir / "runs"
        REPORTS_DIR = runtime_dir / "reports"
        AI_RUNTIME_DIR = runtime_dir / "ai"
        AI_KNOWLEDGE_DIR = knowledge_dir
        AI_ENABLE_KNOWLEDGE_BASE = False
        AI_PROJECT_DOC_PATHS = ()
        AI_PROVIDER = "openai_compatible"
        AI_BASE_URL = "https://example.invalid/v1"
        AI_API_KEY = "test-key"
        AI_MODEL = "deepseek-chat"
        AI_TOP_K = 3

    app = create_app(LocalTestingConfig)
    app.config["TESTING"] = True

    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_header(app):
    with app.app_context():
        hashed = bcrypt.hashpw("test123".encode("utf-8"), bcrypt.gensalt()).decode(
            "utf-8"
        )
        user = User(
            username="ai_user",
            password=hashed,
            real_name="AI User",
            email="ai@example.com",
            role="admin",
            status="active",
        )
        db.session.add(user)
        db.session.commit()
        token = JWTUtils.generate_token(user.id)
        return {"Authorization": f"Bearer {token}"}, user.id


@pytest.fixture
def seeded_records(app, auth_header):
    _, user_id = auth_header
    with app.app_context():
        asset = Asset(
            filename="mission_001.csv",
            kind="csv",
            stored_path="/tmp/mission_001.csv",
            size_bytes=128,
            sha256="abc123",
            created_by=user_id,
        )
        asset.set_meta({"prepared_csv_path": "/tmp/mission_001.csv"})
        db.session.add(asset)
        db.session.flush()

        run = RunRecord(
            task="offline_multiclass",
            status="success",
            progress=1.0,
            asset_id=asset.id,
            created_by=user_id,
        )
        run.set_params({"asset_id": asset.id, "model": "cnn1d"})
        run.set_result(
            {
                "prediction": {"label": "battery_failure", "confidence": 0.91},
                "severity": {"level": "heavy", "score": 0.88},
                "recommendations": [{"title": "立即更换电池包", "priority": "P1"}],
            }
        )
        db.session.add(run)
        db.session.flush()

        report = DiagnosisReport(
            run_id=run.id,
            asset_id=asset.id,
            model_name="cnn1d",
            prediction_label="battery_failure",
            confidence=0.91,
            pdf_path="",
            created_by=user_id,
        )
        report.set_probs({"battery_failure": 0.91, "normal": 0.09})
        report.set_summary(
            {
                "severity": {"level": "heavy", "score": 0.88, "method": "rule_v1"},
                "recommendations": [
                    {
                        "title": "立即更换电池包",
                        "priority": "P1",
                        "reason_codes": ["battery_heavy_replace"],
                    }
                ],
                "decision": {
                    "episode_id": "ep_ai_001",
                    "state": "DECIDED",
                    "fault_label": "battery_failure",
                    "severity": "heavy",
                    "warning_stage": "critical",
                    "warning_reason": "统一决策判定为高风险，应立即处置。",
                    "evidence_summary": {
                        "source": "offline",
                        "fault_label": "battery_failure",
                        "severity": {
                            "level": "heavy",
                            "score": 0.88,
                            "method": "rule_v1",
                        },
                        "top_evidence": [
                            {"metric": "battery_warning", "status": "used"}
                        ],
                    },
                    "recommendations": [{"title": "立即更换电池包", "priority": "P1"}],
                },
            }
        )
        db.session.add(report)
        db.session.commit()
        return {"asset_id": asset.id, "run_id": run.id, "report_id": report.id}


def test_knowledge_reindex_and_source_listing_returns_empty_when_disabled(
    client, auth_header, seeded_records
):
    headers, _ = auth_header
    response = client.post("/api/ai/knowledge/reindex", headers=headers)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["data"]["enabled"] is False
    assert payload["data"]["documents"] == 0
    assert payload["data"]["chunks"] == 0

    response = client.get("/api/ai/knowledge/sources", headers=headers)
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["enabled"] is False
    assert data["total_documents"] == 0
    assert data["total_chunks"] == 0
    assert data["by_type"] == {}


def test_knowledge_search_returns_empty_when_disabled(app):
    from uav_dashboard_api.services.knowledge_base import knowledge_base

    with app.app_context():
        assert knowledge_base.search("电池故障怎么修", top_k=3) == []


def test_analyze_report_creates_conversation_and_messages(
    app, client, auth_header, seeded_records, monkeypatch
):
    headers, user_id = auth_header

    def fake_call_model(messages, **kwargs):
        return (
            "结论：报告显示 battery_failure 风险高，建议立即检查电池。",
            "siliconflow",
            "deepseek-chat",
            False,
        )

    monkeypatch.setattr(ai_assistant_service, "_call_model", fake_call_model)

    response = client.post(
        "/api/ai/analyze-report",
        headers=headers,
        json={"report_id": seeded_records["report_id"]},
    )
    assert response.status_code == 200
    payload = response.get_json()["data"]
    assert payload["conversation"]["scene"] == "report_analysis"
    assert payload["conversation"]["report_id"] == seeded_records["report_id"]
    assert payload["message"]["role"] == "assistant"
    assert payload["used_context"]["report_id"] == seeded_records["report_id"]
    assert any(item["source_type"] == "report" for item in payload["citations"])

    with app.app_context():
        conv = AIConversation.query.filter_by(created_by=user_id).one()
        assert conv.report_id == seeded_records["report_id"]
        assert AIMessage.query.filter_by(conversation_id=conv.id).count() == 2


def test_report_context_prioritizes_unified_decision(app, seeded_records):
    with app.app_context():
        report = db.session.get(DiagnosisReport, seeded_records["report_id"])
        blocks, citations = ai_assistant_service._build_report_context(report)
        joined = "\n".join(blocks)

        assert "统一决策说明" in joined
        assert "统一决策判定为高风险" in joined
        assert "battery_warning" in joined
        assert "立即更换电池包" in joined
        assert "报告摘要:" not in joined
        assert "统一决策:" not in joined
        assert any(item["source_type"] == "report" for item in citations)


def test_chat_endpoint_reuses_conversation(
    client, auth_header, seeded_records, monkeypatch
):
    headers, _ = auth_header

    monkeypatch.setattr(
        ai_assistant_service,
        "_call_model",
        lambda messages, **kwargs: (
            "这是基于当前任务上下文的分析回复。",
            "siliconflow",
            "deepseek-chat",
            False,
        ),
    )

    first = client.post(
        "/api/ai/chat",
        headers=headers,
        json={
            "message": "帮我分析这份报告。",
            "scene": "report_analysis",
            "report_id": seeded_records["report_id"],
        },
    )
    assert first.status_code == 200
    first_data = first.get_json()["data"]
    conv_id = first_data["conversation"]["id"]

    second = client.post(
        "/api/ai/chat",
        headers=headers,
        json={
            "conversation_id": conv_id,
            "message": "再总结一下风险点。",
            "scene": "report_analysis",
            "report_id": seeded_records["report_id"],
        },
    )
    assert second.status_code == 200
    second_data = second.get_json()["data"]
    assert second_data["conversation"]["id"] == conv_id

    listing = client.get("/api/ai/conversations", headers=headers)
    assert listing.status_code == 200
    items = listing.get_json()["data"]
    assert len(items) == 1
    assert items[0]["id"] == conv_id

    detail = client.get(f"/api/ai/conversations/{conv_id}", headers=headers)
    assert detail.status_code == 200
    detail_data = detail.get_json()["data"]
    assert len(detail_data["messages"]) == 4


def test_analyze_run_endpoint_removed(client, auth_header, seeded_records):
    headers, _ = auth_header

    response = client.post(
        "/api/ai/analyze-run",
        headers=headers,
        json={"run_id": seeded_records["run_id"]},
    )
    assert response.status_code == 404


def test_legacy_report_serialization_keeps_canonical_task(app, seeded_records):
    with app.app_context():
        run = RunRecord(
            task="offline_5class",
            status="success",
            progress=1.0,
            asset_id=seeded_records["asset_id"],
        )
        run.set_result({"prediction": {"label": "motor", "confidence": 0.8}})
        db.session.add(run)
        db.session.flush()

        report = DiagnosisReport(
            run_id=run.id,
            asset_id=seeded_records["asset_id"],
            model_name="legacy-model",
            prediction_label="motor",
            confidence=0.8,
            pdf_path="",
        )
        report.set_summary({"severity": {"level": "medium"}, "recommendations": []})
        db.session.add(report)
        db.session.commit()

        payload = report.to_dict()
        assert payload["diagnosis"]["model"]["task"] == "offline_multiclass"


def test_system_prompt_prefers_natural_language_without_inline_reference_sections():
    assert "不要使用 Markdown 标题或粗体" in SYSTEM_PROMPT
    assert "不要在正文中直接写“参考资料”" in SYSTEM_PROMPT
    assert "无法从当前资料确认" in SYSTEM_PROMPT


def test_fallback_answer_uses_natural_language_without_reference_sections(
    app, seeded_records
):
    with app.app_context():
        scene_context = ai_assistant_service._build_scene_context(
            scene="report_analysis",
            report_id=seeded_records["report_id"],
            run_id=seeded_records["run_id"],
        )
        answer = ai_assistant_service._fallback_answer(
            scene_context=scene_context,
            message="请判断这份报告的风险重点。",
            knowledge_hits=[
                {
                    "source_type": "knowledge_base",
                    "source_id": "kb-1",
                    "title": "电池维护手册",
                    "snippet": "电池故障通常表现为电压异常下降。",
                }
            ],
        )

        assert "结论：" in answer
        assert "证据：" in answer
        assert "风险：" in answer
        assert "建议动作：" in answer
        assert "不确定性与限制：" in answer
        assert "需要人工确认：" in answer
        assert "知识片段" not in answer
        assert "参考资料" not in answer
        assert "电池故障" in answer
        assert "高风险" in answer
        assert "立即更换电池包" in answer
