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
        "鏃犱汉鏈虹數姹犳晠闅滈€氬父琛ㄧ幇涓虹數鍘嬩笅闄嶃€乥attery_warning 澧炲ぇ銆?,
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
                "recommendations": [{"title": "绔嬪嵆鏇存崲鐢垫睜鍖?, "priority": "P1"}],
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
            pdf_path=str(app.config["REPORTS_DIR"] / "report.pdf"),
            created_by=user_id,
        )
        report.set_probs({"battery_failure": 0.91, "normal": 0.09})
        report.set_summary(
            {
                "severity": {"level": "heavy", "score": 0.88, "method": "rule_v1"},
                "recommendations": [
                    {
                        "title": "绔嬪嵆鏇存崲鐢垫睜鍖?,
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
                    "warning_reason": "缁熶竴鍐崇瓥鍒ゅ畾涓洪珮椋庨櫓锛屽簲绔嬪嵆澶勭疆銆?,
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
                    "recommendations": [{"title": "绔嬪嵆鏇存崲鐢垫睜鍖?, "priority": "P1"}],
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
        assert knowledge_base.search("鐢垫睜鏁呴殰鎬庝箞淇?, top_k=3) == []


def test_analyze_report_creates_conversation_and_messages(
    app, client, auth_header, seeded_records, monkeypatch
):
    headers, user_id = auth_header

    def fake_call_model(messages, **kwargs):
        return (
            "缁撹锛氭姤鍛婃樉绀?battery_failure 椋庨櫓楂橈紝寤鸿绔嬪嵆妫€鏌ョ數姹犮€?,
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

        assert "缁熶竴鍐崇瓥" in joined
        assert "缁熶竴鍐崇瓥鍒ゅ畾涓洪珮椋庨櫓" in joined
        assert "battery_warning" in joined
        assert "绔嬪嵆鏇存崲鐢垫睜鍖? in joined
        assert "鎶ュ憡鎽樿:" not in joined
        assert "缁熶竴鍐崇瓥:" not in joined
        assert any(item["source_type"] == "report" for item in citations)


def test_chat_endpoint_reuses_conversation(
    client, auth_header, seeded_records, monkeypatch
):
    headers, _ = auth_header

    monkeypatch.setattr(
        ai_assistant_service,
        "_call_model",
        lambda messages, **kwargs: (
            "杩欐槸鍩轰簬褰撳墠浠诲姟涓婁笅鏂囩殑鍒嗘瀽鍥炲銆?,
            "siliconflow",
            "deepseek-chat",
            False,
        ),
    )

    first = client.post(
        "/api/ai/chat",
        headers=headers,
        json={
            "message": "甯垜鍒嗘瀽杩欎唤鎶ュ憡銆?,
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
            "message": "鍐嶆€荤粨涓€涓嬮闄╃偣銆?,
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
            pdf_path="/tmp/legacy-report.pdf",
        )
        report.set_summary({"severity": {"level": "medium"}, "recommendations": []})
        db.session.add(report)
        db.session.commit()

        payload = report.to_dict()
        assert payload["diagnosis"]["model"]["task"] == "offline_multiclass"


def test_system_prompt_prefers_natural_language_without_inline_reference_sections():
    assert "涓嶈浣跨敤 Markdown 鏍囬鎴栫矖浣? in SYSTEM_PROMPT
    assert "涓嶈鍦ㄦ鏂囦腑鐩存帴鍐欌€滃弬鑰冭祫鏂欌€? in SYSTEM_PROMPT
    assert "鏃犳硶浠庡綋鍓嶈祫鏂欑‘璁? in SYSTEM_PROMPT


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
            message="\u8bf7\u5224\u65ad\u8fd9\u4efd\u62a5\u544a\u7684\u98ce\u9669\u91cd\u70b9\u3002",
            knowledge_hits=[
                {
                    "source_type": "knowledge_base",
                    "source_id": "kb-1",
                    "title": "\u7535\u6c60\u7ef4\u62a4\u624b\u518c",
                    "snippet": "\u7535\u6c60\u6545\u969c\u901a\u5e38\u8868\u73b0\u4e3a\u7535\u538b\u5f02\u5e38\u4e0b\u964d\u3002",
                }
            ],
        )

        assert "\u76f4\u63a5\u7ed3\u8bba\uff1a" in answer
        assert "\u539f\u56e0\u89e3\u91ca\uff1a" in answer
        assert "\u98ce\u9669\u4e0e\u5efa\u8bae\uff1a" in answer
        assert "\u77e5\u8bc6\u7247\u6bb5" not in answer
        assert "\u53c2\u8003\u8d44\u6599" not in answer
        assert "battery_failure" in answer
