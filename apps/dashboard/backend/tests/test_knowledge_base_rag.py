from collections import Counter
import json
from pathlib import Path

import bcrypt
import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.models.user import User
from uav_dashboard_api.services.ai_assistant import ai_assistant_service
from uav_dashboard_api.services.ai_provider import OpenAICompatibleProvider
from uav_dashboard_api.services.knowledge_base import knowledge_base
from uav_dashboard_api.utils.jwt_utils import JWTUtils
from uav_dashboard_api.config import TestingConfig


@pytest.fixture
def app(tmp_path: Path):
    db_path = tmp_path / "test_kb_rag.db"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    (knowledge_dir / "motor_doc.txt").write_text(
        "bearing winding thermal inspection manual",
        encoding="utf-8",
    )
    (knowledge_dir / "mission_doc.txt").write_text(
        "route planning crosswind weather envelope",
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
        AI_ENABLE_KNOWLEDGE_BASE = True
        AI_PROJECT_DOC_PATHS = ()
        AI_PROVIDER = "openai_compatible"
        AI_BASE_URL = "https://example.invalid/v1"
        AI_API_KEY = "test-key"
        AI_MODEL = "test-chat"
        AI_EMBEDDING_BASE_URL = "https://example.invalid/v1"
        AI_EMBEDDING_API_KEY = "test-key"
        AI_EMBEDDING_MODEL = "test-embedding"
        AI_RAG_MODE = "hybrid"
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
            username="rag_user",
            password=hashed,
            real_name="RAG User",
            email="rag@example.com",
            role="admin",
            status="active",
        )
        db.session.add(user)
        db.session.commit()
        token = JWTUtils.generate_token(user.id)
        return {"Authorization": f"Bearer {token}"}, user.id


def _fake_embed_texts(self, texts: list[str]) -> dict:
    def _vectorize(text: str) -> list[float]:
        value = str(text or "").lower()
        if "bearing winding thermal inspection manual" in value or "motor repair" in value:
            return [1.0, 0.0, 0.0]
        if "route planning crosswind weather envelope" in value:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    return {
        "provider": self.provider_name,
        "model": self.embedding_model,
        "embeddings": [_vectorize(text) for text in texts],
    }


def test_knowledge_reindex_builds_hybrid_index_with_embeddings(
    app, client, auth_header, monkeypatch
):
    headers, _ = auth_header
    monkeypatch.setattr(OpenAICompatibleProvider, "embed_texts", _fake_embed_texts)

    response = client.post("/api/ai/knowledge/reindex", headers=headers)
    assert response.status_code == 200
    payload = response.get_json()["data"]
    assert payload["enabled"] is True
    assert payload["documents"] >= 2
    assert payload["retrieval_mode"] == "hybrid"
    assert payload["embedding_enabled"] is True

    sources = client.get("/api/ai/knowledge/sources", headers=headers)
    assert sources.status_code == 200
    data = sources.get_json()["data"]
    assert data["enabled"] is True
    assert data["retrieval_mode"] == "hybrid"
    assert data["embedding_enabled"] is True

    with app.app_context():
        hits = knowledge_base.search("motor repair", top_k=1)
        assert hits[0]["title"] == "motor_doc.txt"
        assert hits[0]["retrieval_mode"] == "hybrid"
        assert hits[0]["score_breakdown"]["semantic"] > 0.99


def test_chat_endpoint_uses_knowledge_hits_from_hybrid_search(
    client, auth_header, monkeypatch
):
    headers, user_id = auth_header
    monkeypatch.setattr(OpenAICompatibleProvider, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(
        ai_assistant_service,
        "_call_model",
        lambda messages, **kwargs: (
            "缁撹锛氱煡璇嗗簱宸插懡涓數鏈虹淮淇枃妗ｃ€?,
            "mock-provider",
            "mock-chat",
            False,
        ),
    )

    response = client.post(
        "/api/ai/chat",
        headers=headers,
        json={"message": "motor repair", "scene": "general"},
    )
    assert response.status_code == 200
    payload = response.get_json()["data"]
    assert payload["used_context"]["knowledge_hits"] >= 1
    assert payload["used_context"]["knowledge_retrieval_mode"] == "hybrid"
    assert any(item["source_type"] == "knowledge_base" for item in payload["citations"])

    with client.application.app_context():
        user = db.session.get(User, user_id)
        assert user is not None


def test_search_prefers_hits_that_cover_repair_intent(app, monkeypatch):
    with app.app_context():
        def make_chunk(chunk_id, source_id, title, text):
            tokens = knowledge_base._tokenize(text)
            return {
                "chunk_id": chunk_id,
                "source_type": "knowledge_base",
                "source_id": source_id,
                "title": title,
                "path": "",
                "text": text,
                "token_counts": dict(Counter(tokens)),
                "token_len": len(tokens),
            }

        index = {
            "built_at": "2026-03-31 19:00:00",
            "documents": [],
            "chunks": [
                make_chunk(
                    "1",
                    "doc-sensor-overview",
                    "浼犳劅鍣ㄦ晠闅滄杩?,
                    "浼犳劅鍣ㄦ晠闅滀富瑕佸垎涓烘紓绉诲亸缃晠闅溿€侀儴鍒嗗け鏁堟晠闅滃拰瀹屽叏澶辨晥鏁呴殰銆?,
                ),
                make_chunk(
                    "2",
                    "doc-sensor-repair",
                    "浼犳劅鍣ㄧ淮淇寚鍗?,
                    "浼犳劅鍣ㄦ晠闅滀慨澶嶉€氬父鍏堟帓鏌ヤ緵鐢典笌杩炴帴鐘舵€侊紝鍐嶆牴鎹儏鍐垫墽琛屾牎鍑嗐€佺淮淇垨鏇存崲銆?,
                ),
            ],
            "df": {},
            "avg_token_len": 1.0,
            "retrieval_mode": "bm25",
            "requested_retrieval_mode": "bm25",
            "embedding_enabled": False,
            "embedding_provider": "",
            "embedding_model": "",
            "embedding_error": None,
            "embedding_dim": 0,
        }
        df = Counter()
        total_len = 0
        for chunk in index["chunks"]:
            df.update(chunk["token_counts"].keys())
            total_len += chunk["token_len"]
        index["df"] = dict(df)
        index["avg_token_len"] = total_len / len(index["chunks"])

        monkeypatch.setattr(knowledge_base, "_load_index", lambda: index)

        hits = knowledge_base.search("浼犳劅鍣ㄦ晠闅滄€庝箞淇", top_k=5)

        assert len(hits) == 1
        assert hits[0]["title"] == "浼犳劅鍣ㄧ淮淇寚鍗?
        assert "淇閫氬父鍏堟帓鏌ヤ緵鐢? in hits[0]["snippet"]


def test_search_builds_snippet_around_matched_query(app, monkeypatch):
    with app.app_context():
        prefix = "鑳屾櫙浠嬬粛銆? * 80
        focus = "杩欓噷寮€濮嬭璁虹數姹犳晠闅滀慨澶嶆楠わ紝鍖呮嫭鎺掓煡渚涚數銆侀噸鏂版牎鍑嗗拰蹇呰鏃舵洿鎹㈤儴浠躲€?
        text = prefix + focus + "灏鹃儴琛ュ厖璇存槑銆?
        tokens = knowledge_base._tokenize(text)
        index = {
            "built_at": "2026-03-31 19:00:00",
            "documents": [],
            "chunks": [
                {
                    "chunk_id": "1",
                    "source_type": "knowledge_base",
                    "source_id": "doc-battery-repair",
                    "title": "鐢垫睜缁翠慨鎵嬪唽",
                    "path": "",
                    "text": text,
                    "token_counts": dict(Counter(tokens)),
                    "token_len": len(tokens),
                }
            ],
            "df": dict(Counter(tokens)),
            "avg_token_len": len(tokens),
            "retrieval_mode": "bm25",
            "requested_retrieval_mode": "bm25",
            "embedding_enabled": False,
            "embedding_provider": "",
            "embedding_model": "",
            "embedding_error": None,
            "embedding_dim": 0,
        }

        monkeypatch.setattr(knowledge_base, "_load_index", lambda: index)

        hits = knowledge_base.search("鐢垫睜鏁呴殰鎬庝箞淇", top_k=5)

        assert len(hits) == 1
        assert "鐢垫睜鏁呴殰淇姝ラ" in hits[0]["snippet"]
        assert "鑳屾櫙浠嬬粛" not in hits[0]["snippet"][:40]


def test_load_index_rebuilds_when_report_or_run_sources_are_stale(app, monkeypatch):
    with app.app_context():
        stale_index = {
            "built_at": "2026-03-31 19:00:00",
            "documents": [
                {
                    "doc_id": 1,
                    "kind": "report",
                    "source_type": "report",
                    "source_id": "999",
                    "title": "璇婃柇鎶ュ憡 #999",
                    "path": "",
                    "updated_at": "2026-03-31 19:00:00",
                },
                {
                    "doc_id": 2,
                    "kind": "run",
                    "source_type": "run",
                    "source_id": "888",
                    "title": "浠诲姟璁板綍 #888",
                    "path": "",
                    "updated_at": "2026-03-31 19:00:00",
                },
            ],
            "chunks": [],
            "df": {},
            "avg_token_len": 0.0,
            "retrieval_mode": "bm25",
            "requested_retrieval_mode": "bm25",
            "embedding_enabled": False,
            "embedding_provider": "",
            "embedding_model": "",
            "embedding_error": None,
            "embedding_dim": 0,
        }

        rebuilt_index = {
            "built_at": "2026-03-31 19:05:00",
            "documents": [
                {
                    "doc_id": 1,
                    "kind": "file",
                    "source_type": "knowledge_base",
                    "source_id": "doc-1",
                    "title": "鏃犱汉鏈虹淮鎶ゆ墜鍐?,
                    "path": "",
                    "updated_at": "2026-03-31 19:05:00",
                }
            ],
            "chunks": [],
            "df": {},
            "avg_token_len": 0.0,
            "retrieval_mode": "bm25",
            "requested_retrieval_mode": "bm25",
            "embedding_enabled": False,
            "embedding_provider": "",
            "embedding_model": "",
            "embedding_error": None,
            "embedding_dim": 0,
        }

        path = knowledge_base._index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stale_index, ensure_ascii=False), encoding="utf-8")

        def fake_rebuild():
            path.write_text(json.dumps(rebuilt_index, ensure_ascii=False), encoding="utf-8")
            return {"enabled": True}

        monkeypatch.setattr(knowledge_base, "rebuild_index", fake_rebuild)

        loaded = knowledge_base._load_index()

        assert loaded["documents"][0]["source_type"] == "knowledge_base"
        assert all(item["source_type"] != "run" for item in loaded["documents"])
        assert all(item["source_type"] != "report" for item in loaded["documents"])


def test_requested_retrieval_mode_defaults_to_bm25_when_config_missing(app):
    with app.app_context():
        app.config.pop("AI_RAG_MODE", None)

        assert knowledge_base._requested_retrieval_mode() == "bm25"
