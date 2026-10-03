from pathlib import Path

import bcrypt
import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.models.user import User
from uav_dashboard_api.utils.jwt_utils import JWTUtils
from uav_dashboard_api.config import TestingConfig


@pytest.fixture
def app(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    class LocalTestingConfig(TestingConfig):
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_path / 'settings.db'}"
        RUNTIME_DIR = runtime_dir
        UPLOAD_DIR = runtime_dir / "uploads"
        DERIVED_DIR = runtime_dir / "derived"
        RUNS_DIR = runtime_dir / "runs"
        REPORTS_DIR = runtime_dir / "reports"
        AI_RUNTIME_DIR = runtime_dir / "ai"
        AI_SETTINGS_PATH = runtime_dir / "ai_settings.json"
        AI_PROVIDER = "openai_compatible"
        AI_BASE_URL = "https://env.example/v1"
        AI_API_KEY = "env-secret-key"
        AI_MODEL = "env-model"
        AI_EMBEDDING_BASE_URL = "https://embedding-env.example/v1"
        AI_EMBEDDING_API_KEY = "embedding-env-secret-key"
        AI_EMBEDDING_MODEL = "env-embedding-model"
        ENABLE_AI = True

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


def _auth_headers(app, *, role: str = "admin"):
    with app.app_context():
        password = bcrypt.hashpw("test123".encode("utf-8"), bcrypt.gensalt()).decode(
            "utf-8"
        )
        user = User(
            username=f"{role}_settings_user",
            password=password,
            real_name="Settings User",
            email=f"{role}_settings@example.com",
            role=role,
            status="active",
        )
        db.session.add(user)
        db.session.commit()
        token = JWTUtils.generate_token(user.id)
        return {"Authorization": f"Bearer {token}"}


def test_get_ai_settings_returns_masked_effective_config(client, app):
    response = client.get("/api/settings/ai", headers=_auth_headers(app))

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["provider"] == "openai_compatible"
    assert data["base_url"] == "https://env.example/v1"
    assert data["model"] == "env-model"
    assert data["api_key_set"] is True
    assert data["api_key_preview"].startswith("env-")
    assert data["embedding_base_url"] == "https://embedding-env.example/v1"
    assert data["embedding_model"] == "env-embedding-model"
    assert data["embedding_api_key_set"] is True
    assert data["embedding_api_key_preview"].startswith("embe")
    response_text = response.get_data(as_text=True)
    assert "env-secret-key" not in response_text
    assert "embedding-env-secret-key" not in response_text


def test_put_ai_settings_persists_runtime_override_without_returning_secret(client, app):
    headers = _auth_headers(app)

    response = client.put(
        "/api/settings/ai",
        headers=headers,
        json={
            "provider": "deepseek",
            "base_url": "https://runtime.example/v1/",
            "model": "runtime-model",
            "api_key": "runtime-secret-key",
        },
    )

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["provider"] == "deepseek"
    assert data["base_url"] == "https://runtime.example/v1"
    assert data["model"] == "runtime-model"
    assert data["api_key_set"] is True
    assert "runtime-secret-key" not in response.get_data(as_text=True)

    response = client.get("/api/settings/ai", headers=headers)
    data = response.get_json()["data"]
    assert data["base_url"] == "https://runtime.example/v1"
    assert data["model"] == "runtime-model"
    assert data["api_key_preview"].startswith("runt")


def test_embedding_settings_do_not_inherit_assistant_config(client, app):
    app.config["AI_EMBEDDING_BASE_URL"] = ""
    app.config["AI_EMBEDDING_API_KEY"] = ""
    app.config["AI_EMBEDDING_MODEL"] = ""

    response = client.get("/api/settings/ai", headers=_auth_headers(app))

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["embedding_base_url"] == ""
    assert data["embedding_model"] == ""
    assert data["embedding_api_key_set"] is False


def test_put_embedding_settings_persists_independent_runtime_override(client, app):
    headers = _auth_headers(app)

    response = client.put(
        "/api/settings/ai",
        headers=headers,
        json={
            "embedding_base_url": "https://embedding-runtime.example/v1/",
            "embedding_api_key": "embedding-runtime-secret",
            "embedding_model": "runtime-embedding-model",
            "rag_mode": "hybrid",
        },
    )

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["embedding_base_url"] == "https://embedding-runtime.example/v1"
    assert data["embedding_model"] == "runtime-embedding-model"
    assert data["embedding_api_key_set"] is True
    assert data["rag_mode"] == "hybrid"
    assert "embedding-runtime-secret" not in response.get_data(as_text=True)


def test_put_empty_api_key_keeps_existing_key_and_clear_flag_removes_it(client, app):
    headers = _auth_headers(app)
    client.put(
        "/api/settings/ai",
        headers=headers,
        json={
            "base_url": "https://runtime.example/v1",
            "model": "runtime-model",
            "api_key": "runtime-secret-key",
        },
    )

    response = client.put(
        "/api/settings/ai",
        headers=headers,
        json={"base_url": "https://runtime2.example/v1", "api_key": ""},
    )
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["base_url"] == "https://runtime2.example/v1"
    assert data["api_key_set"] is True
    assert data["api_key_preview"].startswith("runt")

    response = client.put(
        "/api/settings/ai",
        headers=headers,
        json={"clear_api_key": True},
    )
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["api_key_set"] is False
    assert data["api_key_preview"] == ""


def test_ai_settings_requires_admin(client, app):
    response = client.get("/api/settings/ai", headers=_auth_headers(app, role="user"))

    assert response.status_code == 403


def test_ai_settings_test_connection_uses_candidate_config(client, app, monkeypatch):
    headers = _auth_headers(app)
    seen = {}

    def fake_chat(self, messages, *, temperature=0.2):
        seen["base_url"] = self.base_url
        seen["api_key"] = self.api_key
        seen["model"] = self.model
        seen["messages"] = messages
        return {"provider": self.provider_name, "model": self.model, "content": "ok"}

    monkeypatch.setattr("uav_dashboard_api.routes.settings.OpenAICompatibleProvider.chat", fake_chat)

    response = client.post(
        "/api/settings/ai/test",
        headers=headers,
        json={
            "base_url": "https://candidate.example/v1",
            "model": "candidate-model",
            "api_key": "candidate-secret",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["data"]["ok"] is True
    assert seen["base_url"] == "https://candidate.example/v1"
    assert seen["api_key"] == "candidate-secret"
    assert seen["model"] == "candidate-model"
    assert "candidate-secret" not in response.get_data(as_text=True)


def test_embedding_connection_uses_candidate_embedding_config(client, app, monkeypatch):
    headers = _auth_headers(app)
    seen = {}

    def fake_embed_texts(self, texts):
        seen["base_url"] = self.embedding_base_url
        seen["api_key"] = self.embedding_api_key
        seen["model"] = self.embedding_model
        seen["texts"] = texts
        return {
            "provider": self.provider_name,
            "model": self.embedding_model,
            "embeddings": [[0.1, 0.2]],
        }

    monkeypatch.setattr(
        "uav_dashboard_api.routes.settings.OpenAICompatibleProvider.embed_texts",
        fake_embed_texts,
    )

    response = client.post(
        "/api/settings/ai/test-embedding",
        headers=headers,
        json={
            "embedding_base_url": "https://candidate-embedding.example/v1",
            "embedding_api_key": "candidate-embedding-secret",
            "embedding_model": "candidate-embedding-model",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["data"]["ok"] is True
    assert seen["base_url"] == "https://candidate-embedding.example/v1"
    assert seen["api_key"] == "candidate-embedding-secret"
    assert seen["model"] == "candidate-embedding-model"
    assert seen["texts"] == ["Embedding connection test."]
    assert "candidate-embedding-secret" not in response.get_data(as_text=True)
