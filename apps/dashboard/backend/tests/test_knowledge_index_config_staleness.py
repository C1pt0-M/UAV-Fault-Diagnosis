from dashboard.backend.tests.test_ai_assistant import app  # noqa: F401

from uav_dashboard_api.services.knowledge_base import knowledge_base


def test_index_becomes_stale_when_embedding_model_changes(app):
    with app.app_context():
        index = {
            "documents": [],
            "embedding_model": "BAAI/bge-m3",
            "embedding_provider": "openai_compatible",
            "requested_retrieval_mode": "hybrid",
        }

        app.config["AI_EMBEDDING_MODEL"] = "Qwen/Qwen3-Embedding-8B"
        app.config["AI_PROVIDER"] = "openai_compatible"
        app.config["AI_RAG_MODE"] = "hybrid"

        assert knowledge_base._index_is_stale(index) is True
