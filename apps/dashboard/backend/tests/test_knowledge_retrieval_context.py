"""Synthetic retrieval contracts; all embedding/chat providers are local fakes."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

from flask import Flask
import pytest

from tests.test_knowledge_base_rag import app, auth_header, client  # noqa: F401
from uav_dashboard_api.models.ai_conversation import AIMessage
from uav_dashboard_api.services.ai_assistant import ai_assistant_service
from uav_dashboard_api.services.ai_provider import AIProviderError
from uav_dashboard_api.services.knowledge_base import KnowledgeBaseService, knowledge_base


GOLDEN = Path(__file__).parent / "fixtures/knowledge_retrieval_legacy.json"
SCENARIOS = [
    "bm25", "bm25_positive", "semantic", "hybrid", "candidate_one", "candidate_two",
    "quality_before_threshold", "rounding", "source_order", "ties",
    "context_one", "all_low",
]


class EmbeddingAdapter:
    is_embedding_configured = True

    def __init__(self, outcome="vector"):
        self.outcome = outcome
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(texts)
        if self.outcome == "error":
            raise AIProviderError("synthetic embedding failure")
        if self.outcome == "empty":
            return {"embeddings": []}
        vector = [0.0, 0.0] if self.outcome == "orthogonal" else [1.0, 0.0]
        return {"embeddings": [vector]}


def scenario(name):
    mode = name if name in {"bm25", "semantic", "hybrid"} else "semantic"
    documents = [
        ("bad", "bad", "motor \ue010", "motor motor motor maintenance", 0.99),
        ("clean", "clean", "motor guide", "motor bearing maintenance", 0.80),
        ("third", "third", "motor notes", "motor winding inspection", 0.65),
    ]
    limit, maximum = 5, 2
    if name == "bm25_positive":
        mode = "bm25"
        documents += [
            (f"weather-{i}", f"weather-{i}", "weather", "weather flight conditions", 0.0)
            for i in range(3)
        ]
    elif name in {"candidate_one", "candidate_two"}:
        limit = 1 if name == "candidate_one" else 2
    elif name == "quality_before_threshold":
        documents = [documents[0], (*documents[1][:-1], 0.55), (*documents[2][:-1], 0.54)]
    elif name == "rounding":
        documents = [
            ("round-up", "one", "motor guide", "motor maintenance", 0.34996),
            ("round-down", "two", "motor notes", "motor inspection", 0.34994),
        ]
    elif name == "source_order":
        documents = [
            ("first", "shared", "motor first", "motor first section", 0.50),
            ("later", "shared", "motor later", "motor later section", 0.99),
            ("other", "other", "motor guide", "motor maintenance", 0.80),
        ]
    elif name == "ties":
        documents = [
            ("one", "one", "motor first", "motor maintenance", 0.70),
            ("two", "two", "motor second", "motor inspection", 0.70),
        ]
    elif name == "context_one":
        maximum = 1
    elif name == "all_low":
        documents = [("low", "low", "motor guide", "motor maintenance", 0.30)]
    return mode, documents, limit, maximum


def make_service(root, mode="hybrid", documents=None, adapter=None):
    if documents is None:
        documents = scenario("hybrid")[1]
    service = KnowledgeBaseService()
    config = {
        "AI_ENABLE_KNOWLEDGE_BASE": True, "AI_RAG_MODE": mode,
        "AI_KNOWLEDGE_DIR": str(root / "knowledge"),
    }
    service._config = lambda: config
    chunks, frequencies = [], Counter()
    for chunk_id, source, title, text, cosine in documents:
        tokens = service._tokenize(text)
        frequencies.update(set(tokens))
        chunks.append({
            "chunk_id": chunk_id, "source_type": "knowledge_base",
            "source_id": source, "title": title, "path": "manuals/motor.md",
            "text": text, "token_counts": dict(Counter(tokens)),
            "token_len": len(tokens),
            "embedding": [cosine, math.sqrt(1 - cosine * cosine)],
        })
    index = {
        "documents": [], "chunks": chunks, "df": dict(frequencies),
        "avg_token_len": sum(chunk["token_len"] for chunk in chunks) / max(1, len(chunks)),
        "retrieval_mode": mode, "requested_retrieval_mode": mode,
        "embedding_enabled": mode != "bm25",
    }
    provider = adapter or EmbeddingAdapter()
    service._load_index = lambda: index
    service._embedding_client = lambda: provider
    return service, index, config, provider


@pytest.mark.parametrize("case", SCENARIOS)
def test_hits_citations_and_prompt_context_match_pre_refactor_snapshot(tmp_path, case):
    mode, documents, limit, maximum = scenario(case)
    service, _, _, _ = make_service(tmp_path, mode, documents)
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"][case]
    with Flask("retrieval-contract").app_context():
        candidates = service.search("motor", top_k=limit)
        result = service.retrieve_for_interpretation("motor", candidate_limit=limit, max_items=maximum)
    assert candidates == expected["candidates"]
    assert result.hits == expected["hits"]
    assert json.dumps(result.hits, sort_keys=True) == json.dumps(expected["hits"], sort_keys=True)
    assert ai_assistant_service._format_knowledge_context(result.hits) == expected["prompt_blocks"]
    assert result.actual_mode == mode
    assert result.index_embedding_enabled is (mode != "bm25")


@pytest.mark.parametrize("outcome", ["error", "empty", "orthogonal", "unconfigured", "index_disabled"])
@pytest.mark.parametrize("query,has_hits", [("motor", True), ("unmatched", False)])
def test_query_fallback_metadata_survives_zero_hits(tmp_path, outcome, query, has_hits):
    adapter = EmbeddingAdapter(outcome)
    if outcome == "unconfigured":
        adapter.is_embedding_configured = False
    documents = scenario("bm25_positive")[1]
    service, index, _, _ = make_service(tmp_path, documents=documents, adapter=adapter)
    if outcome == "index_disabled":
        index["embedding_enabled"] = False
    with Flask("query-fallback").app_context():
        result = service.retrieve_for_interpretation(query)
    assert bool(result.hits) is has_hits
    assert result.actual_mode == "bm25"
    assert result.index_embedding_enabled is (outcome != "index_disabled")
    assert index["retrieval_mode"] == "hybrid"
    assert all(hit["retrieval_mode"] == "bm25" for hit in result.hits)


@pytest.mark.parametrize("case", ["disabled", "empty_query", "empty_index"])
def test_skipped_queries_have_no_actual_mode_or_embedding_request(tmp_path, case):
    service, index, config, provider = make_service(tmp_path)
    query = "motor"
    if case == "disabled":
        config["AI_ENABLE_KNOWLEDGE_BASE"] = False
    elif case == "empty_query":
        query = " \n "
    else:
        index["chunks"] = []
    if case != "empty_index":
        service._load_index = lambda: pytest.fail("Skipped query loaded an index")
    result = service.retrieve_for_interpretation(query)
    assert result.hits == [] and result.actual_mode is None
    assert result.index_embedding_enabled is (case == "empty_index")
    assert provider.calls == []


def test_metadata_is_local_to_each_result(tmp_path):
    provider = EmbeddingAdapter()
    service, _, _, _ = make_service(tmp_path, adapter=provider)
    first = service.retrieve_for_interpretation("motor")
    provider.outcome = "empty"
    second = service.retrieve_for_interpretation("unmatched")
    assert first.actual_mode == "hybrid" and first.hits
    assert second.actual_mode == "bm25" and second.hits == []
    assert first.hits is not second.hits


def test_concurrent_queries_keep_their_own_metadata(tmp_path):
    class PerQueryAdapter(EmbeddingAdapter):
        def embed_texts(self, texts):
            return {"embeddings": [[1.0, 0.0]] if texts[0] == "motor" else []}

    service, _, _, _ = make_service(tmp_path, adapter=PerQueryAdapter())
    flask_app = Flask("concurrent-retrieval")

    def retrieve(query):
        with flask_app.app_context():
            return service.retrieve_for_interpretation(query)

    with ThreadPoolExecutor(max_workers=2) as executor:
        positive, empty = list(executor.map(retrieve, ["motor", "unmatched"]))
    assert positive.actual_mode == "hybrid" and positive.hits
    assert empty.actual_mode == "bm25" and empty.hits == []


@pytest.mark.parametrize("answer_fallback", [False, True])
def test_chat_records_actual_fallback_mode_without_reloading_index(
    app, client, auth_header, monkeypatch, tmp_path, answer_fallback,
):
    _, index, _, _ = make_service(tmp_path)
    loads = []

    def load_index():
        loads.append(True)
        return index

    monkeypatch.setattr(knowledge_base, "_load_index", load_index)
    monkeypatch.setattr(knowledge_base, "_embedding_client", lambda: EmbeddingAdapter("error"))
    monkeypatch.setattr(knowledge_base, "retrieval_info", lambda: pytest.fail("Chat reloaded index metadata"))
    monkeypatch.setattr(ai_assistant_service, "_call_model", lambda *args, **kwargs: (
        "Synthetic answer", "test-provider", "test-chat", answer_fallback,
    ))
    headers, _ = auth_header
    response = client.post("/api/ai/chat", headers=headers, json={"message": "unmatched", "scene": "general"})
    assert response.status_code == 200
    data = response.get_json()["data"]
    used = data["used_context"]
    assert used["knowledge_hits"] == 0
    assert used["knowledge_retrieval_mode"] == "bm25"
    assert used["knowledge_embedding_enabled"] is True
    assert used["fallback_used"] is answer_fallback
    assert data["citations"] == []
    assert len(loads) == 1
    assert AIMessage.query.filter_by(role="assistant").one().get_used_context() == used


@pytest.mark.parametrize("case", ["bm25_positive", "semantic", "hybrid"])
def test_chat_uses_unchanged_citations_and_prompt_context(
    app, client, auth_header, monkeypatch, tmp_path, case,
):
    mode, documents, _, _ = scenario(case)
    _, index, _, provider = make_service(tmp_path, mode, documents)
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"][case]
    monkeypatch.setattr(knowledge_base, "_load_index", lambda: index)
    monkeypatch.setattr(knowledge_base, "_embedding_client", lambda: provider)
    monkeypatch.setattr(knowledge_base, "retrieval_info", lambda: pytest.fail("Chat reloaded index metadata"))
    captured = []

    def answer(messages, **kwargs):
        captured.extend(message["content"] for message in messages)
        assert kwargs["knowledge_hits"] == expected["hits"]
        return "Synthetic answer", "test-provider", "test-chat", False

    monkeypatch.setattr(ai_assistant_service, "_call_model", answer)
    headers, _ = auth_header
    response = client.post("/api/ai/chat", headers=headers, json={"message": "motor", "scene": "general"})
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["citations"] == data["message"]["citations"] == expected["hits"]
    assert data["used_context"]["knowledge_retrieval_mode"] == mode
    assert "\n\n".join(expected["prompt_blocks"]) in "\n\n".join(captured)
