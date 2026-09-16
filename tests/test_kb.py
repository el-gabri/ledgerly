"""Grounded offline answers and optional retrieval failure recovery."""
from __future__ import annotations

import json

import pytest

from ledgerly.agents import kb as kb_module
from ledgerly.agents.kb import KnowledgeBaseAgent
from ledgerly.graph import build_app, run_turn
from ledgerly.llm import OfflineBackend


@pytest.mark.parametrize("question, expected, article", [
    ("When do transfer limits reset?", "midnight UTC", "transfer-limits"),
    ("What is the policy on reopening a closed account?", "reopened within 30 days", "account-closure"),
    ("What are the available statement file formats?", "PDF and CSV", "statements-tax"),
])
def test_offline_answer_includes_supporting_later_paragraph(conversation, question, expected, article):
    state = conversation(question)
    assert state["conv_state"] == "RESPONDED"
    assert expected in state["messages"][-1].content
    assert state["draft"].citations == [article]
    assert f"(Source: {article})" in state["messages"][-1].content


@pytest.mark.parametrize("as_fallback", [False, True])
def test_dense_search_failure_retries_tfidf(monkeypatch, as_fallback):
    records = []
    queries = []

    class BrokenDenseIndex:
        def __init__(self, docs):
            assert len(docs) == 15

        def search(self, query, k):
            queries.append(query)
            raise RuntimeError("private query text")

    monkeypatch.setenv("LEDGERLY_EMBEDDINGS", "st")
    monkeypatch.setattr(kb_module, "_SentenceTransformerIndex", BrokenDenseIndex)
    monkeypatch.setattr(kb_module, "log_event", lambda event, state=None, **kw: records.append({"event": event, **kw}))
    backend = OfflineBackend()
    agent = KnowledgeBaseAgent(backend)
    app = build_app(backend=backend, kb_agent=agent)
    query = "How do I close my account?" if as_fallback else "What are the transfer limits?"
    state = run_turn(app, "dense-failure", query, chaos="vendor_timeout" if as_fallback else None)
    assert state["conv_state"] == "RESPONDED"
    assert state["messages"][-1].agent == "kb"
    assert state["draft"].citations == (["account-closure"] if as_fallback else ["transfer-limits"])
    assert state["fallback_attempted"] is as_fallback
    # The failed optional index is retired for subsequent queries.
    agent.answer(query)
    assert queries == [query]
    failure = next(record for record in records if record["event"] == "kb_embeddings_fallback")
    assert failure["stage"] == "search"
    assert failure["fallback"] == "tfidf"
    assert failure["error_type"] == "RuntimeError"
    assert "private query text" not in json.dumps(records)


def test_injected_document_directory_retains_body_and_weak_retrieval(tmp_path):
    (tmp_path / "custom.md").write_text("# Custom article\n\nZebras are striped.\n\nZebras live in herds.", encoding="utf-8")
    agent = KnowledgeBaseAgent(OfflineBackend(), docs_dir=tmp_path)
    draft = agent.answer("Zebras")
    assert "Zebras are striped." in draft.content
    assert "Zebras live in herds." in draft.content
    assert "# Custom article" not in draft.content
    assert draft.citations == ["custom"]
    weak = agent.answer("qwerty")
    assert weak.confidence < 0.5
    assert weak.citations == []
