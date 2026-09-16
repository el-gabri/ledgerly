"""Shared fixtures. All tests run in offline mode: deterministic, no network."""
from __future__ import annotations

import pytest

from ledgerly.graph import build_app, new_conversation_id, run_turn


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    """Do not inherit optional network modes from the developer's shell."""
    monkeypatch.setenv("LEDGERLY_LLM_MODE", "offline")
    monkeypatch.setenv("LEDGERLY_EMBEDDINGS", "tfidf")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


@pytest.fixture()
def app():
    """A fresh compiled graph in offline mode."""
    return build_app()


@pytest.fixture()
def conversation(app):
    """(app, conversation_id) pair plus a convenience turn runner."""
    cid = new_conversation_id()

    def turn(text: str, chaos: str | None = None) -> dict:
        return run_turn(app, cid, text, chaos=chaos)

    return turn


def states_visited(state: dict) -> list[str]:
    """Flatten the event log into the sequence of states entered."""
    return [ev.to_state for ev in state["events"]]
