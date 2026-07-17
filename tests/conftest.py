"""Shared fixtures. All tests run in offline mode: deterministic, no network."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledgerly.graph import build_app, new_conversation_id, run_turn  # noqa: E402


@pytest.fixture()
def app(monkeypatch):
    """A fresh compiled graph in offline mode."""
    monkeypatch.setenv("LEDGERLY_LLM_MODE", "offline")
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
