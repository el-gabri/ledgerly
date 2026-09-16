"""Deposit lookups use fixture evidence rather than statement instructions."""
from __future__ import annotations

import json

import pytest

from ledgerly.agents.account import AccountAgent


def test_deposit_request_asks_for_identification(conversation):
    state = conversation("What is the status of my deposit?")
    assert state["current_intent"] == "account"
    assert state["conv_state"] == "RESPONDED"
    assert "date" in state["messages"][-1].content
    assert "statement" not in state["messages"][-1].content.lower()
    assert state["draft"].confidence < 0.5


def test_identified_deposit_returns_recorded_status(conversation):
    state = conversation("What is the status of my deposit on 2026-07-12?")
    assert state["conv_state"] == "RESPONDED"
    content = state["messages"][-1].content
    for fact in ("Salary deposit", "2026-07-12", "+2,150.00", "settled"):
        assert fact in content
    assert state["draft"].confidence == 0.9


@pytest.mark.parametrize("query", [
    "My deposit on 2026-09-01", "My Salary deposit on 2026-09-01", "My bonus deposit",
])
def test_unknown_deposit_is_not_invented(query):
    draft = AccountAgent().answer(query)
    assert draft.confidence < 0.5
    assert "settled" not in draft.content


def test_statement_behavior_preserved(conversation):
    state = conversation("Where is my statement?")
    assert "Settings > Documents" in state["messages"][-1].content


def test_injected_fixture_and_ambiguous_deposit(tmp_path):
    path = tmp_path / "accounts.json"
    path.write_text(json.dumps({
        "currency": "USD", "balance": "27.00",
        "recent_transactions": [
            {"date": "2026-07-12", "description": "Salary deposit", "amount": "+100.00", "status": "settled"},
            {"date": "2026-08-12", "description": "Salary deposit", "amount": "+100.00", "status": "pending"},
        ],
    }), encoding="utf-8")
    agent = AccountAgent(fixtures_path=path)
    assert "27.00" in agent.answer("My balance").content
    assert agent.answer("My salary deposit").confidence < 0.5
    assert "pending" in agent.answer("My deposit on 2026-08-12").content
