"""Routing: intent classification and dispatch decisions."""
from __future__ import annotations

import pytest

from ledgerly.llm import OfflineBackend
from ledgerly.router import _apply_rules
from ledgerly.state import Intent


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I was charged a fee I don't understand", Intent.BILLING),
        ("How do I reset my password?", Intent.HOW_TO),
        ("What are the transfer limits?", Intent.PRODUCT),
        ("What's my balance?", Intent.ACCOUNT),
        ("This app is terrible and not working", Intent.COMPLAINT),
        ("asdf qwerty zzz", Intent.UNKNOWN),
    ],
)
def test_offline_classification(text, expected):
    assert OfflineBackend().classify_intent(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("There's a charge I didn't make", Intent.FRAUD_CLAIM),
        ("My card was stolen", Intent.FRAUD_CLAIM),
        ("I will contact my lawyer about this", Intent.LEGAL_THREAT),
        ("Let me talk to a human please", Intent.HUMAN_REQUEST),
        ("How do I reset my password?", None),  # no rule fires; model decides
    ],
)
def test_rule_layer(text, expected):
    """Restricted intents and human requests are rule decisions, not model ones."""
    assert _apply_rules(text) is expected or _apply_rules(text) == expected


def test_account_question_routes_to_account_agent(conversation):
    state = conversation("What's my current balance?")
    assert state["current_intent"] == "account"
    assert state["messages"][-1].agent == "account"
    assert "1,284.50" in state["messages"][-1].content


def test_product_question_routes_to_kb_agent(conversation):
    state = conversation("What are the transfer limits?")
    assert state["current_intent"] == "product"
    assert state["messages"][-1].agent == "kb"
    assert "transfer-limits" in state["draft"].citations


def test_howto_question_routes_to_vendor(conversation):
    state = conversation("How do I reset my password?")
    assert state["current_intent"] == "how_to"
    assert state["messages"][-1].agent == "mock_vendor_llm"


def test_intent_shift_mid_conversation(conversation):
    """Routing happens per turn: the same conversation moves across agents."""
    conversation("How do I reset my password?")
    state = conversation("What's my balance?")
    assert state["intent_history"] == ["how_to", "account"]
    assert state["messages"][-1].agent == "account"
