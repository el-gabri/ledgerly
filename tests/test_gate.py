"""Response Gate: escalation trigger logic."""
from __future__ import annotations

from ledgerly.agents.vendor import VendorResult
from ledgerly.graph import build_app, new_conversation_id, run_turn


class _OverconfidentEmptyVendor:
    """Returns a response the gate must reject despite its confidence."""

    name = "overconfident_empty_vendor"

    def invoke(self, projection, chaos=None):
        return VendorResult(ok=True, content="", confidence=0.99)


def test_empty_vendor_response_escalates_despite_high_confidence(monkeypatch):
    monkeypatch.setenv("LEDGERLY_LLM_MODE", "offline")
    app = build_app(vendor_adapter=_OverconfidentEmptyVendor())

    state = run_turn(app, new_conversation_id(), "How do I reset my password?")

    assert state["human_active"] is True
    assert state["escalation"].trigger == "invalid_response"
    assert state["escalation"].package["agents_attempted"] == [{
        "agent": "overconfident_empty_vendor",
        "outcome": "reply",
        "confidence": 0.99,
    }]


def test_frustration_escalates_on_second_signal(conversation):
    """First frustrated message gets an empathetic reply; second escalates."""
    state = conversation("My payment is not working")
    assert state["frustration_count"] == 1
    assert state.get("human_active") is not True

    state = conversation("This is ridiculous, it's still not working")
    assert state["frustration_count"] == 2
    assert state["human_active"] is True
    assert state["escalation"].trigger == "user_frustration"


def test_low_confidence_streak_escalates(conversation):
    """Two consecutive unclassifiable queries -> vendor hedges twice -> human."""
    state = conversation("zxcv mumble jumble")
    assert state["low_confidence_streak"] == 1
    state = conversation("qwerty flibber jabber")
    assert state["human_active"] is True
    assert state["escalation"].trigger == "low_confidence"


def test_turn_limit_escalates(conversation):
    """A conversation that drags past the turn limit escalates as unresolved."""
    state = {}
    for _ in range(9):
        state = conversation("How do I reset my password?")
        if state.get("human_active"):
            break
    assert state["human_active"] is True
    assert state["escalation"].trigger == "turn_limit"


def test_confident_replies_never_escalate(conversation):
    for text in ("What are the transfer limits?", "What's my balance?",
                 "How do I reset my password?"):
        state = conversation(text)
        assert state.get("human_active") is not True
        assert state["gate_decision"] == "respond"
