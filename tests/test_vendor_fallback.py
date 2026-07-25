"""Vendor failure modes: graceful degradation and escalation."""
from __future__ import annotations

from conftest import states_visited
from ledgerly.agents.vendor import build_projection
from ledgerly.state import Message


def test_vendor_projection_redacts_pii():
    projection = build_projection({
        "current_intent": "billing",
        "messages": [Message(
            role="user",
            content=("My email is alex@example.com; card 4242 4242 4242 4242; "
                     "CPF 123.456.789-00; phone +55 11 99876-1234"),
        )],
    })

    content = projection.transcript[0]["content"]
    assert "alex@example.com" not in content
    assert "4242 4242 4242 4242" not in content
    assert "123.456.789-00" not in content
    assert "99876-1234" not in content
    assert projection.redaction_types == ["card_number", "cpf", "email", "phone"]


def test_vendor_timeout_falls_back_to_kb(conversation):
    """Hard vendor failure: the internal KB agent answers instead."""
    state = conversation("How do I close my account?", chaos="vendor_timeout")
    seq = states_visited(state)
    assert "FALLBACK" in seq
    assert state["vendor_failure"].kind == "timeout"
    assert state["fallback_attempted"] is True
    # Conversation still succeeds: KB agent answered from the docs.
    assert state["messages"][-1].agent == "kb"
    assert state["conv_state"] == "RESPONDED"
    assert "account-closure" in state["draft"].citations


def test_vendor_failure_with_weak_fallback_escalates(conversation):
    """Vendor fails AND the fallback can't answer confidently -> human."""
    # A how-to question the KB corpus can't answer: vendor times out, the KB
    # fallback retrieves nothing useful, so the turn escalates.
    state = conversation("How do I do the thing with the stuff?", chaos="vendor_timeout")
    assert state["human_active"] is True
    assert state["escalation"].trigger == "vendor_exhausted"
    assert state["escalation"].package["agents_attempted"] == [
        {"agent": "mock_vendor_llm", "outcome": "failure", "failure_kind": "timeout"},
        {"agent": "kb", "outcome": "reply", "confidence": 0.30},
    ]


def test_vendor_low_confidence_counts_toward_streak(conversation):
    """Soft vendor failure: two weak replies in a row escalate."""
    state = conversation("How do I reset my password?", chaos="vendor_low_confidence")
    assert state["low_confidence_streak"] == 1
    assert state["conv_state"] == "RESPONDED"  # first weak reply still delivered

    state = conversation("How do I set up notifications?", chaos="vendor_low_confidence")
    assert state["human_active"] is True
    assert state["escalation"].trigger == "low_confidence"


def test_confident_reply_resets_streak(conversation):
    conversation("How do I reset my password?", chaos="vendor_low_confidence")
    state = conversation("What are the transfer limits?")  # KB, confident
    assert state["low_confidence_streak"] == 0
