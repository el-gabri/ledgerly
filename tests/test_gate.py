"""Response Gate: escalation trigger logic."""
from __future__ import annotations


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
