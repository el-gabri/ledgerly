"""Concierge: greetings and clarification menus."""
from __future__ import annotations

from ledgerly.llm import OfflineBackend
from ledgerly.state import Intent


def test_greeting_classification():
    backend = OfflineBackend()
    for text in ("Hi", "hello!", "Hey there", "Good morning"):
        assert backend.classify_intent(text) is Intent.GREETING
    # Word boundaries: "hi" inside a word must not fire.
    assert backend.classify_intent("this is broken") is not Intent.GREETING


def test_greeting_gets_a_greeting_back(conversation):
    """A 'Hi' gets a polite hello + 'how can I help', not a vendor hedge."""
    state = conversation("Hi")
    reply = state["messages"][-1]
    assert reply.agent == "concierge"
    assert "how can i help" in reply.content.lower()
    # A greeting is a fully correct reply: it must not feed the streak.
    assert state["low_confidence_streak"] == 0
    assert state["conv_state"] == "RESPONDED"


def test_greeting_with_substance_routes_on_substance(conversation):
    """'Hi, I was charged twice' is a billing question, not small talk."""
    state = conversation("Hi, I was charged twice")
    assert state["current_intent"] == "billing"


def test_unclear_intent_gets_capability_menu(conversation):
    """When intent can't be established, the user gets options — not a hedge."""
    state = conversation("ehh the thing isn't")
    reply = state["messages"][-1]
    assert reply.agent == "concierge"
    for option in ("Billing", "account", "How-to", "human"):
        assert option in reply.content
    # A clarification is not a resolution: it counts toward the streak...
    assert state["low_confidence_streak"] == 1


def test_persistently_unclear_user_reaches_a_human(conversation):
    """...so two unclear turns in a row still escalate."""
    conversation("ehh the thing isn't")
    state = conversation("you know, that one")
    assert state["human_active"] is True
    assert state["escalation"].trigger == "low_confidence"
