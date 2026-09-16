"""Concierge: greetings and clarification menus."""
from __future__ import annotations

import pytest

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


@pytest.mark.parametrize("choice, intent", [
    ("1", "billing"), ("2", "account"), ("3", "how_to"), ("4", "product"),
])
def test_menu_selection_routes_without_an_unnecessary_handoff(conversation, choice, intent):
    """A numbered answer to the displayed menu is valid user context."""
    state = conversation("GABRIEL")
    assert state["awaiting_menu_selection"] is True

    state = conversation(choice)
    assert state["current_intent"] == intent
    assert state["selected_category"] == intent
    assert state["messages"][-1].agent == "concierge"
    assert "?" in state["messages"][-1].content
    assert state["low_confidence_streak"] == 0
    assert state.get("human_active") is not True


@pytest.mark.parametrize("choice, question, agent, answer", [
    ("2", "transactions please", "account", "Grocery Mart"),
    ("2", "What's my balance?", "account", "1,284.50"),
    ("4", "countries please", "kb", "Source: supported-countries"),
    ("4", "What are the transfer limits?", "kb", "2,500 USD"),
])
def test_menu_category_supports_a_concrete_followup(conversation, choice, question, agent, answer):
    conversation("GABRIEL")
    conversation(choice)
    state = conversation(question)
    assert state["conv_state"] == "RESPONDED"
    assert state["messages"][-1].agent == agent
    assert answer in state["messages"][-1].content
    assert state["selected_category"] is None


def test_selected_category_does_not_override_an_intent_shift(conversation):
    conversation("GABRIEL")
    conversation("2")
    state = conversation("What are the transfer limits?")
    assert state["current_intent"] == "product"
    assert state["messages"][-1].agent == "kb"
    assert state["selected_category"] is None


def test_unclear_followups_after_menu_selection_still_escalate(conversation):
    conversation("GABRIEL")
    conversation("2")
    state = conversation("zxcv mumble jumble")
    assert state["selected_category"] == "account"
    assert state["low_confidence_streak"] == 1
    state = conversation("qwerty flibber jabber")
    assert state["escalation"].trigger == "low_confidence"


def test_weak_substantive_answers_after_menu_selection_still_escalate(conversation):
    conversation("GABRIEL")
    conversation("4")
    state = conversation("What is zxcv qwerty?")
    assert state["low_confidence_streak"] == 1
    state = conversation("What is flibber jabber?")
    assert state["escalation"].trigger == "low_confidence"


def test_wallet_question_routes_to_account_agent(conversation):
    state = conversation("Hi. How can I check my wallet?")
    assert state["current_intent"] == "account"
    assert state["messages"][-1].agent == "account"
    assert "1,284.50" in state["messages"][-1].content


def test_persistently_unclear_user_reaches_a_human(conversation):
    """...so two unclear turns in a row still escalate."""
    conversation("ehh the thing isn't")
    state = conversation("you know, that one")
    assert state["human_active"] is True
    assert state["escalation"].trigger == "low_confidence"
