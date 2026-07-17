"""Human handoff: restricted intents and the context package."""
from __future__ import annotations

from conftest import states_visited


def test_fraud_claim_goes_straight_to_human(conversation):
    """Restricted intent: no AI agent ever handles the turn."""
    state = conversation("There's a charge on my card I didn't make")
    assert state["human_active"] is True
    assert state["escalation"].trigger == "restricted_intent"
    seq = states_visited(state)
    assert "AGENT_ACTIVE" not in seq  # rule fired before any agent ran
    assert seq[-1] == "HUMAN_ACTIVE"


def test_unrecognized_purchase_is_a_fraud_claim(conversation):
    """Regression: 'I don't recognize a purchase' must hit the fraud rule,
    not drift to the vendor as an unknown intent."""
    state = conversation("I don't recognize a purchase")
    assert state["current_intent"] == "fraud_claim"
    assert state["human_active"] is True
    assert state["escalation"].trigger == "restricted_intent"


def test_dispute_routes_as_billing(conversation):
    state = conversation("I want to dispute a transaction")
    assert state["current_intent"] == "billing"
    assert state["conv_state"] == "RESPONDED"


def test_explicit_human_request_is_honored(conversation):
    state = conversation("I want to speak to a real person")
    assert state["human_active"] is True
    assert state["escalation"].trigger == "user_requested_human"


def test_context_package_contents(conversation):
    """The human agent starts warm: summary, transcript, attempts, actions."""
    conversation("My payment is not working")
    state = conversation("This is ridiculous, it's still not working")

    package = state["escalation"].package
    assert package["trigger"] == "user_frustration"
    assert "Escalated because" in package["summary"] or package["summary"]
    assert len(package["transcript"]) >= 3  # 2 user turns + at least 1 reply
    assert package["intents_seen"]  # intent history is included
    assert package["suggested_actions"]  # human gets next-step guidance
    assert all({"role", "content"} <= set(m) for m in package["transcript"])


def test_user_is_told_about_the_handoff(conversation):
    state = conversation("I need to talk to a human")
    final = state["messages"][-1]
    assert final.role == "assistant"
    assert "specialist" in final.content.lower()
