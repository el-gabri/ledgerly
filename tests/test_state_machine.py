"""State-machine invariants: transitions follow DESIGN_DOC.md section 3."""
from __future__ import annotations

from conftest import states_visited


def test_happy_path_state_sequence(conversation):
    state = conversation("How do I reset my password?")
    seq = states_visited(state)
    assert seq == ["INTAKE", "ROUTING", "AGENT_ACTIVE", "GATING", "RESPONDED"]


def test_every_transition_has_a_reason(conversation):
    state = conversation("What are the transfer limits?")
    assert all(ev.reason for ev in state["events"])


def test_turn_counter_increments(conversation):
    conversation("How do I reset my password?")
    state = conversation("What are the transfer limits?")
    assert state["turn_count"] == 2


def test_human_active_short_circuits_ai(conversation):
    """Once HUMAN_ACTIVE, later turns never reach the router or any agent."""
    state = conversation("I want to talk to a human")
    assert state["human_active"] is True
    assert state["conv_state"] == "HUMAN_ACTIVE"

    n_events = len(state["events"])
    state = conversation("hello? are you still there?")
    # No new state-machine transitions: the orchestrator only records.
    assert len(state["events"]) == n_events
    assert state["conv_state"] == "HUMAN_ACTIVE"
    assert state["messages"][-1].agent == "orchestrator"


def test_per_turn_scratch_fields_reset(conversation):
    state = conversation("How do I close my account?", chaos="vendor_timeout")
    assert state["fallback_attempted"] is True
    state = conversation("What are the transfer limits?")
    assert state["fallback_attempted"] is False
    assert state["vendor_failure"] is None
