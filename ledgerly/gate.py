"""Response Gate: the single checkpoint every candidate reply passes.

No matter which agent produced a draft — vendor, KB, or account — it flows
through this one node before reaching the user. That gives the platform one
auditable place where escalation policy lives, instead of policy fragments
scattered across agents.

Triggers are evaluated in a fixed, documented order (DESIGN_DOC.md §5).
"""
from __future__ import annotations

from .config import (
    FRUSTRATION_LIMIT,
    LOW_CONFIDENCE_STREAK_LIMIT,
    LOW_CONFIDENCE_THRESHOLD,
    TURN_LIMIT,
)
from .logging_utils import log_event
from .state import ConvState, OrchestratorState, transition


def gate_node(state: OrchestratorState) -> dict:
    """Evaluate escalation triggers against the current draft reply."""
    draft = state.get("draft")
    confidence = draft.confidence if draft else 0.0

    # Streak accounting happens before trigger evaluation so the current
    # draft counts toward its own streak.
    streak = state.get("low_confidence_streak", 0)
    streak = streak + 1 if confidence < LOW_CONFIDENCE_THRESHOLD else 0

    update: dict = {"low_confidence_streak": streak}
    trigger = None

    # Ordered trigger evaluation — first match wins.
    if state.get("fallback_attempted") and confidence < LOW_CONFIDENCE_THRESHOLD:
        trigger = ("vendor_exhausted",
                   "vendor failed and the internal fallback is not confident")
    elif streak >= LOW_CONFIDENCE_STREAK_LIMIT:
        trigger = ("low_confidence",
                   f"{streak} consecutive replies below confidence "
                   f"{LOW_CONFIDENCE_THRESHOLD}")
    elif state.get("frustration_count", 0) >= FRUSTRATION_LIMIT:
        trigger = ("user_frustration",
                   f"{state['frustration_count']} frustration signals detected")
    elif state.get("turn_count", 0) > TURN_LIMIT:
        trigger = ("turn_limit",
                   f"conversation exceeded {TURN_LIMIT} turns without resolution")

    if trigger:
        update["gate_decision"] = "escalate"
        update["pending_escalation"] = trigger
    else:
        update["gate_decision"] = "respond"

    log_event(
        "gate_decision", state,
        decision=update["gate_decision"],
        agent=draft.agent if draft else None,
        confidence=confidence,
        low_confidence_streak=streak,
        trigger=trigger[0] if trigger else None,
    )
    return update


def route_after_gate(state: OrchestratorState) -> str:
    """Conditional edge: deliver the reply or hand off to a human."""
    return "escalate" if state.get("gate_decision") == "escalate" else "respond"
