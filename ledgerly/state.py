"""Typed conversation state shared by every node in the orchestration graph.

This module is the "context contract" described in DESIGN_DOC.md section 4:
a single typed state object flows through the graph, and every agent reads
from it but writes only its own fields. Changing this file is a design
decision, not a casual diff.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional, TypedDict


class ConvState(str, Enum):
    """States of the conversation state machine (DESIGN_DOC.md section 3)."""

    INTAKE = "INTAKE"
    ROUTING = "ROUTING"
    AGENT_ACTIVE = "AGENT_ACTIVE"
    FALLBACK = "FALLBACK"
    GATING = "GATING"
    RESPONDED = "RESPONDED"
    ESCALATING = "ESCALATING"
    HUMAN_ACTIVE = "HUMAN_ACTIVE"
    RESOLVED = "RESOLVED"


class Intent(str, Enum):
    """Intents the router can assign to a user turn."""

    BILLING = "billing"
    HOW_TO = "how_to"
    PRODUCT = "product"
    ACCOUNT = "account"
    COMPLAINT = "complaint"
    HUMAN_REQUEST = "human_request"
    FRAUD_CLAIM = "fraud_claim"
    LEGAL_THREAT = "legal_threat"
    UNKNOWN = "unknown"


#: Intents that must NEVER be handled by an AI agent. The rule layer routes
#: these straight to a human before any LLM sees the conversation.
RESTRICTED_INTENTS = frozenset({Intent.FRAUD_CLAIM, Intent.LEGAL_THREAT})


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Message:
    """A single chat message. `agent` records which agent produced a reply."""

    role: str  # "user" | "assistant"
    content: str
    agent: Optional[str] = None


@dataclass
class TransitionEvent:
    """One edge traversal in the conversation state machine (the audit log)."""

    from_state: str
    to_state: str
    reason: str
    at: str = field(default_factory=_utcnow)


@dataclass
class DraftReply:
    """A candidate reply produced by an agent, pending the Response Gate."""

    agent: str
    content: str
    confidence: float
    citations: list[str] = field(default_factory=list)


@dataclass
class VendorFailure:
    """A failure reported by (or detected in) the vendor AI adapter."""

    kind: str  # "timeout" | "error" | "low_confidence"
    detail: str


@dataclass
class Escalation:
    """A human handoff: which trigger fired and the context package."""

    trigger: str
    reason: str
    package: dict


class OrchestratorState(TypedDict, total=False):
    """The shared graph state.

    Fields annotated with ``operator.add`` are append-only logs: nodes emit
    new entries and LangGraph merges them. All other fields are last-write.
    Per-turn scratch fields (draft, vendor_failure, ...) are reset by the
    intake node at the start of every turn.
    """

    conversation_id: str
    # -- append-only logs -------------------------------------------------
    messages: Annotated[list[Message], operator.add]
    events: Annotated[list[TransitionEvent], operator.add]
    intent_history: Annotated[list[str], operator.add]
    # -- durable conversation-level fields --------------------------------
    conv_state: str
    turn_count: int
    low_confidence_streak: int
    frustration_count: int
    escalation: Optional[Escalation]
    human_active: bool
    # -- per-turn scratch fields (reset at intake) ------------------------
    current_intent: Optional[str]
    active_agent: Optional[str]
    draft: Optional[DraftReply]
    vendor_failure: Optional[VendorFailure]
    fallback_attempted: bool
    pending_escalation: Optional[tuple]  # (trigger, reason)
    gate_decision: Optional[str]
    chaos: Optional[str]  # failure-injection knob, consumed by the vendor node


def transition(from_state: str, to_state: ConvState, reason: str) -> TransitionEvent:
    """Build a state-machine transition event for the audit log."""
    return TransitionEvent(from_state=from_state, to_state=to_state.value, reason=reason)


def last_user_message(state: OrchestratorState) -> str:
    """Return the content of the most recent user message ('' if none)."""
    for msg in reversed(state.get("messages", [])):
        if msg.role == "user":
            return msg.content
    return ""
