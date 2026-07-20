"""Concierge: orchestrator-owned turns that need no downstream agent.

Two cases live here, both cheap and deterministic:

- **Greetings** — "Hi" deserves "Hi, how can I help you?", not a vendor
  invocation. High confidence: this IS the correct reply.
- **Unclear intent** — instead of a vague hedge, the user gets a menu of what
  the system can actually do. The menu deliberately reports LOW confidence:
  a clarification is not a resolution, so a user who stays unclear twice
  escalates to a human through the normal low-confidence streak trigger.
"""
from __future__ import annotations

from ..logging_utils import log_event
from ..state import ConvState, DraftReply, Intent, OrchestratorState, transition

_GREETING_REPLY = (
    "Hi! Welcome to Ledgerly support — happy to help. "
    "How can I help you today?"
)

_CLARIFICATION_MENU = (
    "I want to make sure I point you in the right direction. "
    "Here's what I can help with:\n"
    "  1. Billing and fees — charges, refunds, disputes\n"
    "  2. Your account — balance, transactions, card status\n"
    "  3. How-to guides — passwords, transfers, closing an account\n"
    "  4. Product questions — limits, supported countries, currencies\n"
    "You can also say \"talk to a human\" at any time and I'll connect you "
    "with a specialist. What would you like to do?"
)


def concierge_node(state: OrchestratorState) -> dict:
    """Answer greetings and unclear turns directly from the orchestrator."""
    if state.get("current_intent") == Intent.GREETING.value:
        draft = DraftReply(agent="concierge", content=_GREETING_REPLY, confidence=0.95)
    else:  # unknown intent -> capability menu, low confidence on purpose
        draft = DraftReply(agent="concierge", content=_CLARIFICATION_MENU, confidence=0.45)

    events = [
        transition(state.get("conv_state", "ROUTING"), ConvState.AGENT_ACTIVE,
                   "handled by orchestrator concierge"),
        transition(ConvState.AGENT_ACTIVE.value, ConvState.GATING,
                   "concierge produced a draft reply"),
    ]
    log_event("concierge_reply", state, intent=state.get("current_intent"),
              confidence=draft.confidence)
    return {"conv_state": ConvState.GATING.value, "events": events, "draft": draft}
