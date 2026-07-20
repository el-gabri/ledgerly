"""Intent router: rule layer first, then model classification.

The rule layer is deterministic and runs before any LLM: restricted intents
(fraud claims, legal threats) and explicit human requests are policy
decisions, and policy must be auditable — a regex you can point at beats a
prompt you can only hope about.
"""
from __future__ import annotations

from .llm import LLMBackend, detect_frustration
from .logging_utils import log_event
from .state import (
    RESTRICTED_INTENTS,
    ConvState,
    Intent,
    OrchestratorState,
    last_user_message,
    transition,
)

_RULE_PATTERNS: list[tuple[Intent, list[str]]] = [
    (Intent.FRAUD_CLAIM, [
        "unauthorized", "didn't make", "did not make", "didnt make", "stolen",
        "fraud", "hacked", "someone charged", "not my charge",
        "don't recognize", "do not recognize", "dont recognize",
        "unrecognized", "never authorized",
    ]),
    (Intent.LEGAL_THREAT, [
        "lawyer", "attorney", "sue ", "suing", "legal action", "lawsuit",
    ]),
    (Intent.HUMAN_REQUEST, [
        "human", "real person", "real agent", "representative",
        "speak to someone", "talk to someone", "speak to a person",
    ]),
]

#: Which agent serves which intent. GREETING and UNKNOWN stay inside the
#: orchestrator (concierge): a greeting doesn't warrant a vendor call, and an
#: unclear message gets a capability menu instead of a hedge.
_INTENT_TO_AGENT = {
    Intent.ACCOUNT: "account",
    Intent.PRODUCT: "kb",
    Intent.BILLING: "vendor",
    Intent.HOW_TO: "vendor",
    Intent.COMPLAINT: "vendor",
    Intent.GREETING: "concierge",
    Intent.UNKNOWN: "concierge",
}


def _apply_rules(text: str) -> Intent | None:
    """Deterministic pre-classification. Returns None if no rule matches."""
    lowered = text.lower()
    for intent, patterns in _RULE_PATTERNS:
        if any(p in lowered for p in patterns):
            return intent
    return None


def make_router_node(backend: LLMBackend):
    """Build the router node bound to an LLM backend."""

    def router_node(state: OrchestratorState) -> dict:
        text = last_user_message(state)
        events = [transition(state.get("conv_state", "INTAKE"), ConvState.ROUTING,
                             "classifying user turn")]

        rule_intent = _apply_rules(text)
        intent = rule_intent or backend.classify_intent(text)
        decided_by = "rule" if rule_intent else backend.name

        update: dict = {
            "conv_state": ConvState.ROUTING.value,
            "events": events,
            "current_intent": intent.value,
            "intent_history": [intent.value],
        }

        # Frustration is tracked independently of intent: an angry message
        # about billing is still a billing question — and a signal.
        if detect_frustration(text):
            update["frustration_count"] = state.get("frustration_count", 0) + 1

        if intent in RESTRICTED_INTENTS:
            update["pending_escalation"] = (
                "restricted_intent",
                f"'{intent.value}' is restricted: AI agents may not handle it",
            )
        elif intent is Intent.HUMAN_REQUEST:
            update["pending_escalation"] = (
                "user_requested_human",
                "user explicitly asked for a human agent",
            )
        else:
            update["active_agent"] = _INTENT_TO_AGENT[intent]

        log_event(
            "routing_decision", state,
            intent=intent.value, decided_by=decided_by,
            target=update.get("active_agent", "human"),
            frustration_count=update.get("frustration_count",
                                         state.get("frustration_count", 0)),
        )
        return update

    return router_node


def route_after_router(state: OrchestratorState) -> str:
    """Conditional edge: dispatch to the chosen agent or straight to handoff."""
    if state.get("pending_escalation"):
        return "escalate"
    return state["active_agent"]
