"""Human handoff: build the context package and transfer ownership.

The metric that matters for handoffs is "the customer never repeats
themselves": the human agent receives a summary, the full transcript, every
agent already attempted (with confidence scores), the trigger that fired,
and suggested next actions — a warm start, not a cold transfer.
"""
from __future__ import annotations

from .llm import LLMBackend
from .logging_utils import log_event
from .state import ConvState, Escalation, Message, OrchestratorState, transition

_SUGGESTED_ACTIONS = {
    "invalid_response": [
        "The AI response failed a basic integrity check; do not rely on it",
        "Answer the customer's original question from the transcript",
    ],
    "restricted_intent": [
        "Verify the customer's identity before discussing the claim",
        "Follow the fraud/legal escalation runbook",
        "Do not reference AI-generated content in the case notes",
    ],
    "user_requested_human": [
        "Acknowledge the request immediately",
        "Review the transcript for unresolved questions",
    ],
    "vendor_exhausted": [
        "Vendor AI failed this turn; treat AI context as incomplete",
        "Re-ask the customer's core question in your own words",
    ],
    "low_confidence": [
        "AI agents could not answer confidently; likely an uncovered topic",
        "Consider filing a KB gap report after resolution",
    ],
    "user_frustration": [
        "Open with an apology and a direct commitment",
        "Prioritize resolution speed over process",
    ],
    "turn_limit": [
        "Long unresolved conversation; read the summary before the transcript",
        "Identify why earlier answers did not resolve the issue",
    ],
}


def _transcript_from_state(state: OrchestratorState) -> list[dict]:
    return [
        {"role": message.role, "agent": message.agent, "content": message.content}
        for message in state.get("messages", [])
    ]


def build_package(state: OrchestratorState, trigger: str, reason: str,
                  backend: LLMBackend) -> dict:
    """Assemble the structured context package a human agent receives."""
    transcript = _transcript_from_state(state)
    attempts = []
    for attempt in state.get("agent_attempts", []):
        item = {"agent": attempt.agent, "outcome": attempt.outcome}
        if attempt.confidence is not None:
            item["confidence"] = attempt.confidence
        if attempt.citations:
            item["citations"] = attempt.citations
        if attempt.failure_kind:
            item["failure_kind"] = attempt.failure_kind
        attempts.append(item)
    intents = state.get("intent_history", [])

    fallback_summary = (
        f"Conversation of {state.get('turn_count', 0)} turn(s). "
        f"Intents seen: {', '.join(intents) or 'none'}. "
        f"Escalated because: {reason}."
    )
    summary = backend.generate(
        system="Summarize this support conversation for the human agent taking over. Two sentences, factual.",
        prompt="\n".join(f"{m['role']}: {m['content']}" for m in transcript),
        fallback=fallback_summary,
    )

    return {
        "summary": summary,
        "transcript": transcript,
        "intents_seen": intents,
        "agents_attempted": attempts,
        "trigger": trigger,
        "trigger_reason": reason,
        "suggested_actions": _SUGGESTED_ACTIONS.get(trigger, []),
    }


def make_escalate_node(backend: LLMBackend):
    """Build the escalation node: package context, transfer to human."""

    def escalate_node(state: OrchestratorState) -> dict:
        trigger, reason = state.get("pending_escalation") or (
            "unknown", "escalation requested without a recorded trigger")
        package = build_package(state, trigger, reason, backend)

        current = state.get("conv_state", "GATING")
        events = [
            transition(current, ConvState.ESCALATING, f"trigger: {trigger}"),
            transition(ConvState.ESCALATING.value, ConvState.HUMAN_ACTIVE,
                       "context package delivered to human queue"),
        ]
        log_event("human_handoff", state, trigger=trigger, reason=reason,
                  package_summary=package["summary"])

        return {
            "conv_state": ConvState.HUMAN_ACTIVE.value,
            "events": events,
            "escalation": Escalation(trigger=trigger, reason=reason, package=package),
            "human_active": True,
            "messages": [Message(
                role="assistant",
                content=("I'm connecting you with a support specialist who has "
                         "the full context of our conversation. They'll take it "
                         "from here."),
                agent="orchestrator",
            )],
        }

    return escalate_node


def human_hold_node(state: OrchestratorState) -> dict:
    """Keep the handoff package current while the human owns the conversation."""
    update: dict = {}
    escalation = state.get("escalation")
    if escalation:
        package = dict(escalation.package)
        prior_transcript = package.get("transcript", [])
        transcript = _transcript_from_state(state)
        new_messages = transcript[len(prior_transcript):]
        package["transcript"] = transcript
        if new_messages:
            package["post_handoff_messages"] = [
                *package.get("post_handoff_messages", []),
                *new_messages,
            ]
        update["escalation"] = Escalation(
            trigger=escalation.trigger,
            reason=escalation.reason,
            package=package,
        )

    log_event("human_hold", state, note="conversation owned by human agent; AI muted",
              new_message_count=len(new_messages) if escalation else 0)
    return {
        **update,
        "messages": [Message(
            role="assistant",
            content="(A specialist is handling this conversation — your message has "
                    "been added to the case.)",
            agent="orchestrator",
        )],
    }
