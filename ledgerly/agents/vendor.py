"""Vendor AI agent: adapter interface + mock implementation.

Two design points matter here:

1. **The adapter interface is the product.** `VendorAdapter` is shaped so a
   Bedrock or Vertex AI implementation is a drop-in: it receives a redacted
   projection of the conversation and returns a result-or-failure value. The
   mock is the only live implementation in this demo, but the seam is real.

2. **The projection is a security boundary.** The vendor never receives the
   full graph state — only the transcript and the current intent. Account
   data, retrieval internals, and orchestration metadata stay inside.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..llm import LLMBackend
from ..logging_utils import log_event
from ..state import (
    ConvState,
    DraftReply,
    Intent,
    OrchestratorState,
    VendorFailure,
    last_user_message,
    transition,
)


@dataclass
class VendorProjection:
    """The redacted view of the conversation a third-party vendor may see."""

    transcript: list[dict] = field(default_factory=list)  # [{role, content}]
    intent: str = "unknown"


@dataclass
class VendorResult:
    """Either a reply (ok=True) or a structured failure (ok=False)."""

    ok: bool
    content: str = ""
    confidence: float = 0.0
    failure: Optional[VendorFailure] = None


class VendorAdapter(ABC):
    """Contract every vendor integration must satisfy."""

    name: str = "vendor"

    @abstractmethod
    def invoke(self, projection: VendorProjection, chaos: Optional[str] = None) -> VendorResult:
        """Handle one turn. `chaos` is a test-only failure-injection knob."""


_TEMPLATES = {
    Intent.BILLING.value: (
        "Thanks for reaching out about billing. Charges usually settle within "
        "2 business days; duplicates are reversed automatically. If a charge "
        "still looks wrong after that, I can open a billing review for you.",
        0.85,
    ),
    Intent.HOW_TO.value: (
        "Here's how to do that in the Ledgerly app: open Settings, choose the "
        "relevant section, and follow the guided steps. I can walk you "
        "through it step by step if you tell me where you get stuck.",
        0.80,
    ),
    Intent.COMPLAINT.value: (
        "I'm sorry this has been a poor experience — that's not what we want. "
        "Let me look into what went wrong and make it right.",
        0.70,
    ),
}
_UNKNOWN_TEMPLATE = (
    "I'm not entirely sure I understood that correctly. Could you rephrase, "
    "or tell me a bit more about what you're trying to do?",
    0.40,  # honest hedge: below the gate's confidence threshold on purpose
)


class MockVendorLLM(VendorAdapter):
    """Simulates a third-party support LLM.

    Behavior is deterministic per intent; the `chaos` knob injects the two
    vendor failure modes the orchestrator must survive: hard failure
    (timeout) and soft failure (confidently mediocre output, reported here
    honestly as low confidence).
    """

    name = "mock_vendor_llm"

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    def invoke(self, projection: VendorProjection, chaos: Optional[str] = None) -> VendorResult:
        if chaos == "vendor_timeout":
            return VendorResult(
                ok=False,
                failure=VendorFailure("timeout", "vendor API did not respond in 10s (injected)"),
            )
        if chaos == "vendor_low_confidence":
            return VendorResult(
                ok=True,
                content="It might be related to your settings, possibly. Hard to say.",
                confidence=0.25,
            )

        template, confidence = _TEMPLATES.get(projection.intent, _UNKNOWN_TEMPLATE)
        user_text = projection.transcript[-1]["content"] if projection.transcript else ""
        content = self._backend.generate(
            system=(
                "You are a third-party customer-support assistant for Ledgerly, "
                "a digital wallet app. Be concise and helpful. You have no "
                "access to account data."
            ),
            prompt=user_text,
            fallback=template,
        )
        return VendorResult(ok=True, content=content, confidence=confidence)


def build_projection(state: OrchestratorState) -> VendorProjection:
    """Redact graph state down to what the vendor is allowed to see."""
    transcript = [
        {"role": m.role, "content": m.content}
        for m in state.get("messages", [])
        if m.role in ("user", "assistant")
    ]
    return VendorProjection(transcript=transcript, intent=state.get("current_intent", "unknown"))


def make_vendor_node(adapter: VendorAdapter):
    """Build the vendor graph node bound to a concrete adapter."""

    def vendor_node(state: OrchestratorState) -> dict:
        chaos = state.get("chaos")
        projection = build_projection(state)
        log_event("vendor_invoked", state, adapter=adapter.name,
                  redacted_fields=["account_data", "retrieval_context", "orchestration_metadata"],
                  chaos=chaos)

        result = adapter.invoke(projection, chaos=chaos)
        events = [transition(state.get("conv_state", "ROUTING"), ConvState.AGENT_ACTIVE,
                             f"dispatched to {adapter.name}")]

        if not result.ok:
            events.append(transition(ConvState.AGENT_ACTIVE.value, ConvState.FALLBACK,
                                     f"vendor failure: {result.failure.kind}"))
            log_event("vendor_failure", state, kind=result.failure.kind,
                      detail=result.failure.detail)
            return {
                "conv_state": ConvState.FALLBACK.value,
                "events": events,
                "vendor_failure": result.failure,
                "chaos": None,  # knob is consumed either way
            }

        events.append(transition(ConvState.AGENT_ACTIVE.value, ConvState.GATING,
                                 "vendor produced a draft reply"))
        log_event("vendor_reply", state, confidence=result.confidence)
        return {
            "conv_state": ConvState.GATING.value,
            "events": events,
            "draft": DraftReply(agent=adapter.name, content=result.content,
                                confidence=result.confidence),
            "chaos": None,
        }

    return vendor_node


def route_after_vendor(state: OrchestratorState) -> str:
    """Conditional edge: fall back to an internal agent on vendor failure."""
    return "fallback" if state.get("vendor_failure") else "gate"
