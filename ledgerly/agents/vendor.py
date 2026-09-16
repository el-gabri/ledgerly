"""Vendor AI agent: adapter interface + mock implementation.

Two design points matter here:

1. **The adapter interface is the product.** `VendorAdapter` is shaped so a
   Bedrock or Vertex AI implementation is a drop-in: it receives a redacted
   projection of the conversation and returns a result-or-failure value. The
   mock is the only live implementation in this demo, but the seam is real.

2. **The projection is a security boundary.** The vendor never receives the
   full graph state — only a filtered transcript, intent, and safe redaction
   metadata. Internal account replies are omitted before pattern filtering.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Optional

from ..llm import LLMBackend
from ..logging_utils import log_event
from ..state import (
    AgentAttempt,
    ConvState,
    DraftReply,
    Intent,
    OrchestratorState,
    VendorFailure,
    transition,
)


@dataclass
class VendorProjection:
    """The redacted view of the conversation a third-party vendor may see."""

    transcript: list[dict] = field(default_factory=list)  # [{role, content}]
    intent: str = "unknown"
    redaction_types: list[str] = field(default_factory=list)
    excluded_message_counts: dict[str, int] = field(default_factory=dict)


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


# This is a defensive outbound boundary, not an identity system. It removes
# common high-risk values a customer may paste into a message before that
# message leaves Ledgerly. New vendor integrations must use this projection
# rather than the graph state directly.
_CARD_NUMBER_RE = re.compile(r"(?<![\d+])(?:\d[ -]?){13,19}(?!\d)")
# Match a complete plus-prefixed international number with 8-15 digits.
# Apply before CPF/card patterns: an 11-digit international number can also
# look like a CPF when its leading plus is ignored.
_INTERNATIONAL_PHONE_RE = re.compile(r"(?<![\w+])\+[1-9]\d{7,14}(?!\w)")
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cpf", re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")),
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("phone", re.compile(
        r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{2,3}\)?[ .-]?)?\d{3,5}[ .-]\d{4}(?!\d)"
    )),
)


def _passes_luhn(digits: str) -> bool:
    """Avoid classifying phone numbers as card numbers in audit metadata."""
    total = 0
    for index, digit in enumerate(reversed(digits)):
        value = int(digit)
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Filter supported PII patterns; this is not complete anonymization."""
    redacted = text
    found: list[str] = []

    if _INTERNATIONAL_PHONE_RE.search(redacted):
        found.append("phone")
        redacted = _INTERNATIONAL_PHONE_RE.sub("[REDACTED_PHONE]", redacted)

    def redact_card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group())
        if _passes_luhn(digits):
            found.append("card_number")
            return "[REDACTED_CARD_NUMBER]"
        return match.group()

    redacted = _CARD_NUMBER_RE.sub(redact_card, redacted)
    for pii_type, pattern in _PII_PATTERNS:
        if pattern.search(redacted):
            if pii_type not in found:
                found.append(pii_type)
            redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)
    return redacted, found


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
    transcript: list[dict] = []
    redaction_types: set[str] = set()
    excluded_message_counts: dict[str, int] = {}
    for message in state.get("messages", []):
        if message.role not in ("user", "assistant"):
            continue
        if message.role == "assistant" and message.agent == "account":
            excluded_message_counts["account"] = excluded_message_counts.get("account", 0) + 1
            continue
        content, found = redact_pii(message.content)
        transcript.append({"role": message.role, "content": content})
        redaction_types.update(found)
    return VendorProjection(
        transcript=transcript,
        intent=state.get("current_intent", "unknown"),
        redaction_types=sorted(redaction_types),
        excluded_message_counts=excluded_message_counts,
    )


def make_vendor_node(adapter: VendorAdapter):
    """Build the vendor graph node bound to a concrete adapter."""

    def vendor_node(state: OrchestratorState) -> dict:
        chaos = state.get("chaos")
        projection = build_projection(state)
        log_event("vendor_invoked", state, adapter=adapter.name,
                  projection_fields=["transcript", "intent"],
                  excluded_message_counts=projection.excluded_message_counts,
                  pii_redaction_types=projection.redaction_types,
                  chaos=chaos)

        failure_source = "result"
        exception_type = None
        try:
            result = adapter.invoke(projection, chaos=chaos)
        except Exception as exc:  # dependency boundary: retain the KB recovery path
            failure_source = "exception"
            exception_type = type(exc).__name__
            result = VendorResult(ok=False, failure=VendorFailure(
                kind="timeout" if isinstance(exc, TimeoutError) else "error",
                detail="vendor adapter raised an exception",
            ))
        if not result.ok and result.failure is None:
            failure_source = "incomplete_result"
            result = VendorResult(ok=False, failure=VendorFailure(
                kind="error", detail="vendor adapter returned failure without details",
            ))
        events = [transition(state.get("conv_state", "ROUTING"), ConvState.AGENT_ACTIVE,
                             f"dispatched to {adapter.name}")]

        if not result.ok:
            events.append(transition(ConvState.AGENT_ACTIVE.value, ConvState.FALLBACK,
                                     f"vendor failure: {result.failure.kind}"))
            # Adapter error messages may include customer text or credentials.
            # Log safe classifications only, including the recovery destination.
            log_event("vendor_failure", state, adapter=adapter.name,
                      kind=result.failure.kind, failure_source=failure_source,
                      exception_type=exception_type, recovery="kb")
            return {
                "conv_state": ConvState.FALLBACK.value,
                "events": events,
                "vendor_failure": result.failure,
                "agent_attempts": [AgentAttempt(
                    agent=adapter.name,
                    outcome="failure",
                    failure_kind=result.failure.kind,
                )],
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
            "agent_attempts": [AgentAttempt(
                agent=adapter.name,
                outcome="reply",
                confidence=result.confidence,
            )],
            "chaos": None,
        }

    return vendor_node


def route_after_vendor(state: OrchestratorState) -> str:
    """Conditional edge: fall back to an internal agent on vendor failure."""
    return "fallback" if state.get("vendor_failure") else "gate"
