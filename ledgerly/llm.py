"""LLM backend abstraction: deterministic offline mode + optional OpenAI mode.

The orchestrator never calls a model directly; it calls this interface.
Offline mode is the default so the demo (and the test suite) runs with zero
network dependency and fully deterministic behavior. OpenAI mode swaps in
real inference behind the same interface — proving the seam, not just
claiming it.
"""
from __future__ import annotations

import re
from typing import Protocol

from .config import llm_mode, openai_model
from .logging_utils import log_event
from .state import Intent


class LLMBackend(Protocol):
    """What the orchestrator needs from a language model — nothing more."""

    def classify_intent(self, text: str) -> Intent: ...

    def generate(self, system: str, prompt: str, fallback: str) -> str:
        """Generate text; MUST return `fallback` if generation is unavailable."""
        ...


# ---------------------------------------------------------------------------
# Offline backend: keyword rules, deterministic, zero dependencies.
# ---------------------------------------------------------------------------

# Ordered list: first matching intent wins. Restricted intents come first so
# no later pattern can shadow them.
_INTENT_PATTERNS: list[tuple[Intent, list[str]]] = [
    (Intent.FRAUD_CLAIM, [
        "unauthorized", "didn't make", "did not make", "didnt make", "stolen",
        "fraud", "hacked", "someone charged", "not my charge", "never made",
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
    # Task-shaped phrases that would otherwise be shadowed by the possessive
    # ACCOUNT patterns below ("close MY ACCOUNT" is a how-to, not a lookup).
    (Intent.HOW_TO, [
        "close my account", "delete my account", "reset my password",
    ]),
    (Intent.ACCOUNT, [
        "my balance", "balance", "my account", "my transaction", "my card",
        "my transfer", "my payment", "my statement", "my deposit",
    ]),
    (Intent.BILLING, [
        "fee", "charged twice", "charge", "refund", "invoice", "billing",
        "overcharg", "double charg", "dispute",
    ]),
    (Intent.HOW_TO, [
        "how do i", "how to", "how can i", "where do i", "reset my password",
        "set up", "close my account", "delete my account",
    ]),
    (Intent.PRODUCT, [
        "what is", "what are", "does ledgerly", "limit", "policy", "supported",
        "which countries", "currency", "exchange rate", "is there",
    ]),
    (Intent.COMPLAINT, [
        "not working", "doesn't work", "terrible", "awful", "worst", "useless",
        "ridiculous", "fed up", "frustrated", "angry",
    ]),
]

_FRUSTRATION_MARKERS = [
    "frustrated", "frustrating", "angry", "furious", "ridiculous", "useless",
    "terrible", "awful", "worst", "fed up", "waste of time", "not working",
    "still not", "unacceptable",
]


def detect_frustration(text: str) -> bool:
    """Sentiment-lite: does this message carry a frustration signal?"""
    lowered = text.lower()
    return any(marker in lowered for marker in _FRUSTRATION_MARKERS)


class OfflineBackend:
    """Deterministic keyword-based backend. Primitive by design: routing,
    state transitions, and escalation are the system under demonstration,
    not response eloquence."""

    name = "offline"

    def classify_intent(self, text: str) -> Intent:
        lowered = text.lower()
        for intent, patterns in _INTENT_PATTERNS:
            if any(p in lowered for p in patterns):
                return intent
        return Intent.UNKNOWN

    def generate(self, system: str, prompt: str, fallback: str) -> str:
        return fallback


# ---------------------------------------------------------------------------
# OpenAI backend: real inference behind the same interface.
# ---------------------------------------------------------------------------

_VALID_LABELS = {i.value for i in Intent}


class OpenAIBackend:
    """Backend using the OpenAI API. Any failure degrades to offline behavior
    rather than crashing the conversation — the same posture the orchestrator
    takes toward the vendor agent."""

    name = "openai"

    def __init__(self) -> None:
        from openai import OpenAI  # lazy import: optional dependency

        self._client = OpenAI()
        self._offline = OfflineBackend()

    def classify_intent(self, text: str) -> Intent:
        labels = ", ".join(sorted(_VALID_LABELS))
        try:
            resp = self._client.chat.completions.create(
                model=openai_model(),
                temperature=0,
                max_tokens=10,
                messages=[
                    {"role": "system", "content": (
                        "Classify the customer-support message into exactly one "
                        f"label from: {labels}. Reply with the label only."
                    )},
                    {"role": "user", "content": text},
                ],
            )
            label = re.sub(r"[^a-z_]", "", resp.choices[0].message.content.strip().lower())
            if label in _VALID_LABELS:
                return Intent(label)
            log_event("llm_classify_invalid_label", label=label)
        except Exception as exc:  # noqa: BLE001 — degrade, never crash a turn
            log_event("llm_classify_error", error=str(exc))
        return self._offline.classify_intent(text)

    def generate(self, system: str, prompt: str, fallback: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=openai_model(),
                temperature=0.3,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            log_event("llm_generate_error", error=str(exc))
            return fallback


def get_backend() -> LLMBackend:
    """Select the backend from LEDGERLY_LLM_MODE (offline is the default)."""
    if llm_mode() == "openai":
        try:
            return OpenAIBackend()
        except Exception as exc:  # missing package or key — degrade loudly
            log_event("llm_backend_fallback", error=str(exc), fallback="offline")
    return OfflineBackend()
