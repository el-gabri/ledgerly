"""Deterministic, auditable routing policy shared by every classifier.

Policy-critical intents must be recognized before an LLM is involved.  Keeping
these rules in one module prevents the router and the offline backend from
drifting apart, while still allowing the latter to be used independently in
tests or tools.
"""
from __future__ import annotations

import re
import unicodedata

from .state import Intent


def _normalize(text: str) -> str:
    """Case-fold text and remove accents so policy works across supported locales."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


# The order is deliberate: a fraud or legal claim must not be softened into a
# generic request for support. Patterns use word boundaries rather than raw
# substrings, so punctuation such as "I will sue." is handled correctly.
_POLICY_RULES: tuple[tuple[Intent, tuple[re.Pattern[str], ...]], ...] = (
    (
        Intent.FRAUD_CLAIM,
        (
            re.compile(r"\b(?:unauthorized|unrecognized|stolen|fraud|hacked)\b"),
            re.compile(
                r"\b(?:do not|don't|dont|did not|didn't|didnt)\s+"
                r"(?:recognize|recognise|authorize|authorise|make)\b"
            ),
            re.compile(r"\b(?:not my charge|someone charged|never (?:authorized|made))\b"),
            re.compile(
                r"\b(?:nao (?:reconheco|autorizei|fui eu|fiz)|"
                r"cartao roubado|fraude|invadiram minha conta)\b"
            ),
        ),
    ),
    (
        Intent.LEGAL_THREAT,
        (
            re.compile(r"\b(?:lawyer|attorney|lawsuit|legal action|sue|suing)\b"),
            re.compile(r"\b(?:advogado|processo judicial|acao legal|processar)\b"),
        ),
    ),
    (
        Intent.HUMAN_REQUEST,
        (
            re.compile(r"\b(?:human|real (?:person|agent)|representative)\b"),
            re.compile(
                r"\b(?:speak|talk) to (?:someone|a (?:person|human|representative))\b"
            ),
            re.compile(r"\b(?:atendente|pessoa real|suporte humano)\b"),
            re.compile(r"\bfalar com (?:alguem|uma pessoa)\b"),
        ),
    ),
)


def apply_policy_rules(text: str) -> Intent | None:
    """Return the highest-priority policy intent, if the text triggers one."""
    normalized = _normalize(text)
    for intent, patterns in _POLICY_RULES:
        if any(pattern.search(normalized) for pattern in patterns):
            return intent
    return None
