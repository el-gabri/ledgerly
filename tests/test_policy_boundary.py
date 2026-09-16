"""Restricted routing must bypass every inference and responder boundary."""
from __future__ import annotations

import pytest

from ledgerly.graph import build_app, run_turn
from ledgerly.llm import OfflineBackend
from ledgerly.policy import apply_policy_rules
from ledgerly.state import Intent


class GuardedBackend(OfflineBackend):
    def __init__(self):
        self.blocked = False
        self.calls = []

    def classify_intent(self, text):
        self.calls.append("classify")
        assert not self.blocked, "restricted turn reached classifier"
        return super().classify_intent(text)

    def generate(self, system, prompt, fallback):
        self.calls.append("generate")
        assert not self.blocked, "restricted turn reached generator"
        return fallback


class ForbiddenResponder:
    name = "forbidden"

    def invoke(self, *args, **kwargs):
        pytest.fail("restricted turn reached vendor")

    def answer(self, *args, **kwargs):
        pytest.fail("restricted turn reached internal responder")


_RESTRICTED_CASES = [
    (f"I don{apostrophe}t recognize this charge", Intent.FRAUD_CLAIM)
    for apostrophe in ("'", "\u2018", "\u2019", "\u02bc")
] + [
    (f"There is a charge on my card I didn{apostrophe}t make", Intent.FRAUD_CLAIM)
    for apostrophe in ("'", "\u2019")
] + [
    ("Não reconheço esta cobrança", Intent.FRAUD_CLAIM),
    ("My card was stolen", Intent.FRAUD_CLAIM),
    ("I will sue.", Intent.LEGAL_THREAT),
    ("I will contact my lawyer", Intent.LEGAL_THREAT),
]


@pytest.mark.parametrize("text, intent", _RESTRICTED_CASES)
@pytest.mark.parametrize("after_menu", [False, True])
def test_restricted_turn_never_invokes_ai(text, intent, after_menu):
    backend = GuardedBackend()
    responder = ForbiddenResponder()
    app = build_app(backend=backend, vendor_adapter=responder,
                    kb_agent=responder, account_agent=responder)
    cid = "restricted-boundary"
    if after_menu:
        prior = run_turn(app, cid, "Unclear request")
        assert prior["awaiting_menu_selection"]
    backend.calls.clear()
    backend.blocked = True

    assert apply_policy_rules(text) is intent
    state = run_turn(app, cid, text)

    assert backend.calls == []
    assert state["current_intent"] == intent.value
    assert state["conv_state"] == "HUMAN_ACTIVE"
    assert state["escalation"].trigger == "restricted_intent"
    package = state["escalation"].package
    assert package["transcript"][-1]["content"] == text
    assert package["intents_seen"][-1] == intent.value
    assert "Escalated because" in package["summary"]
    assert package["suggested_actions"]
    assert [attempt["agent"] for attempt in package["agents_attempted"]] == (
        ["concierge"] if after_menu else []
    )
