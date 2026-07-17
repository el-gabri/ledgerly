"""Internal account agent: answers account-specific questions from mock data.

This agent exists to prove a boundary: internal agents hold tools and data
the vendor AI must never see. The account fixtures never enter the vendor
projection (see agents/vendor.py).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..logging_utils import log_event
from ..state import ConvState, DraftReply, OrchestratorState, last_user_message, transition

_FIXTURES = Path(__file__).resolve().parent.parent.parent / "data" / "accounts.json"


class AccountAgent:
    """Keyword-driven lookups over mock account fixtures. In production this
    would be a tool-calling agent over real account APIs with authn — the
    orchestration seam is identical."""

    name = "account"

    def __init__(self, fixtures_path: Path = _FIXTURES) -> None:
        self._account = json.loads(fixtures_path.read_text(encoding="utf-8"))

    def answer(self, query: str) -> DraftReply:
        lowered = query.lower()
        acct = self._account

        if "balance" in lowered:
            content = f"Your current balance is {acct['balance']} {acct['currency']}."
        elif "transaction" in lowered or "payment" in lowered or "transfer" in lowered:
            lines = [
                f"- {t['date']}: {t['description']} — {t['amount']} {acct['currency']} ({t['status']})"
                for t in acct["recent_transactions"]
            ]
            content = "Here are your recent transactions:\n" + "\n".join(lines)
        elif "card" in lowered:
            content = (f"Your card ending in {acct['card']['last4']} is "
                       f"{acct['card']['status']}.")
        elif "statement" in lowered or "deposit" in lowered:
            content = ("Your monthly statements are available under Settings > "
                       "Documents; the latest one covers last month.")
        else:
            return DraftReply(
                agent=self.name,
                content=("I can check your balance, transactions, or card status "
                         "— which would you like?"),
                confidence=0.40,
            )
        return DraftReply(agent=self.name, content=content, confidence=0.90)


def make_account_node(agent: AccountAgent):
    """Build the account graph node."""

    def account_node(state: OrchestratorState) -> dict:
        draft = agent.answer(last_user_message(state))
        events = [
            transition(state.get("conv_state", "ROUTING"), ConvState.AGENT_ACTIVE,
                       "dispatched to internal account agent"),
            transition(ConvState.AGENT_ACTIVE.value, ConvState.GATING,
                       "account agent produced a draft reply"),
        ]
        log_event("account_reply", state, confidence=draft.confidence)
        return {"conv_state": ConvState.GATING.value, "events": events, "draft": draft}

    return account_node
