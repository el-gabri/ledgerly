"""Internal account agent: answers account-specific questions from mock data.

This agent exists to prove a boundary: internal agents hold tools and data
the vendor AI must never see. The account fixtures never enter the vendor
projection (see agents/vendor.py).
"""
from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path

from ..logging_utils import log_event
from ..state import (
    AgentAttempt,
    ConvState,
    DraftReply,
    OrchestratorState,
    last_user_message,
    transition,
)


class AccountAgent:
    """Keyword-driven lookups over mock account fixtures. In production this
    would be a tool-calling agent over real account APIs with authn — the
    orchestration seam is identical."""

    name = "account"

    def __init__(self, fixtures_path: Path | None = None) -> None:
        source = (fixtures_path if fixtures_path is not None else
                  files("ledgerly").joinpath("data").joinpath("accounts.json"))
        self._account = json.loads(source.read_text(encoding="utf-8"))

    def answer(self, query: str) -> DraftReply:
        lowered = query.lower()
        acct = self._account

        if "balance" in lowered or "wallet" in lowered:
            content = f"Your current balance is {acct['balance']} {acct['currency']}."
        elif "deposit" in lowered:
            return self._answer_deposit(lowered)
        elif "transaction" in lowered or "payment" in lowered or "transfer" in lowered:
            lines = [
                f"- {t['date']}: {t['description']} — {t['amount']} {acct['currency']} ({t['status']})"
                for t in acct["recent_transactions"]
            ]
            content = "Here are your recent transactions:\n" + "\n".join(lines)
        elif "card" in lowered:
            content = (f"Your card ending in {acct['card']['last4']} is "
                       f"{acct['card']['status']}.")
        elif "statement" in lowered:
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

    def _answer_deposit(self, query: str) -> DraftReply:
        deposits = [
            transaction for transaction in self._account["recent_transactions"]
            if "deposit" in transaction["description"].lower()
        ]
        dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query))
        descriptions = {
            transaction["description"].lower() for transaction in deposits
            if transaction["description"].lower() in query
        }
        matches = [
            transaction for transaction in deposits
            if (dates or descriptions)
            and (not dates or dates == {transaction["date"]})
            and (not descriptions or transaction["description"].lower() in descriptions)
        ]
        if len(matches) != 1:
            return DraftReply(
                agent=self.name,
                content=("I can't identify which recorded deposit you mean. "
                         "Please provide its date (YYYY-MM-DD) or description."),
                confidence=0.40,
            )
        deposit = matches[0]
        return DraftReply(
            agent=self.name,
            content=(f"The recorded {deposit['description']} dated {deposit['date']} "
                     f"for {deposit['amount']} {self._account['currency']} is "
                     f"{deposit['status']}."),
            confidence=0.90,
        )


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
        return {
            "conv_state": ConvState.GATING.value,
            "events": events,
            "draft": draft,
            "agent_attempts": [AgentAttempt(
                agent=draft.agent,
                outcome="reply",
                confidence=draft.confidence,
                citations=draft.citations,
            )],
        }

    return account_node
