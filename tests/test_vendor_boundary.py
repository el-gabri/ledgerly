"""Adapter-input isolation and failure handling at the actual vendor seam."""
from __future__ import annotations

import json

import pytest

from ledgerly.agents import vendor as vendor_module
from ledgerly.agents.vendor import VendorResult, redact_pii
from ledgerly.graph import build_app, run_turn
from ledgerly.llm import OfflineBackend


class CapturingVendor:
    name = "capture_vendor"

    def __init__(self):
        self.projections = []

    def invoke(self, projection, chaos=None):
        self.projections.append(projection)
        return VendorResult(ok=True, content="Here is a permitted support reply.", confidence=0.8)


@pytest.mark.parametrize("account_question, sensitive", [
    ("What's my balance?", ["1,284.50"]),
    ("Show my transactions", ["Grocery Mart", "Salary deposit", "62.10", "Sam R."]),
    ("What's my card status?", ["4821", "is active"]),
])
def test_vendor_never_receives_prior_account_replies(monkeypatch, account_question, sensitive):
    records = []
    monkeypatch.setattr(vendor_module, "log_event", lambda event, state=None, **kw: records.append({"event": event, **kw}))
    vendor = CapturingVendor()
    app = build_app(backend=OfflineBackend(), vendor_adapter=vendor)
    run_turn(app, "projection", "Hi")
    prior = run_turn(app, "projection", account_question)
    original = prior["messages"][-1].content
    state = run_turn(app, "projection", "How do I reset my password?")
    projection = vendor.projections[-1]
    outgoing = json.dumps(projection.transcript)
    assert original not in outgoing
    assert all(value not in outgoing for value in sensitive)
    assert any(message["content"] == "Hi" for message in projection.transcript)
    assert any(message["role"] == "assistant" for message in projection.transcript)
    assert projection.transcript[-1]["content"] == "How do I reset my password?"
    assert projection.excluded_message_counts == {"account": 1}
    assert any(message.content == original for message in state["messages"])
    invocation = next(record for record in records if record["event"] == "vendor_invoked")
    assert invocation["excluded_message_counts"] == {"account": 1}
    assert "redacted_fields" not in invocation


@pytest.mark.parametrize("phone", ["+5511998761234", "+14155552671", "+55 11 99876-1234", "(415) 555-2671"])
def test_phone_redaction_at_adapter_input(phone):
    vendor = CapturingVendor()
    app = build_app(backend=OfflineBackend(), vendor_adapter=vendor)
    run_turn(app, "phone", f"How do I reset my password? My phone is {phone}")
    projection = vendor.projections[-1]
    assert phone not in projection.transcript[-1]["content"]
    assert "[REDACTED_PHONE]" in projection.transcript[-1]["content"]
    assert projection.redaction_types == ["phone"]


@pytest.mark.parametrize("text", ["Balance 1,284.50 USD", "A fee of +12.50", "Reference 1234", "Limit +2500"])
def test_redaction_keeps_ordinary_amounts_and_short_numbers(text):
    assert redact_pii(text) == (text, [])


@pytest.mark.parametrize("failure, expected_kind", [("timeout", "timeout"), ("runtime", "error"), ("missing", "error")])
@pytest.mark.parametrize("covered", [True, False])
def test_adapter_failures_recover_and_record_attempts(monkeypatch, failure, expected_kind, covered):
    records = []
    monkeypatch.setattr(vendor_module, "log_event", lambda event, state=None, **kw: records.append({"event": event, **kw}))

    class FailingVendor:
        name = "failing_vendor"

        def invoke(self, projection, chaos=None):
            if failure == "missing":
                return VendorResult(ok=False)
            exception = TimeoutError if failure == "timeout" else RuntimeError
            raise exception("private customer text should not appear in logs")

    app = build_app(backend=OfflineBackend(), vendor_adapter=FailingVendor())
    query = "How do I close my account?" if covered else "How do I do the thing with the stuff?"
    state = run_turn(app, "failure", query)
    assert state["fallback_attempted"] is True
    assert state["vendor_failure"].kind == expected_kind
    assert state["agent_attempts"][0].outcome == "failure"
    assert state["agent_attempts"][0].failure_kind == expected_kind
    assert state["agent_attempts"][1].agent == "kb"
    if covered:
        assert state["conv_state"] == "RESPONDED"
        assert state["messages"][-1].agent == "kb"
    else:
        assert state["escalation"].trigger == "vendor_exhausted"
        assert state["escalation"].package["agents_attempted"][0]["failure_kind"] == expected_kind
    failure_log = next(record for record in records if record["event"] == "vendor_failure")
    assert failure_log["recovery"] == "kb"
    assert failure_log["kind"] == expected_kind
    assert "private customer text" not in json.dumps(records)
