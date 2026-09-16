"""Structured JSON logging for orchestration decisions.

Routing decisions, responder outcomes, gate evaluations, and handoffs emit
JSON records. Calls with graph state include conversation_id/turn; backend
and index initialization failures may not have conversation context.
State transitions are separate checkpointed records exposed by /trace.
Optional LangSmith tracing can be enabled through its environment settings.
"""
from __future__ import annotations

import json
import logging
import sys

_logger = logging.getLogger("ledgerly")
if not _logger.handlers:  # avoid duplicate handlers under pytest re-imports
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def log_event(event: str, state: dict | None = None, **fields) -> None:
    """Emit one structured log line.

    `state` (the graph state) is used only to stamp conversation_id and turn;
    callers pass decision-specific fields explicitly so the log line contains
    exactly what a reviewer needs to answer "why did the system do that?".
    """
    record = {"event": event}
    if state is not None:
        record["conversation_id"] = state.get("conversation_id", "?")
        record["turn"] = state.get("turn_count", 0)
    record.update(fields)
    _logger.info(json.dumps(record, ensure_ascii=False, default=str))
