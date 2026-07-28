"""Structured JSON logging for orchestration decisions.

Every routing decision, agent invocation, gate evaluation, and state
transition emits one JSON line keyed by conversation_id/turn. This is the
same information a LangSmith trace would carry, without the external
dependency; LangSmith can be enabled on top via the standard
LANGCHAIN_TRACING_V2 env vars since the graph is plain LangGraph.
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
