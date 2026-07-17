"""Central configuration: thresholds and mode flags.

Every tunable lives here so reviewers can see the orchestrator's policy
surface in one place. In production these would come from a config service
with per-environment overrides; env vars are the demo-scale equivalent.
"""
from __future__ import annotations

import os

# --- Escalation policy -----------------------------------------------------
#: A draft reply below this confidence counts toward the low-confidence streak.
LOW_CONFIDENCE_THRESHOLD = 0.5
#: Consecutive low-confidence replies before escalating to a human.
LOW_CONFIDENCE_STREAK_LIMIT = 2
#: Frustrated user messages tolerated before escalating (1st gets an apology).
FRUSTRATION_LIMIT = 2
#: Maximum user turns before the conversation escalates as unresolved.
TURN_LIMIT = 8

# --- KB retrieval ----------------------------------------------------------
#: Number of documents retrieved per query.
KB_TOP_K = 3
#: Cosine score below which a KB answer is considered weak.
KB_WEAK_SCORE = 0.08


def llm_mode() -> str:
    """'offline' (deterministic, default) or 'openai' (real LLM calls)."""
    return os.environ.get("LEDGERLY_LLM_MODE", "offline").lower()


def openai_model() -> str:
    return os.environ.get("LEDGERLY_OPENAI_MODEL", "gpt-4o-mini")


def embeddings_mode() -> str:
    """'tfidf' (default, zero-dependency) or 'st' (sentence-transformers)."""
    return os.environ.get("LEDGERLY_EMBEDDINGS", "tfidf").lower()
