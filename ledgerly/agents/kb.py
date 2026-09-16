"""Internal knowledge-base agent: RAG over Ledgerly's support docs.

Retrieval is a tiny hand-rolled TF-IDF + cosine index (~50 lines, zero
dependencies) so the demo runs anywhere. The retrieval function is a seam:
set LEDGERLY_EMBEDDINGS=st to swap in sentence-transformers dense
embeddings — the same retrieve-then-generate pattern as production
embedding+FAISS systems, at demo scale.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from importlib.resources import files
from pathlib import Path

from ..config import KB_TOP_K, KB_WEAK_SCORE, embeddings_mode
from ..llm import LLMBackend
from ..logging_utils import log_event
from ..state import (
    AgentAttempt,
    ConvState,
    DraftReply,
    OrchestratorState,
    last_user_message,
    transition,
)

_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it my of on or "
    "the to what when where which with you your".split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in _STOPWORDS]


class TfIdfIndex:
    """Minimal TF-IDF / cosine index over a small document corpus."""

    def __init__(self, docs: dict[str, str]) -> None:
        self._docs = docs
        self._doc_vecs: dict[str, dict[str, float]] = {}
        df: Counter = Counter()
        tokenized = {doc_id: _tokenize(text) for doc_id, text in docs.items()}
        for tokens in tokenized.values():
            df.update(set(tokens))
        n_docs = max(len(docs), 1)
        self._idf = {t: math.log(n_docs / (1 + c)) + 1.0 for t, c in df.items()}
        for doc_id, tokens in tokenized.items():
            self._doc_vecs[doc_id] = self._vectorize(tokens)

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {t: (c / len(tokens)) * self._idf.get(t, 0.0) for t, c in tf.items()} if tokens else {}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        qvec = self._vectorize(_tokenize(query))
        scores = [
            (doc_id, sum(qvec.get(t, 0.0) * w for t, w in dvec.items()))
            for doc_id, dvec in self._doc_vecs.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class KnowledgeBaseAgent:
    """Retrieve-then-answer over the support-doc corpus, with honest
    confidence reporting: a weak retrieval score is surfaced as low
    confidence so the Response Gate can act on it."""

    name = "kb"

    def __init__(self, backend: LLMBackend, docs_dir: Path | None = None) -> None:
        self._backend = backend
        source = (docs_dir if docs_dir is not None else
                  files("ledgerly").joinpath("data").joinpath("kb_docs"))
        self._docs = {
            p.name[:-3]: p.read_text(encoding="utf-8")
            for p in sorted(source.iterdir(), key=lambda p: p.name)
            if p.is_file() and p.name.endswith(".md")
        }
        # Keep the dependency-free index ready for failures during query encoding
        # as well as during optional model initialization.
        self._tfidf_index = TfIdfIndex(self._docs)
        self._index = self._tfidf_index
        if embeddings_mode() == "st":
            try:
                self._index = _SentenceTransformerIndex(self._docs)
            except Exception as exc:  # noqa: BLE001
                self._log_embeddings_fallback(exc, stage="initialization")

    @staticmethod
    def _log_embeddings_fallback(exc: Exception, stage: str) -> None:
        # Dependency exception messages can echo the query; log safe metadata only.
        log_event("kb_embeddings_fallback", dependency="sentence_transformers",
                  stage=stage, error_type=type(exc).__name__, fallback="tfidf")

    def answer(self, query: str) -> DraftReply:
        try:
            hits = self._index.search(query, KB_TOP_K)
        except Exception as exc:  # noqa: BLE001 — only the optional dependency degrades
            if self._index is self._tfidf_index:
                raise
            self._log_embeddings_fallback(exc, stage="search")
            self._index = self._tfidf_index
            hits = self._index.search(query, KB_TOP_K)
        top_id, top_score = hits[0] if hits else ("", 0.0)

        if not hits or top_score < KB_WEAK_SCORE:
            return DraftReply(
                agent=self.name,
                content=("I couldn't find a support article that covers this. "
                         "Could you rephrase the question?"),
                confidence=0.30,
                citations=[],
            )

        doc = self._docs[top_id]
        # These are short articles: retain all supporting body text, including
        # later paragraphs that may directly answer the customer's question.
        fallback = "\n".join(
            line for line in doc.splitlines() if not re.match(r"^\s{0,3}#{1,6}\s", line)
        ).strip()

        content = self._backend.generate(
            system=("You are Ledgerly's internal support assistant. Answer ONLY "
                    "from the provided article; if it doesn't cover the "
                    "question, say so. Cite the article ID in your answer."),
            prompt=f"Question: {query}\n\nArticle ID: {top_id}\n\nArticle:\n{doc}",
            fallback=f"{fallback}\n\n(Source: {top_id})",
        )
        confidence = min(0.9, 0.55 + top_score)
        # The generator receives only ``top_id``. Do not claim citations for
        # retrieval candidates it has never seen or used in its answer.
        return DraftReply(agent=self.name, content=content, confidence=confidence,
                          citations=[top_id])


class _SentenceTransformerIndex:
    """Optional dense-embedding index (requires sentence-transformers)."""

    def __init__(self, docs: dict[str, str]) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._ids = list(docs)
        self._embs = self._model.encode([docs[i] for i in self._ids], normalize_embeddings=True)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self._embs @ q
        ranked = sorted(zip(self._ids, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:k]


def make_kb_node(agent: KnowledgeBaseAgent):
    """Build the KB graph node. Reachable directly (product questions) or as
    the fallback target after a vendor failure."""

    def kb_node(state: OrchestratorState) -> dict:
        query = last_user_message(state)
        draft = agent.answer(query)
        current = state.get("conv_state", "ROUTING")
        events = [
            transition(current, ConvState.AGENT_ACTIVE, "dispatched to internal KB agent"),
            transition(ConvState.AGENT_ACTIVE.value, ConvState.GATING,
                       "KB agent produced a draft reply"),
        ]
        log_event("kb_reply", state, confidence=draft.confidence,
                  citations=draft.citations, as_fallback=state.get("fallback_attempted", False))
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

    return kb_node
