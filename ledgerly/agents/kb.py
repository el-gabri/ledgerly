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
from pathlib import Path

from ..config import KB_TOP_K, KB_WEAK_SCORE, embeddings_mode
from ..llm import LLMBackend
from ..logging_utils import log_event
from ..state import ConvState, DraftReply, OrchestratorState, last_user_message, transition

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "kb_docs"

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

    def __init__(self, backend: LLMBackend, docs_dir: Path = _DATA_DIR) -> None:
        self._backend = backend
        self._docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(docs_dir.glob("*.md"))}
        if embeddings_mode() == "st":
            try:  # optional dense-embedding path; falls back silently to TF-IDF
                self._index = _SentenceTransformerIndex(self._docs)
            except Exception as exc:  # noqa: BLE001
                log_event("kb_embeddings_fallback", error=str(exc))
                self._index = TfIdfIndex(self._docs)
        else:
            self._index = TfIdfIndex(self._docs)

    def answer(self, query: str) -> DraftReply:
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
        # Offline fallback: first content paragraph of the best-matching doc.
        paragraphs = [p.strip() for p in doc.split("\n\n") if p.strip() and not p.startswith("#")]
        fallback = paragraphs[0] if paragraphs else doc.strip()

        content = self._backend.generate(
            system=("You are Ledgerly's internal support assistant. Answer ONLY "
                    "from the provided articles; if they don't cover the "
                    "question, say so. Cite the article you used."),
            prompt=f"Question: {query}\n\nArticles:\n{doc}",
            fallback=f"{fallback}\n\n(Source: {top_id})",
        )
        confidence = min(0.9, 0.55 + top_score)
        return DraftReply(agent=self.name, content=content, confidence=confidence,
                          citations=[doc_id for doc_id, _ in hits])


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
        return {"conv_state": ConvState.GATING.value, "events": events, "draft": draft}

    return kb_node
