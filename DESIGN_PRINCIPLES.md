# Design Principles

The standards this codebase follows, and why. Written the way I'd write them
for a team: each rule exists to make a specific failure mode harder.

## 1. Policy must be auditable

Escalation triggers, restricted-intent rules, and confidence thresholds are
deterministic code in exactly two files (`router.py` rule layer, `gate.py`),
with every tunable in `config.py`. When compliance asks "can the AI ever
handle a fraud claim?", the answer is a regex you can point at, not a prompt
you can only hope about.

## 2. One checkpoint, not scattered checks

Every candidate reply — vendor, KB, or account — passes through the Response
Gate before reaching the user. Escalation policy lives in one node. The
alternative (each agent deciding for itself when to escalate) is how policy
drifts apart across a platform.

## 3. Interfaces where production will differ

Every simplification sits behind a seam shaped like the production component:

| Demo | Seam | Production swap |
|---|---|---|
| Mock vendor LLM | `VendorAdapter` | Bedrock / Vertex AI adapter |
| TF-IDF retrieval | index object with `.search()` | dense embeddings + FAISS/vector DB |
| Offline keyword backend | `LLMBackend` protocol | hosted LLM |
| `MemorySaver` | LangGraph checkpointer | Redis / Postgres checkpointer |
| JSON log lines | `log_event()` | OTel / LangSmith exporter |

The point of a prototype is to prove the seams, not to fake the scale.

## 4. The state schema is a contract, not a convenience

`state.py` is the one shared surface between all agents. Agents read shared
state but write only their own fields; the vendor gets a redacted projection,
never the raw state. Changing `state.py` is a design review, not a diff —
in a real team this file would have mandatory codeowner review.

## 5. Degrade, never crash a conversation

A support conversation must survive every dependency failure: vendor timeout
falls back to an internal agent; a broken optional mode (OpenAI key missing,
embeddings unavailable) falls back to offline behavior; and when nothing can
answer confidently, the failure mode is a *warm human handoff*, not an error
message. `except Exception` is allowed only at these degradation boundaries,
and always logs.

## 6. Every decision leaves a trace

Routing choices, gate evaluations, vendor invocations (including what was
redacted), and state transitions each emit one structured log line with
`conversation_id` and `turn`. The test for sufficiency: can you answer "why
did conversation X escalate?" from logs alone? (`/trace` in the CLI is this
question as a feature.)

## 7. Deterministic by default, stochastic by opt-in

Tests and demos run with zero network and zero randomness. LLM inference is
an opt-in enhancement behind a flag. This inverts the common failure mode of
LLM demos (flaky live calls, untestable behavior) and is the property that
makes the 32-test suite meaningful.

## 8. Comments explain why, names explain what

Docstrings state the design intent of a module (often pointing at the design
doc section they implement). Inline comments are reserved for non-obvious
decisions — trigger ordering, why UNKNOWN routes to the vendor, why the
streak counts the current draft.
