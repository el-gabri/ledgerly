# Design Principles

The standards this codebase follows, and why. Written the way I'd write them
for a team: each rule exists to make a specific failure mode harder.

## 1. Policy must be auditable

Restricted-intent rules live in `policy.py`, dispatch in `router.py`, and
ordered response escalation in `gate.py`, with thresholds in `config.py`.
The supported fraud/legal patterns run before classification. A matched
restricted turn also skips inference during handoff summarization. Tests
guard the actual backend and responder calls, not just graph state names.

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

`state.py` defines the shared TypedDict and dataclass values. Sequential
responders update the shared draft and append durable attempt records;
intake resets scratch fields between turns. The vendor receives a filtered
projection that excludes account-agent replies and redacts supported PII
patterns. State changes must preserve reducers and checkpoint behavior.

## 5. Degrade, never crash a conversation

Vendor exceptions and structured failures enter the KB fallback. Optional
OpenAI initialization/call failures use offline behavior, and dense retrieval
initialization/search failures switch to TF-IDF. A weak vendor fallback
produces a warm human handoff. Broad exception handling stays at dependency
boundaries and records the recovery path; unexpected graph errors remain visible.

## 6. Every decision leaves a trace

Routing choices, gate evaluations, vendor invocations, responder outcomes,
and handoffs emit structured decision logs. Records supplied with graph state
include `conversation_id` and `turn`; dependency records can lack that context.
State transitions are checkpointed separately with reasons and timestamps;
`/trace` prints that conversation history. Vendor audit metadata reports
actual account-message omissions and matched redaction types.

## 7. Deterministic by default, stochastic by opt-in

Tests force offline generation, TF-IDF retrieval, and disabled tracing;
optional dependency failures are exercised with stubs. Demos default to
offline behavior. Decisions and reply text are deterministic; conversation
IDs and timestamps vary. The wheel smoke check also runs offline outside the
checkout to verify the installed resources.

## 8. Comments explain why, names explain what

Docstrings state the design intent of a module (often pointing at the design
doc section they implement). Inline comments are reserved for non-obvious
decisions — trigger ordering, why UNKNOWN routes to concierge, why the
streak counts the current draft.
