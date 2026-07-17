# Ledgerly Support Orchestrator

A working prototype of a **unified orchestration layer** for customer-support
conversations, coordinating three kinds of responders behind one interface:

- a **vendor AI agent** (mock third-party support LLM behind a clean adapter),
- **internal specialized agents** (a RAG knowledge-base agent and an account agent),
- **human agents** (explicit escalation with a structured context package).

Built in Python on **LangGraph**. Personal demo project — Ledgerly is a
fictional digital-wallet app; no real company data or systems are involved.

## What it demonstrates

- **Intent routing per turn** — rule layer first (restricted topics like fraud
  claims bypass all AI), then intent classification; the same conversation
  moves between agents as the user's need shifts.
- **Shared state and context passing** — one typed conversation state flows
  through every node; the vendor receives only a redacted projection (never
  account data).
- **An explicit conversation state machine** — every transition is recorded
  with a reason (`/trace` shows it live).
- **Graceful vendor degradation** — inject a vendor timeout mid-conversation
  and watch the internal KB agent take over; if the fallback is also weak,
  the conversation escalates.
- **Human handoff that starts warm** — escalations carry a context package:
  summary, transcript, agents attempted with confidence scores, the trigger
  that fired, and suggested next actions.
- **Observability** — every routing decision, agent invocation, gate
  evaluation, and state transition emits a structured JSON log line.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q          # 32 tests, all offline/deterministic

# Scripted demos
python -m ledgerly.cli --scenario scenarios/happy_path.json
python -m ledgerly.cli --scenario scenarios/vendor_failure.json
python -m ledgerly.cli --scenario scenarios/frustration_escalation.json --show-package
python -m ledgerly.cli --scenario scenarios/fraud_immediate.json --show-package

# Interactive
python -m ledgerly.cli
#   /chaos timeout   -> vendor timeout on next turn
#   /chaos lowconf   -> weak vendor reply on next turn
#   /trace           -> print the state-machine event log
```

## Modes

Everything runs **offline and deterministic by default** — no API keys, no
network. Optional upgrades, each behind the same interface:

| Env var | Effect |
|---|---|
| `LEDGERLY_LLM_MODE=openai` (+ `OPENAI_API_KEY`) | Real LLM for classification and generation (`pip install openai`) |
| `LEDGERLY_EMBEDDINGS=st` | Dense-embedding retrieval via sentence-transformers instead of TF-IDF |
| `LANGCHAIN_TRACING_V2=true` (+ LangSmith key) | LangSmith tracing — the graph is plain LangGraph, so this just works |

Any failure in an optional mode degrades to offline behavior instead of
crashing a conversation — the same posture the orchestrator takes toward the
vendor agent.

## Project layout

```
ledgerly/
  state.py          # the context contract: typed shared state + state machine enums
  router.py         # rule layer + intent classification + dispatch
  gate.py           # Response Gate: single checkpoint, ordered escalation triggers
  handoff.py        # human handoff: context package + ownership transfer
  graph.py          # LangGraph wiring and per-turn driver
  llm.py            # LLM backend seam (offline default / OpenAI optional)
  config.py         # every tunable threshold in one place
  logging_utils.py  # structured JSON decision logs
  agents/
    vendor.py       # VendorAdapter interface + mock vendor LLM + redaction
    kb.py           # internal RAG agent over data/kb_docs (TF-IDF, ~50 LOC index)
    account.py      # internal account agent over mock fixtures
data/               # 15 fictional support docs + account fixtures
scenarios/          # scripted demo conversations
tests/              # routing, state machine, gate, fallback, handoff
```

See `DESIGN_DOC.md` for the architecture and state machine, and
`DESIGN_PRINCIPLES.md` for the coding standards the codebase follows.
