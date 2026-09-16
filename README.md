# Ledgerly Support Orchestrator

A working prototype of a **unified orchestration layer** for customer-support
conversations, coordinating three kinds of responders behind one interface:

- a **vendor AI agent** (mock third-party support LLM behind a clean adapter),
- **internal specialized agents** (a RAG knowledge-base agent and an account agent),
- **human agents** (explicit escalation with a structured context package).

Built in Python on **LangGraph**. Personal demo project — Ledgerly is a
fictional digital-wallet app; no real company data or systems are involved.

## What it demonstrates

- **Intent routing per turn** — rule layer first (matched fraud claims and
  legal threats bypass inference, including summarization), then classification; the same conversation
  moves between agents as the user's need shifts.
- **Shared state and context passing** — one typed conversation state flows
  through every node; the vendor projection excludes internal account replies
  and filters supported email, CPF, card-number, and phone-number patterns.
  Pattern filtering is not complete anonymization.
- **An explicit conversation state machine** — every transition is recorded
  with a reason (`/trace` shows it live).
- **Graceful vendor degradation** — inject a vendor timeout mid-conversation
  and watch the internal KB agent take over; if the fallback is also weak,
  the conversation escalates.
- **Human handoff that starts warm** — escalations carry a context package:
  summary, transcript, agents attempted with confidence scores, the trigger
  that fired, and suggested next actions.
- **Observability** — routing, responder outcomes, gate decisions, and handoffs
  emit structured JSON logs. State transitions are kept separately in the
  checkpointed event history, available through `/trace`.
- **Useful clarification** — every numbered menu choice asks a category-specific
  follow-up; a valid selection does not count as another failed answer.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q          # deterministic, offline test suite
python scripts/check_wheel.py      # build/install outside checkout; no downloads

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
| `LANGCHAIN_TRACING_V2=true` (+ LangSmith key) | Optional LangSmith graph tracing |

OpenAI initialization/call failures use the offline backend or fallback text.
Optional embedding initialization/search failures switch to TF-IDF. Vendor
adapter exceptions and structured failures enter the internal KB fallback;
a weak fallback produces a human handoff.

The account fixture and KB articles are package resources, so a regular
`pip install .` also supports `python -m ledgerly.cli` outside the checkout.
The scripted scenarios and wheel verification script are repository tools.

## Project layout

```
ledgerly/
  state.py          # the context contract: typed shared state + state machine enums
  router.py         # rule layer + intent classification + dispatch
  policy.py         # shared deterministic fraud/legal/human-request rules
  gate.py           # Response Gate: single checkpoint, ordered escalation triggers
  handoff.py        # human handoff: context package + ownership transfer
  graph.py          # LangGraph wiring and per-turn driver
  llm.py            # LLM backend seam (offline default / OpenAI optional)
  config.py         # every tunable threshold in one place
  logging_utils.py  # structured JSON decision logs
  agents/
    vendor.py       # VendorAdapter interface + mock vendor LLM + redaction
    kb.py           # internal RAG over packaged support docs (TF-IDF by default)
    account.py      # internal account agent over mock fixtures
    concierge.py    # greetings, menu, and category clarification
  data/             # 15 fictional support docs + account fixture
scenarios/          # scripted demo conversations
tests/              # routing, state machine, gate, fallback, handoff
scripts/            # isolated wheel smoke check
```

See `DESIGN_DOC.md` for the architecture and state machine, and
`DESIGN_PRINCIPLES.md` for the coding standards the codebase follows.
