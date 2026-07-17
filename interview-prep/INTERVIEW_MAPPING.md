# Project → Job Description Mapping

One line per JD requirement: where it lives in the project and the one sentence to say about it. Keep this for prep only — in conversation, let the mapping surface naturally.

| JD requirement | Where it lives | What to say |
|---|---|---|
| **Unified orchestration layer coordinating vendor AI, internal multi-agent systems, and human agents** | The whole graph (`graph.py`): mock vendor behind `VendorAdapter`, two internal agents, explicit human-handoff path | "I built the smallest system that has all three responder types behind one orchestration surface — the coordination is the product, not any single agent." |
| **Intent routing** | `router.py`: deterministic rule layer + classifier, re-run every turn | "Routing is layered: rules for policy-critical intents that must never reach an AI, a model for everything else. Rules beat model — auditability is the point." |
| **State transitions** | `state.py` `ConvState` enum + event log; asserted in `test_state_machine.py` | "The conversation is an explicit 9-state machine, every transition logged with a reason. `/trace` replays it — that's how you debug 'why did this escalate?'" |
| **Context sharing across agents** | `OrchestratorState` (the context contract) + `VendorProjection` redaction in `agents/vendor.py` | "One typed state flows through everything; agents write only their own fields, and the vendor sees a redacted projection — never account data. Context sharing and data boundaries are the same problem." |
| **Human agent handoff** | `handoff.py`: triggers + structured context package | "Escalation is a first-class outcome, not a failure branch: the human gets a summary, transcript, agents attempted with confidences, and suggested actions. Metric: the customer never repeats themselves." |
| **Vendor AI/LLM frameworks (Bedrock, Vertex)** | `VendorAdapter` interface; mock is the only live impl | "I mocked the vendor deliberately — the interface is shaped so a Bedrock or Vertex adapter is a drop-in, and the mock let me test failure modes no real vendor lets you inject on demand." |
| **LangGraph** | Whole orchestration graph, incl. checkpointer-based multi-turn state | "LangGraph's node/edge/checkpoint model maps cleanly to conversation orchestration; swapping MemorySaver for a Postgres checkpointer is the production persistence story." |
| **LangSmith** | Structured JSON decision logs (`logging_utils.py`); LangSmith enableable via env | "I shipped the observability content — every routing and gate decision as structured logs — dependency-free; LangSmith is a config flag on top since it's plain LangGraph." |
| **Production-grade Python service** | Tests (32, deterministic), typed contracts, injectable dependencies, graceful degradation everywhere | "Deterministic-by-default is the design choice I'm proudest of: the LLM is opt-in, so the test suite actually means something." |
| **Technical design docs for cross-functional stakeholders** | `DESIGN_DOC.md`, written *before* the code | "I wrote the design doc first and built to it — the doc's state machine table is literally asserted by the tests." |
| **Coding/system-design standards, mentoring** | `DESIGN_PRINCIPLES.md` | "I wrote the standards down the way I'd onboard an engineer: each rule tied to the failure mode it prevents." |
| *(Your background bridge)* | KB agent (`agents/kb.py`) | "The KB agent is the same retrieve-then-judge pattern as Similis at iFood — BGE-M3 + FAISS across 300 subcategories there, TF-IDF at demo scale here; the seam is where embeddings swap in." |

**Framing sentence for the whole project** (use once, early): "I don't have Coinbase's domain, so I built the problem from your JD in a fictional domain — the orchestration, routing, and handoff mechanics are what I wanted to show, and everything mocked is mocked behind an interface shaped like the real thing."
