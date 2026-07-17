# Live Demo Script (5–10 minutes)

**Setup before the call:** terminal open in the repo, venv active, tests already run once (warm), font size up. Have `DESIGN_DOC.md` open in a tab. Offline mode (default) — zero network risk.

---

## 0. Frame it (30s, no screen share yet)

> "I wanted to walk in with something concrete, so I built a small orchestration layer against your job description: one LangGraph service coordinating a vendor AI, internal agents, and human handoff. Fictional domain — a payments app called Ledgerly — because the mechanics are what I wanted to show, not domain knowledge I don't have yet. Can I show you four scenarios, about 90 seconds each?"

## 1. Happy path — routing + context sharing (2 min)

```bash
python -m ledgerly.cli --scenario scenarios/happy_path.json
```

Point at, in order:
- Three questions, three different agents answer: vendor → internal KB (with citation) → internal account agent. **"Routing happens every turn — same conversation, three responders, one shared state."**
- The state-machine trace at the end: INTAKE → ROUTING → AGENT_ACTIVE → GATING → RESPONDED, per turn. **"Every transition has a reason attached. This is the artifact you replay when support ops asks why a conversation went the way it did."**
- One sentence on the gate: **"Every reply — no matter which agent — passes one Response Gate. Escalation policy lives in exactly one place."**

## 2. Kill the vendor — graceful degradation (2 min)

```bash
python -m ledgerly.cli --scenario scenarios/vendor_failure.json
```

- Turn 2 injects a vendor timeout. **"Watch: vendor dies mid-conversation, the trace shows AGENT_ACTIVE → FALLBACK, and the internal KB agent answers the same question from our own docs. The user never sees the failure."**
- **"The chaos knob exists because the mock vendor let me test failure modes a real vendor never lets you inject on demand. And if the fallback is *also* weak, it escalates to a human instead — degrade, never dead-end."**

## 3. Fraud claim — the rule layer (1.5 min)

```bash
python -m ledgerly.cli --scenario scenarios/fraud_immediate.json --show-package
```

- **"One turn: 'a charge I didn't make.' Look at the trace — ROUTING → ESCALATING. No AI agent ever ran. That's a deterministic rule, not a prompt: for policy-critical intents, auditability beats intelligence."**
- Scroll the context package briefly: summary, transcript, trigger, suggested actions for the human.

## 4. Frustrated user — escalation + warm handoff (2 min)

```bash
python -m ledgerly.cli --scenario scenarios/frustration_escalation.json --show-package
```

- Narrate honestly: **"Turn 1, the offline classifier actually misreads the need — routes to the account agent, which shows transactions. Realistic failure. Turn 2 the user is frustrated; the gate has counted two frustration signals and hands off."**
- Land on the package: **"This is the part I care most about: the human starts with the summary, everything the AI attempted with confidence scores, and next actions. The metric is that the customer never repeats themselves."**

## 5. Close with the engineering (1–2 min)

```bash
python -m pytest tests -q
```

- **"32 tests, all deterministic — the LLM is opt-in behind a flag, which is why the state machine and routing logic are actually testable. Design doc was written before the code; the state table in it is asserted by these tests. And every simplification — mock vendor, TF-IDF retrieval, in-memory checkpointing — sits behind an interface shaped like the production component: Bedrock adapter, embeddings + vector store, Redis/Postgres checkpointer."**

Then stop talking and let them ask questions — the follow-up doc covers the likely ones.

---

**Fallback plan:** if anything at all misbehaves live, run `python -m pytest tests -q` first (it always works), and walk scenarios from the trace output in the README instead. If asked "is this a real LLM?": "In this mode, no — deliberately. Flip `LEDGERLY_LLM_MODE=openai` and the same graph runs on real inference; the demo runs offline so the orchestration is what you're watching, not API latency."
