# Likely Follow-up Questions — and Honest Answers

The pattern for every answer: name what the prototype simplifies, then show you know exactly what the production version looks like. Never defend the prototype as production-ready; it isn't, and pretending otherwise is the failure mode.

---

## 1. "How does this scale to millions of concurrent users?"

**Honest starting point:** "The prototype is single-process with in-memory state — that's a demo choice, and the design keeps the swap-points explicit."

The production shape: the graph itself is stateless per turn — all conversation state lives in the checkpointer. So you scale horizontally: N workers behind a queue, conversation-affinity by hashing `conversation_id` (or no affinity at all if the checkpointer is fast, since any worker can rehydrate any conversation). MemorySaver becomes Redis for hot state with Postgres for durable history. The real bottlenecks won't be the orchestrator — they'll be LLM inference latency and vendor rate limits, so the production design needs per-vendor connection pools, request hedging, and queue backpressure that degrades to "you're in line for a human" rather than dropping conversations. The event log becomes a Kafka topic, which also feeds analytics and model retraining.

**Bridge to background:** "I've worked at that transaction scale at Mercado Libre — the discipline of 'every request must land somewhere sane under overload' is the same."

## 2. "What about security? You're passing conversation context to a third-party vendor."

**Honest starting point:** "The prototype has no authn/authz, and redaction is structural rather than content-aware — but the boundary itself is built."

The vendor never receives the graph state; it receives a `VendorProjection` — transcript and intent only, never account data, and the redaction is logged per invocation. In production I'd extend that in three ways: PII scrubbing *inside* the transcript itself (the user may type their card number at the vendor), since structural redaction doesn't catch content; per-vendor data-processing contracts encoded as code — each adapter declares what fields it's allowed to receive and a policy test enforces it; and audit logging of every projection sent, for compliance review. Same posture on the inbound side: vendor output is untrusted input — it passes the Response Gate, and in production a moderation/safety check, before reaching a user.

## 3. "What are the vendor failure modes and how do you handle them?"

The taxonomy I built for: **hard failures** (timeout, 5xx, rate limit) — detectable, handled by fallback to internal agents, with retry budgets and circuit breakers in production so a degraded vendor doesn't add latency to every conversation; **soft failures** (confidently wrong or hedging answers) — the harder class; the prototype uses the vendor's self-reported confidence, which in production I wouldn't trust — I'd add an independent judge model scoring vendor replies, which is exactly the LLM-as-judge diagnostic layer I built for Similis at iFood; and **drift** (vendor silently updates their model and quality shifts) — needs continuous eval: replay a golden conversation set against the vendor weekly and alert on score drops.

**The one-liner:** "Treat the vendor like any other unreliable upstream dependency, plus one extra failure mode traditional services don't have: being wrong fluently."

## 4. "Your intent classifier is keyword-based. That won't survive real users."

**Concede immediately — this is a trap if you defend it.** "Correct, and it misroutes in my own demo — turn 1 of the frustration scenario reads 'payment not working' as a transaction lookup. The keyword layer exists so the demo is deterministic and the tests are meaningful."

Production: an LLM or fine-tuned classifier does the semantic work — but the *architecture* keeps the deterministic rule layer above it, because restricted intents (fraud, legal) must be a matter of auditable rules, not model behavior. The router is also the component you improve continuously: every escalation whose transcript shows a misroute is a labeled training example, so the handoff packages become the classifier's training data flywheel. Confidence thresholds and the intent→agent map are config, not code, so tuning routing doesn't require a deploy.

## 5. "Why did you mock the vendor instead of calling a real API? / Is this just a toy?"

"Three deliberate reasons. First, failure injection: a real vendor never lets you trigger a timeout on demand, and the degradation path was the most important thing to demonstrate. Second, determinism: 32 tests that always pass beat a flashier demo that fails when the WiFi does. Third, honesty about scope: one week, solo, alongside a full-time job — I spent the budget on the orchestration layer because that's what this role owns. The `VendorAdapter` interface is shaped from real Bedrock/Vertex invocation patterns, so the mock is a stand-in, not a dead end — and `LEDGERLY_LLM_MODE=openai` already runs the same graph on real inference."

## 6. "How would you measure whether this orchestration layer is actually good?"

Three layers of metrics. **Customer outcomes:** resolution rate without escalation, time-to-resolution, CSAT split by which agent(s) handled the conversation. **Orchestration quality:** routing accuracy (audited sample + misroute rate inferred from escalation transcripts), escalation precision (of conversations escalated, how many did the human confirm needed a human?) and recall (of conversations that ended badly, how many should have escalated earlier?), and handoff quality — did the customer have to repeat themselves after transfer? **System health:** per-agent latency and failure rates, fallback activation rate, vendor drift scores from the golden-set replays. The event log the prototype already emits is deliberately the data source for all of these.

## 7. (Bonus) "What would you build next if you joined?"

"First, replace self-reported confidence with an independent judge — it's the weakest trust assumption in the design. Second, the golden-conversation eval harness, because you can't manage vendors or tune routing without regression detection. Third, the human side of the handoff: the return path — human resolves, hands *back* to the AI for follow-up — which my state machine sketches (HUMAN_ACTIVE → RESOLVED) but doesn't implement, and which is where a lot of the real product value hides."
