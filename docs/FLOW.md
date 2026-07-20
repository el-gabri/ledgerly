# How a Turn Flows Through the Orchestrator

Every user message triggers one pass through this graph. Conversation
continuity across turns comes from LangGraph checkpointing; escalation
counters (confidence streak, frustration, turn count) persist between turns.

## Turn flow

```mermaid
flowchart TD
    U([User message]) --> IN["Intake<br/>turn++, reset per-turn state"]
    IN -->|human owns the conversation| HH(["Human hold<br/>AI muted, message logged"])
    IN -->|otherwise| R{"Intent Router<br/>rules first, then classifier"}

    R -->|fraud claim or legal threat<br/>deterministic rule, pre-AI| E
    R -->|user asks for a human| E
    R -->|billing, how-to, complaint| V["Vendor AI agent<br/>sees redacted projection only"]
    R -->|product question| K["KB agent<br/>RAG over support docs"]
    R -->|account question| A["Account agent<br/>internal data, never shared with vendor"]
    R -->|greeting or unclear| C["Concierge<br/>hello / capability menu"]

    V -->|reply + confidence| G
    V -->|timeout or failure| F["Fallback<br/>mark vendor attempt failed"]
    F --> K
    K --> G{"Response Gate<br/>ordered escalation triggers"}
    A --> G
    C --> G

    G -->|no trigger fired| OUT([Reply delivered to user])
    G -->|trigger fired| E["Escalate<br/>build context package"]
    E --> H(["Human agent takes over<br/>summary, transcript, attempts, next actions"])
```

**Response Gate triggers, evaluated in order (first match wins):**

1. `vendor_exhausted` — vendor failed and the internal fallback is weak too
2. `low_confidence` — two consecutive replies below the confidence threshold
3. `user_frustration` — two frustration signals across the conversation
4. `turn_limit` — conversation exceeded the turn budget unresolved

(Restricted intents and explicit human requests escalate at the router,
before any AI agent runs — they never reach the gate.)

## Conversation state machine

```mermaid
stateDiagram-v2
    [*] --> INTAKE
    INTAKE --> ROUTING: classify user turn
    ROUTING --> AGENT_ACTIVE: dispatch to agent
    ROUTING --> ESCALATING: restricted intent / human request
    AGENT_ACTIVE --> GATING: draft reply produced
    AGENT_ACTIVE --> FALLBACK: vendor failure
    FALLBACK --> AGENT_ACTIVE: retry on KB agent
    GATING --> RESPONDED: gate passes
    GATING --> ESCALATING: trigger fired
    RESPONDED --> INTAKE: next user turn
    ESCALATING --> HUMAN_ACTIVE: context package delivered
    HUMAN_ACTIVE --> RESOLVED
    RESPONDED --> RESOLVED
    RESOLVED --> [*]
```

Every transition is appended to the event log with a reason — `/trace` in
the CLI replays it for the current conversation.
