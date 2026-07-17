"""LangGraph wiring: nodes, edges, and the per-turn driver.

One graph invocation == one user turn. Conversation continuity across turns
comes from LangGraph checkpointing (MemorySaver here; a Redis or Postgres
checkpointer is the production swap — same graph, different persistence).
"""
from __future__ import annotations

import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents.account import AccountAgent, make_account_node
from .agents.kb import KnowledgeBaseAgent, make_kb_node
from .agents.vendor import MockVendorLLM, VendorAdapter, make_vendor_node, route_after_vendor
from .gate import gate_node, route_after_gate
from .handoff import human_hold_node, make_escalate_node
from .llm import LLMBackend, get_backend
from .logging_utils import log_event
from .router import make_router_node, route_after_router
from .state import ConvState, Message, OrchestratorState, transition


def _intake_node(state: OrchestratorState) -> dict:
    """Start of every turn: bump the turn counter, reset per-turn scratch."""
    update = {
        "turn_count": state.get("turn_count", 0) + 1,
        "current_intent": None,
        "active_agent": None,
        "draft": None,
        "vendor_failure": None,
        "fallback_attempted": False,
        "pending_escalation": None,
        "gate_decision": None,
    }
    if not state.get("human_active"):
        update["conv_state"] = ConvState.INTAKE.value
        update["events"] = [transition(state.get("conv_state", "INTAKE"),
                                       ConvState.INTAKE, "user turn received")]
    log_event("turn_start", {**state, **update},
              human_active=bool(state.get("human_active")))
    return update


def _route_after_intake(state: OrchestratorState) -> str:
    """If a human owns the conversation, the AI pipeline stays out of it."""
    return "human_hold" if state.get("human_active") else "router"


def _fallback_node(state: OrchestratorState) -> dict:
    """Mark that the vendor failed and an internal agent is taking over."""
    log_event("fallback", state, from_agent="vendor", to_agent="kb")
    return {
        "fallback_attempted": True,
        "events": [transition(ConvState.FALLBACK.value, ConvState.FALLBACK,
                              "retrying with internal KB agent")],
    }


def _respond_node(state: OrchestratorState) -> dict:
    """Deliver the gated draft as the assistant's reply for this turn."""
    draft = state["draft"]
    log_event("reply_delivered", state, agent=draft.agent, confidence=draft.confidence)
    return {
        "conv_state": ConvState.RESPONDED.value,
        "events": [transition(ConvState.GATING.value, ConvState.RESPONDED,
                              f"reply delivered by {draft.agent}")],
        "messages": [Message(role="assistant", content=draft.content, agent=draft.agent)],
    }


def build_app(backend: LLMBackend | None = None,
              vendor_adapter: VendorAdapter | None = None,
              kb_agent: KnowledgeBaseAgent | None = None,
              account_agent: AccountAgent | None = None):
    """Assemble and compile the orchestration graph.

    Every dependency is injectable for tests; defaults build the standard
    demo configuration.
    """
    backend = backend or get_backend()
    vendor_adapter = vendor_adapter or MockVendorLLM(backend)
    kb_agent = kb_agent or KnowledgeBaseAgent(backend)
    account_agent = account_agent or AccountAgent()

    builder = StateGraph(OrchestratorState)
    builder.add_node("intake", _intake_node)
    builder.add_node("router", make_router_node(backend))
    builder.add_node("vendor", make_vendor_node(vendor_adapter))
    builder.add_node("kb", make_kb_node(kb_agent))
    builder.add_node("account", make_account_node(account_agent))
    builder.add_node("fallback", _fallback_node)
    builder.add_node("gate", gate_node)
    builder.add_node("respond", _respond_node)
    builder.add_node("escalate", make_escalate_node(backend))
    builder.add_node("human_hold", human_hold_node)

    builder.add_edge(START, "intake")
    builder.add_conditional_edges("intake", _route_after_intake,
                                  {"router": "router", "human_hold": "human_hold"})
    builder.add_conditional_edges("router", route_after_router,
                                  {"vendor": "vendor", "kb": "kb",
                                   "account": "account", "escalate": "escalate"})
    builder.add_conditional_edges("vendor", route_after_vendor,
                                  {"fallback": "fallback", "gate": "gate"})
    builder.add_edge("fallback", "kb")
    builder.add_edge("kb", "gate")
    builder.add_edge("account", "gate")
    builder.add_conditional_edges("gate", route_after_gate,
                                  {"respond": "respond", "escalate": "escalate"})
    builder.add_edge("respond", END)
    builder.add_edge("escalate", END)
    builder.add_edge("human_hold", END)

    return builder.compile(checkpointer=MemorySaver())


def run_turn(app, conversation_id: str, text: str, chaos: str | None = None) -> dict:
    """Execute one user turn against a (possibly ongoing) conversation."""
    payload: dict = {
        "conversation_id": conversation_id,
        "messages": [Message(role="user", content=text)],
    }
    if chaos:
        payload["chaos"] = chaos
    return app.invoke(payload, config={"configurable": {"thread_id": conversation_id}})


def new_conversation_id() -> str:
    return f"conv-{uuid.uuid4().hex[:8]}"
