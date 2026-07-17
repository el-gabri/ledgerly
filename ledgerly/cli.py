"""Demo driver: interactive chat or scripted scenarios.

Usage:
    python -m ledgerly.cli                                   # interactive
    python -m ledgerly.cli --scenario scenarios/vendor_failure.json
    python -m ledgerly.cli --scenario ... --show-package     # print handoff package

Interactive commands:
    /chaos timeout     inject a vendor timeout on the next turn
    /chaos lowconf     inject a low-confidence vendor reply on the next turn
    /trace             print the state-machine event log so far
    /quit              exit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .graph import build_app, new_conversation_id, run_turn

_CHAOS_ALIASES = {"timeout": "vendor_timeout", "lowconf": "vendor_low_confidence"}


def _print_reply(state: dict) -> None:
    msg = state["messages"][-1]
    agent = f" [{msg.agent}]" if msg.agent else ""
    print(f"\nassistant{agent}: {msg.content}")
    print(f"  (state={state['conv_state']}, intent={state.get('current_intent')}, "
          f"turn={state['turn_count']})")


def _print_trace(state: dict) -> None:
    print("\n--- state-machine trace ---")
    for ev in state.get("events", []):
        print(f"  {ev.from_state:>12} -> {ev.to_state:<12} {ev.reason}")
    print("---------------------------")


def _print_package(state: dict) -> None:
    esc = state.get("escalation")
    if not esc:
        return
    print("\n=== HUMAN HANDOFF CONTEXT PACKAGE ===")
    print(json.dumps(esc.package, indent=2, ensure_ascii=False, default=str))
    print("=====================================")


def run_scenario(path: Path, show_package: bool = False) -> dict:
    """Run a scripted conversation from a JSON scenario file."""
    scenario = json.loads(path.read_text(encoding="utf-8"))
    app = build_app()
    cid = new_conversation_id()
    print(f"# scenario: {scenario['name']} ({cid})")
    state: dict = {}
    for turn in scenario["turns"]:
        print(f"\nuser: {turn['text']}" + (f"   [chaos: {turn['chaos']}]" if turn.get("chaos") else ""))
        state = run_turn(app, cid, turn["text"], chaos=turn.get("chaos"))
        _print_reply(state)
    _print_trace(state)
    if show_package:
        _print_package(state)
    return state


def run_interactive() -> None:
    app = build_app()
    cid = new_conversation_id()
    chaos: str | None = None
    state: dict = {}
    print(f"Ledgerly support orchestrator ({cid}). Type /quit to exit.")
    while True:
        try:
            text = input("\nuser: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/trace":
            _print_trace(state)
            continue
        if text.startswith("/chaos"):
            arg = text.split(maxsplit=1)[1] if " " in text else ""
            chaos = _CHAOS_ALIASES.get(arg)
            print(f"  chaos armed: {chaos}" if chaos else "  usage: /chaos timeout|lowconf")
            continue
        state = run_turn(app, cid, text, chaos=chaos)
        chaos = None
        _print_reply(state)
        _print_package(state)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ledgerly orchestrator demo driver")
    parser.add_argument("--scenario", type=Path, help="path to a scenario JSON file")
    parser.add_argument("--show-package", action="store_true",
                        help="print the handoff context package if an escalation occurred")
    args = parser.parse_args(argv)
    if args.scenario:
        run_scenario(args.scenario, show_package=args.show_package)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
