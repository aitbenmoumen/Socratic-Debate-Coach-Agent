"""
graph/debate_graph.py — LangGraph 1.0 graph definition.

Graph architecture:
                        ┌─────────────────┐
                        │   intake_node   │  ← entry point
                        └────────┬────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    fallacy_detector     │
                    └────────────┬────────────┘
                                 │ (parallel fan-out)
              ┌──────────────────┼──────────────────┐
              │                  │                  │
    ┌─────────▼────────┐ ┌───────▼───────┐ ┌───────▼──────────┐
    │  devil_advocate  │ │   socratic_   │ │    argument_     │
    │                  │ │  questioner   │ │     scorer       │
    └─────────┬────────┘ └───────┬───────┘ └───────┬──────────┘
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │ (fan-in)
                    ┌────────────▼────────────┐
                    │    should_continue?     │  ← conditional edge
                    │   (rounds 1..MAX_ROUNDS) │
                    └──────┬──────────┬───────┘
                      more │          │ done
                    ┌──────▼──────┐  ┌▼──────────────┐
                    │ increment_  │  │  final_coach  │
                    │   round     │  │               │
                    └──────┬──────┘  └───────────────┘
                           │ (loops back up)
                           └──────────────────────────►

Key LangGraph 1.0 features used:
  - StateGraph with TypedDict state + Annotated reducers
  - Parallel node execution via Send API / fan-out edges
  - Conditional edges for loop control
  - MemorySaver checkpointer for durable state across interruptions
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from models.state import DebateSession
from agents.nodes import (
    intake_node,
    fallacy_detector_node,
    devil_advocate_node,
    socratic_questioner_node,
    argument_scorer_node,
    increment_round_node,
    final_coach_node,
)

MAX_ROUNDS = 3  # Number of debate rounds before final coaching


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge function — decides whether to loop or end
# ─────────────────────────────────────────────────────────────────────────────
def should_continue(state: DebateSession) -> str:
    """
    After each analysis round, decide:
    - If we've completed MAX_ROUNDS → go to final coach
    - Otherwise → increment round and loop
    """
    if state["round_number"] >= MAX_ROUNDS:
        return "finalize"
    return "continue"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_debate_graph() -> StateGraph:
    """
    Builds and compiles the full LangGraph 1.0 debate graph.

    Returns a compiled graph ready for .invoke() or .astream().
    The MemorySaver checkpointer enables:
      - Resumable sessions after interruption
      - Human-in-the-loop pause points (extendable)
      - State inspection at any point in execution
    """
    builder = StateGraph(DebateSession)

    # ── Add all nodes ──────────────────────────────────────────────────────
    builder.add_node("intake", intake_node)
    builder.add_node("fallacy_detector", fallacy_detector_node)
    builder.add_node("devil_advocate", devil_advocate_node)
    builder.add_node("socratic_questioner", socratic_questioner_node)
    builder.add_node("argument_scorer", argument_scorer_node)
    builder.add_node("increment_round", increment_round_node)
    builder.add_node("final_coach", final_coach_node)

    # ── Entry edge ─────────────────────────────────────────────────────────
    builder.add_edge(START, "intake")

    # ── After intake: run fallacy detector first ───────────────────────────
    builder.add_edge("intake", "fallacy_detector")

    # ── After fallacy detection: fan out to 3 parallel agents ─────────────
    # LangGraph 1.0 supports parallel execution by adding multiple edges
    # from the same source node — they execute concurrently
    builder.add_edge("fallacy_detector", "devil_advocate")
    builder.add_edge("fallacy_detector", "socratic_questioner")
    builder.add_edge("fallacy_detector", "argument_scorer")

    # ── All 3 parallel agents fan back into the conditional checkpoint ─────
    # LangGraph automatically synchronizes when all upstream nodes complete
    builder.add_conditional_edges(
        "devil_advocate",          # trigger node (last to finish triggers check)
        should_continue,
        {
            "continue": "increment_round",
            "finalize": "final_coach",
        }
    )
    builder.add_conditional_edges(
        "socratic_questioner",
        should_continue,
        {
            "continue": "increment_round",
            "finalize": "final_coach",
        }
    )
    builder.add_conditional_edges(
        "argument_scorer",
        should_continue,
        {
            "continue": "increment_round",
            "finalize": "final_coach",
        }
    )

    # ── Loop back: after incrementing round, re-run analysis ───────────────
    builder.add_edge("increment_round", "fallacy_detector")

    # ── Terminal edge ──────────────────────────────────────────────────────
    builder.add_edge("final_coach", END)

    # ── Compile with MemorySaver for durable execution ─────────────────────
    # LangGraph 1.0: MemorySaver provides built-in persistence so the graph
    # can be resumed after crashes, user pauses, or async interruptions.
    checkpointer = MemorySaver()
    compiled = builder.compile(checkpointer=checkpointer)

    return compiled
