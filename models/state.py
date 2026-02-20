"""
models/state.py — Typed state for the entire debate session.
Uses LangGraph 1.0's TypedDict-based state with Annotated reducers.
"""

from __future__ import annotations
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


def _append(existing: list, new: list) -> list:
    """Simple list-append reducer for LangGraph state."""
    return existing + new


class DebateSession(TypedDict):
    # Core setup
    topic: str
    user_position: str
    round_number: int

    # Conversation history (uses LangGraph's built-in add_messages reducer)
    dialogue_history: Annotated[list[BaseMessage], add_messages]

    # Agent outputs accumulated across rounds
    logical_fallacies_found: Annotated[list[dict], _append]
    argument_scores: Annotated[list[dict], _append]
    devil_advocate_args: Annotated[list[str], _append]
    socratic_questions: Annotated[list[str], _append]
    coaching_tips: Annotated[list[str], _append]

    # Final outputs (set once)
    verdict: str
