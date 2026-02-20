"""
utils/visualization.py — Helpers for displaying debate progress and results.
"""

from __future__ import annotations
from typing import Any


def print_score_card(scores: list[dict]) -> None:
    """Render a visual score progression table across rounds."""
    if not scores:
        print("No scores recorded.")
        return

    headers = ["Round", "Clarity", "Evidence", "Logic", "Originality", "Persuasion", "Total"]
    col_w = 12

    print("\n" + "═" * (col_w * len(headers)))
    print("  📊 SCORE PROGRESSION")
    print("═" * (col_w * len(headers)))
    print("".join(h.ljust(col_w) for h in headers))
    print("─" * (col_w * len(headers)))

    for s in scores:
        row = [
            str(s.get("round", "?")),
            str(s.get("clarity", "–")),
            str(s.get("evidence", "–")),
            str(s.get("logic", "–")),
            str(s.get("originality", "–")),
            str(s.get("persuasiveness", "–")),
            f"{s.get('total', 0)}/50",
        ]
        print("".join(v.ljust(col_w) for v in row))

    print("═" * (col_w * len(headers)) + "\n")


def print_fallacy_summary(fallacies: list[dict]) -> None:
    """Print a formatted fallacy detection summary."""
    if not fallacies:
        print("\n✅ No logical fallacies were detected across all rounds.\n")
        return

    print(f"\n⚠️  LOGICAL FALLACIES DETECTED ({len(fallacies)} total):")
    print("─" * 60)
    for f in fallacies:
        severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
            f.get("severity", "medium"), "⚪"
        )
        print(f"{severity_icon} {f.get('fallacy_name', 'Unknown')} [{f.get('severity', '?')}]")
        print(f"   Quote: \"{f.get('quote', '')}\"")
        print(f"   Why: {f.get('explanation', '')}")
        print()


def print_debate_banner(topic: str, max_rounds: int) -> None:
    """Print a styled opening banner."""
    width = 64
    print("\n" + "╔" + "═" * (width - 2) + "╗")
    print("║" + " 🎭 SOCRATIC DEBATE COACH AGENT ".center(width - 2) + "║")
    print("║" + " Powered by LangChain 1.0 + LangGraph 1.0 ".center(width - 2) + "║")
    print("╠" + "═" * (width - 2) + "╣")
    print("║" + f" Topic: {topic[:width-10]}".ljust(width - 2) + "║")
    print("║" + f" Rounds: {max_rounds} | Agents: 5 specialized nodes".ljust(width - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝\n")


def format_session_summary(state: Any) -> str:
    """Generate a brief text summary of a completed session."""
    scores = state.get("argument_scores", [])
    avg_score = sum(s.get("total", 0) for s in scores) / max(len(scores), 1)
    fallacy_count = len(state.get("logical_fallacies_found", []))
    question_count = len(state.get("socratic_questions", []))

    return (
        f"Session Summary:\n"
        f"  Topic: {state.get('topic', 'N/A')}\n"
        f"  Rounds completed: {state.get('round_number', 0)}\n"
        f"  Average score: {avg_score:.1f}/50\n"
        f"  Fallacies caught: {fallacy_count}\n"
        f"  Socratic questions asked: {question_count}\n"
    )
