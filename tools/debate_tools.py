"""
tools/debate_tools.py — LangChain 1.0 custom tools for the debate agent.

These tools can be bound to any agent node that needs external data.
LangChain 1.0 uses @tool decorator with typed signatures for auto-schema generation.
"""

from __future__ import annotations
from langchain.tools import tool
from typing import Optional
import datetime


@tool
def search_argument_database(topic: str, stance: str = "both") -> str:
    """
    Search a curated database of famous arguments, philosophical positions,
    and historical debates on a given topic.

    Args:
        topic: The debate topic to search for
        stance: "pro", "con", or "both" to filter results

    Returns:
        A formatted string of relevant arguments and their sources
    """
    # In production: connect to a real vector store / RAG pipeline
    # e.g., using LangChain's vectorstore retrievers
    mock_results = {
        "pro": [
            "Kurzweil (2005): Exponential growth in computing will solve intractable problems",
            "Bostrom (2014): Superintelligent AI could optimize all human goals at once",
        ],
        "con": [
            "Russell (2019): Value alignment problem makes safe AGI extremely difficult",
            "Tegmark (2017): Economic disruption could destabilize society before benefits arrive",
        ]
    }

    results = []
    if stance in ("pro", "both"):
        results.extend([f"[PRO] {a}" for a in mock_results["pro"]])
    if stance in ("con", "both"):
        results.extend([f"[CON] {a}" for a in mock_results["con"]])

    return f"Arguments found for '{topic}':\n" + "\n".join(results)


@tool
def get_logical_fallacy_definition(fallacy_name: str) -> str:
    """
    Retrieve the formal definition and a clear example of a specific logical fallacy.

    Args:
        fallacy_name: Name of the fallacy (e.g., "Straw Man", "Ad Hominem")

    Returns:
        Definition, formal structure, and example of the fallacy
    """
    fallacy_db = {
        "ad hominem": {
            "definition": "Attacking the person making the argument rather than the argument itself.",
            "structure": "X is a bad person → X's argument must be wrong",
            "example": "You can't trust his climate data — he once drove an SUV."
        },
        "straw man": {
            "definition": "Misrepresenting someone's argument to make it easier to attack.",
            "structure": "Person A says X → Person B distorts X into Y → Person B attacks Y",
            "example": "She said we need gun regulations → He said she wants to ban all guns."
        },
        "false dichotomy": {
            "definition": "Presenting only two options when more exist.",
            "structure": "Either X or Y — since not X, therefore Y",
            "example": "Either we go to war or we show weakness."
        },
        "slippery slope": {
            "definition": "Claiming one event will inevitably lead to extreme consequences without justification.",
            "structure": "If X then Y, if Y then Z (extreme) → therefore not X",
            "example": "If we allow cannabis, everyone will be on heroin within a year."
        },
        "appeal to authority": {
            "definition": "Using the opinion of an authority figure as evidence in an argument.",
            "structure": "X is an authority → X says Y → therefore Y is true",
            "example": "Einstein believed in God, so God must exist."
        }
    }

    key = fallacy_name.lower()
    data = fallacy_db.get(key, {
        "definition": f"No formal entry found for '{fallacy_name}'.",
        "structure": "N/A",
        "example": "N/A"
    })

    return (
        f"**{fallacy_name}**\n"
        f"Definition: {data['definition']}\n"
        f"Logical structure: {data['structure']}\n"
        f"Example: {data['example']}"
    )


@tool
def save_debate_session(
    topic: str,
    final_score: float,
    summary: str,
    session_id: Optional[str] = None
) -> str:
    """
    Save the debate session results to persistent storage for long-term tracking.
    Enables the agent to track user improvement over time.

    Args:
        topic: The debate topic
        final_score: Average score across all rounds (0-50)
        summary: One-sentence summary of performance
        session_id: Optional session identifier for retrieval

    Returns:
        Confirmation with session ID and timestamp
    """
    # In production: write to a database (PostgreSQL, MongoDB, etc.)
    # or use LangGraph's built-in persistence layer
    timestamp = datetime.datetime.now().isoformat()
    sid = session_id or f"session_{hash(topic + timestamp) % 100000:05d}"

    return (
        f"✅ Session saved successfully!\n"
        f"Session ID: {sid}\n"
        f"Topic: {topic}\n"
        f"Score: {final_score}/50\n"
        f"Summary: {summary}\n"
        f"Timestamp: {timestamp}\n"
        f"Retrieve later with: agent.load_session('{sid}')"
    )


@tool
def get_debate_tips_by_score(score: float) -> str:
    """
    Get targeted improvement tips based on the user's debate score range.

    Args:
        score: The user's total score out of 50

    Returns:
        Personalized improvement suggestions
    """
    if score >= 40:
        level = "Advanced"
        tips = [
            "Focus on pre-emptive refutation — address counter-arguments before they're raised.",
            "Study Toulmin's model of argumentation to add warrant and backing to your claims.",
            "Practice steelmanning: represent the opposing view even stronger than they would.",
        ]
    elif score >= 25:
        level = "Intermediate"
        tips = [
            "Add more empirical evidence — cite specific studies, statistics, or events.",
            "Clearly state your core claim (thesis) in every round — don't let it drift.",
            "Practice the PEEL structure: Point, Evidence, Explanation, Link.",
        ]
    else:
        level = "Beginner"
        tips = [
            "Start with one clear, simple claim and defend it fully before adding more.",
            "Read about the 5 most common logical fallacies and practice spotting them.",
            "Watch TED debate clips and note how speakers structure their opening argument.",
        ]

    tip_str = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(tips))
    return f"Level: {level} ({score}/50)\nPersonalized tips:\n{tip_str}"
