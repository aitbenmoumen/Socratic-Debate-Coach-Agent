"""
agents/nodes.py — Each LangGraph node is a focused, single-responsibility agent.

Architecture uses LangChain 1.0's create_agent() + LangGraph 1.0 node pattern:
  - Each node receives the full DebateSession state
  - Each node returns a dict of ONLY the fields it updates
  - LangGraph merges updates via Annotated reducers in models/state.py
"""

from __future__ import annotations
import json
import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from models.state import DebateSession
from prompts.templates import (
    FALLACY_DETECTOR_SYSTEM,
    FALLACY_DETECTOR_HUMAN,
    DEVIL_ADVOCATE_SYSTEM,
    DEVIL_ADVOCATE_HUMAN,
    SOCRATIC_QUESTIONER_SYSTEM,
    SOCRATIC_QUESTIONER_HUMAN,
    ARGUMENT_SCORER_SYSTEM,
    ARGUMENT_SCORER_HUMAN,
    FINAL_COACH_SYSTEM,
    FINAL_COACH_HUMAN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared LLM (LangChain 1.0 provider-agnostic init)
# ─────────────────────────────────────────────────────────────────────────────
def _get_llm(temperature: float = 0.3):
    """
    LangChain 1.0 provider-agnostic model init.
    Set LANGCHAIN_MODEL env var to switch providers, e.g.:
      - "openai:gpt-4o"
      - "anthropic:claude-3-5-sonnet-20241022"
      - "google_vertexai:gemini-2.0-flash"
    Defaults to OpenAI GPT-4o.

    If your OpenAI key requires a custom endpoint, set OPENAI_BASE_URL in .env.
    """
    import os
    from langchain_openai import ChatOpenAI

    model_string = os.getenv("LANGCHAIN_MODEL", "openai:gpt-4o")
    provider, model_name = model_string.split(":", 1)

    
    if provider == "openai":
        base_url = os.getenv("OPENAI_BASE_URL")      
        api_key  = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=base_url, 
        )

    
    return init_chat_model(model=model_name, model_provider=provider, temperature=temperature)


def _extract_json(text: str) -> Any:
    """Robustly extract JSON from LLM output (handles markdown fences)."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()
    return json.loads(text)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: Intake — parses the user's initial argument for the first round
# ─────────────────────────────────────────────────────────────────────────────
async def intake_node(state: DebateSession) -> dict:
    """
    Sets up round 1 by logging the user's initial position as the
    first message in dialogue_history.
    """
    print("🎙️  [Round 1] User position registered. Starting debate analysis...\n")

    return {
        "round_number": 1,
        "dialogue_history": [
            HumanMessage(content=state["user_position"])
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: Fallacy Detector — finds logical fallacies in latest user argument
# ─────────────────────────────────────────────────────────────────────────────
async def fallacy_detector_node(state: DebateSession) -> dict:
    """
    Analyzes the most recent user message for logical fallacies.
    Uses a structured JSON output to ensure consistent, parseable results.
    """
    llm = _get_llm(temperature=0.1)

    # Get the latest human message as the current argument
    latest_arg = next(
        (m.content for m in reversed(state["dialogue_history"])
         if isinstance(m, HumanMessage)),
        state["user_position"]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", FALLACY_DETECTOR_SYSTEM),
        ("human", FALLACY_DETECTOR_HUMAN),
    ])

    chain = prompt | llm
    response = await chain.ainvoke({
        "topic": state["topic"],
        "argument": latest_arg,
    })

    try:
        fallacies = _extract_json(response.content)
    except (json.JSONDecodeError, ValueError):
        fallacies = []

    if fallacies:
        print(f"🔍 [Fallacy Detector] Found {len(fallacies)} fallacy(ies):")
        for f in fallacies:
            print(f"   ⚠️  {f.get('fallacy_name', 'Unknown')} — {f.get('explanation', '')[:80]}...")
    else:
        print("✅ [Fallacy Detector] No logical fallacies detected in this argument.")

    return {
        "logical_fallacies_found": fallacies,
        "dialogue_history": [
            AIMessage(content=f"[Fallacy Analysis] {json.dumps(fallacies, indent=2)}")
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3: Devil's Advocate — generates strong counter-arguments
# ─────────────────────────────────────────────────────────────────────────────
async def devil_advocate_node(state: DebateSession) -> dict:
    """
    Constructs the most compelling counter-arguments using 3 different
    intellectual angles (empirical, philosophical, practical).
    """
    llm = _get_llm(temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", DEVIL_ADVOCATE_SYSTEM),
        ("human", DEVIL_ADVOCATE_HUMAN),
    ])

    chain = prompt | llm
    response = await chain.ainvoke({
        "topic": state["topic"],
        "user_position": state["user_position"],
        "round_number": state["round_number"],
    })

    counter_args = response.content
    print(f"\n😈 [Devil's Advocate] Counter-arguments for Round {state['round_number']}:")
    print(f"   {counter_args[:200]}...\n")

    return {
        "devil_advocate_args": [counter_args],
        "dialogue_history": [
            AIMessage(content=f"[Counter-Arguments]\n{counter_args}")
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4: Socratic Questioner — generates probing follow-up questions
# ─────────────────────────────────────────────────────────────────────────────
async def socratic_questioner_node(state: DebateSession) -> dict:
    """
    Generates Socratic questions that target the weakest premises in the
    user's argument. Avoids repeating questions from previous rounds.
    """
    llm = _get_llm(temperature=0.6)

    latest_arg = next(
        (m.content for m in reversed(state["dialogue_history"])
         if isinstance(m, HumanMessage)),
        state["user_position"]
    )

    previous_q_str = "\n".join(state["socratic_questions"]) or "None yet."

    prompt = ChatPromptTemplate.from_messages([
        ("system", SOCRATIC_QUESTIONER_SYSTEM),
        ("human", SOCRATIC_QUESTIONER_HUMAN),
    ])

    chain = prompt | llm
    response = await chain.ainvoke({
        "topic": state["topic"],
        "argument": latest_arg,
        "previous_questions": previous_q_str,
    })

    questions_text = response.content
    # Parse individual questions from numbered list
    questions = [
        line.strip() for line in questions_text.split("\n")
        if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith("-"))
    ]

    print(f"🤔 [Socratic Questioner] Probing questions generated:")
    for q in questions[:2]:
        print(f"   ❓ {q}")

    return {
        "socratic_questions": questions,
        "dialogue_history": [
            AIMessage(content=f"[Socratic Questions]\n{questions_text}")
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5: Argument Scorer — scores the user's argument quality
# ─────────────────────────────────────────────────────────────────────────────
async def argument_scorer_node(state: DebateSession) -> dict:
    """
    Evaluates argument quality on 5 dimensions with an objective rubric.
    Scores are stored and used in the final coaching report.
    """
    llm = _get_llm(temperature=0.1)

    latest_arg = next(
        (m.content for m in reversed(state["dialogue_history"])
         if isinstance(m, HumanMessage)),
        state["user_position"]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", ARGUMENT_SCORER_SYSTEM),
        ("human", ARGUMENT_SCORER_HUMAN),
    ])

    chain = prompt | llm
    response = await chain.ainvoke({
        "round_number": state["round_number"],
        "argument": latest_arg,
    })

    try:
        score = _extract_json(response.content)
    except (json.JSONDecodeError, ValueError):
        score = {"total": 0, "summary": "Could not parse score."}

    total = score.get("total", 0)
    print(f"\n📊 [Scorer] Round {state['round_number']} score: {total}/50")
    print(f"   → {score.get('summary', '')}")

    return {
        "argument_scores": [{"round": state["round_number"], **score}],
        "dialogue_history": [
            AIMessage(content=f"[Score Round {state['round_number']}] {json.dumps(score)}")
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6: Round Incrementer — advances the debate to the next round
# ─────────────────────────────────────────────────────────────────────────────
async def increment_round_node(state: DebateSession) -> dict:
    """Simple node to advance the round counter."""
    new_round = state["round_number"] + 1
    print(f"\n{'─'*40}")
    print(f"  ▶  Advancing to Round {new_round}")
    print(f"{'─'*40}\n")

    # Simulate user refining their argument (in production: real user input)
    refined_arg = (
        f"[Round {new_round} — Refined position] "
        f"Building on my earlier argument about {state['topic']}: "
        f"{state['user_position']} Furthermore, the evidence suggests this is inevitable."
    )

    return {
        "round_number": new_round,
        "dialogue_history": [HumanMessage(content=refined_arg)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 7: Final Coach — synthesizes everything into a personalized report
# ─────────────────────────────────────────────────────────────────────────────
async def final_coach_node(state: DebateSession) -> dict:
    """
    The culminating agent. Reviews all rounds, fallacies, scores, and
    questions to produce a personalized coaching report with actionable advice.
    """
    llm = _get_llm(temperature=0.5)

    # Summarize collected data
    fallacies_summary = json.dumps(state["logical_fallacies_found"], indent=2) or "None detected."
    scores_summary = json.dumps(state["argument_scores"], indent=2) or "No scores recorded."
    devil_args_summary = "\n---\n".join(state["devil_advocate_args"]) or "None."
    questions_summary = "\n".join(state["socratic_questions"]) or "None."

    prompt = ChatPromptTemplate.from_messages([
        ("system", FINAL_COACH_SYSTEM),
        ("human", FINAL_COACH_HUMAN),
    ])

    chain = prompt | llm
    response = await chain.ainvoke({
        "topic": state["topic"],
        "user_position": state["user_position"],
        "fallacies_summary": fallacies_summary,
        "scores_summary": scores_summary,
        "devil_args_summary": devil_args_summary,
        "questions_summary": questions_summary,
    })

    # Extract coaching tips from the report
    tips = [
        line.strip()
        for line in response.content.split("\n")
        if line.strip().startswith(("•", "-", "Tip", "1.", "2.", "3."))
    ]

    print("\n🏆 [Final Coach] Coaching report generated!")

    return {
        "verdict": response.content,
        "coaching_tips": tips[:5],  # top 5 tips
        "dialogue_history": [AIMessage(content=f"[Final Report]\n{response.content}")],
    }