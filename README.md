# 🎭 Socratic Debate Coach Agent

> **One agent. Five brains. Unlimited intellectual growth.**
> Built with LangChain 1.0 + LangGraph 1.0

---

## 💡 The Idea

**The Socratic Debate Coach** is an AI agent that transforms how people think and argue.
You give it a topic and your position — it becomes your toughest intellectual sparring partner,
your sharpest critic, and your most insightful mentor, all in one session.

Named after Socrates — the philosopher who believed truth is reached not by lecturing,
but by relentless questioning — this agent challenges every assumption you make,
pokes holes in every argument, then helps you rebuild stronger.

---

## 🎯 Use Cases

| Who | How They Use It |
|-----|----------------|
| **Students** | Prepare for debate competitions, philosophy exams, oral defenses |
| **Professionals** | Sharpen business pitches, policy arguments, and negotiation positions |
| **Writers** | Stress-test the logic in essays, op-eds, and persuasive content |
| **Curious minds** | Explore difficult topics (AI ethics, politics, philosophy) with depth |
| **Teams** | Pre-mortem analysis — stress-test decisions before committing |

---

## 🚀 Added Value

### What makes this genuinely different from "just asking ChatGPT":

1. **Stateful multi-round debate** — Not a single response. A real 3-round session where
   the agent *remembers* your previous arguments and escalates its challenges.

2. **5 specialized sub-agents, not 1** — Each node is an expert in its domain:
   - 🔍 Fallacy Detector: Trained to spot 20+ logical fallacies with precision
   - 😈 Devil's Advocate: Generates the *best possible* counter-arguments across 3 angles
   - 🤔 Socratic Questioner: Never answers — only asks progressively deeper questions
   - 📊 Argument Scorer: Objective rubric scoring (Clarity, Evidence, Logic, Originality, Persuasion)
   - 🏆 Final Coach: Synthesizes everything into a personalized improvement report

3. **Durable state (LangGraph 1.0 checkpointing)** — Sessions survive crashes, can be
   paused and resumed. Perfect for long coaching sessions or async workflows.

4. **Parallel execution** — Fallacy detection, devil's advocacy, and Socratic questioning
   run simultaneously, not sequentially. Fast even with multiple agents.

5. **Provider-agnostic** — One env variable switches between GPT-4o, Claude 3.5, or Gemini.

---

## 🏗️ Architecture

```
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
                             │ (fan-in + conditional)
              ┌──────────────┴──────────────┐
              │ rounds < MAX?               │
              │  YES → increment_round ─────┘ (loop)
              │  NO  → final_coach → END
              └─────────────────────────────┘
```

**LangGraph 1.0 features used:**
- `StateGraph` with `TypedDict` state + `Annotated` reducers
- Parallel node execution (fan-out edges)
- Conditional edges for loop control
- `MemorySaver` checkpointer for durable execution
- Async streaming with `astream()`

---

## 📁 Project Structure

```
debate_agent/
├── main.py                    # Entry point — run a debate session
│
├── models/
│   └── state.py               # DebateSession TypedDict with Annotated reducers
│
├── graph/
│   └── debate_graph.py        # LangGraph graph definition + compilation
│
├── agents/
│   └── nodes.py               # All 7 node implementations
│
├── prompts/
│   └── templates.py           # All system + human prompts for each agent
│
├── tools/
│   └── debate_tools.py        # LangChain @tool definitions (search, save, tips)
│
├── utils/
│   └── visualization.py       # Score cards, banners, and session summaries
│
├── requirements.txt           # Pinned dependencies
└── README.md                  # This file
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment

```bash
# .env
OPENAI_API_KEY=sk-...
LANGCHAIN_MODEL=openai:gpt-4o        # or anthropic:claude-3-5-sonnet-20241022
LANGCHAIN_TRACING_V2=true            # Optional: LangSmith observability
LANGCHAIN_API_KEY=ls-...             # Optional: LangSmith API key
```

### 3. Run a debate

```bash
python main.py
```

### 4. Or use the API directly

```python
import asyncio
from graph.debate_graph import build_debate_graph
from models.state import DebateSession

async def my_debate():
    graph = build_debate_graph()

    result = await graph.ainvoke(
        DebateSession(
            topic="Universal Basic Income will reduce poverty",
            user_position="UBI gives everyone a safety net, so poverty becomes a choice not a trap.",
            round_number=0,
            dialogue_history=[],
            logical_fallacies_found=[],
            argument_scores=[],
            devil_advocate_args=[],
            socratic_questions=[],
            verdict="",
            coaching_tips=[],
        ),
        config={"configurable": {"thread_id": "my-session-001"}}  # enables checkpointing
    )

    print(result["verdict"])

asyncio.run(my_debate())
```

---

## 🔧 Configuration

### Switch LLM provider

```bash
# OpenAI (default)
LANGCHAIN_MODEL=openai:gpt-4o

# Anthropic
LANGCHAIN_MODEL=anthropic:claude-3-5-sonnet-20241022

# Google
LANGCHAIN_MODEL=google_vertexai:gemini-2.0-flash
```

### Change number of debate rounds

In `graph/debate_graph.py`:
```python
MAX_ROUNDS = 3   # default — increase for deeper sessions
```

---

## 🧪 Example Output

```
╔══════════════════════════════════════════════════════════════╗
║            🎭 SOCRATIC DEBATE COACH AGENT                    ║
║       Powered by LangChain 1.0 + LangGraph 1.0              ║
╠══════════════════════════════════════════════════════════════╣
║ Topic: AGI will be beneficial for humanity                   ║
║ Rounds: 3 | Agents: 5 specialized nodes                      ║
╚══════════════════════════════════════════════════════════════╝

🎙️  [Round 1] User position registered. Starting debate analysis...

🔍 [Fallacy Detector] Found 1 fallacy:
   ⚠️  Overgeneralization — "will solve ALL problems" is unsubstantiated...

😈 [Devil's Advocate] Counter-arguments for Round 1:
   **Empirical**: There is no existing evidence that any system...

🤔 [Socratic Questioner] Probing questions:
   ❓ 1. What specific mechanism do you believe ensures AGI will...
   ❓ 2. When you say "beneficial," beneficial to whom, exactly?

📊 [Scorer] Round 1 score: 31/50
   → Promising foundation but needs empirical grounding.

════════════════════  Round 2  ════════════════════

...

🏆 [Final Coach] Coaching report generated!

══════════════════════════════════════════════════
  📊 FINAL VERDICT & COACHING REPORT
══════════════════════════════════════════════════
Overall Assessment: Your arguments showed real intellectual ambition...
```

---

## 🔮 Extending the Agent

### Add human-in-the-loop (LangGraph 1.0 native)

```python
# Pause the graph after devil's advocate for user to respond
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["devil_advocate"]  # pause here for human input
)
```

### Add long-term memory across sessions

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
# Stores user debate history, improvement trends, and preferred topics
graph = builder.compile(checkpointer=MemorySaver(), store=store)
```

### Add a new specialist agent

1. Write the node function in `agents/nodes.py`
2. Add the prompt in `prompts/templates.py`
3. Register the node and edge in `graph/debate_graph.py`

---

## 📄 License

MIT License — free to use, modify, and deploy.
