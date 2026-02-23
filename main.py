import asyncio
from dotenv import load_dotenv


load_dotenv()

from graph.debate_graph import build_debate_graph
from models.state import DebateSession


async def run_session(topic: str, user_position: str):
    graph = build_debate_graph()

    initial_state = DebateSession(
        topic=topic,
        user_position=user_position,
        round_number=0,
        dialogue_history=[],
        logical_fallacies_found=[],
        argument_scores=[],
        devil_advocate_args=[],
        socratic_questions=[],
        verdict="",
        coaching_tips=[],
    )

    print(f"\n{'═'*60}")
    print(f"  🎭 SOCRATIC DEBATE COACH — Topic: {topic}")
    print(f"{'═'*60}\n")

    config = {"configurable": {"thread_id": "debate-session-1"}}
    async for event in graph.astream(initial_state, config=config, stream_mode="values"):
        if event.get("verdict"):
            print(f"\n{'═'*60}")
            print("  📊 FINAL VERDICT & COACHING REPORT")
            print(f"{'═'*60}")
            print(event["verdict"])
            for tip in event.get("coaching_tips", []):
                print(f"  💡 {tip}")
            break


if __name__ == "__main__":
    asyncio.run(
        run_session(
            topic="Artificial General Intelligence will be beneficial for humanity",
            user_position="I believe AGI will bring enormous benefits because it will solve problems humans cannot.",
        )
    )