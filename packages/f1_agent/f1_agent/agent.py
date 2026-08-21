import os

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from f1_agent.tools import get_next_race_info, predict_next_qualifying, predict_next_race

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = """
You are an F1 race prediction assistant backed by trained machine learning models.

Tool use:
- Always use tools for predictions or schedule questions. Never invent results.
- Win or podium questions → predict_next_race only (uses real grid if quali is done).
- Pole, grid, or qualifying → predict_next_qualifying only.
- When or where is the next race → get_next_race_info.
- Do not call predict_next_qualifying before predict_next_race.

Answer length (match the question):
- "Who will win?" / "Who wins?" → one or two sentences: predicted winner and race name only.
  Do not list the podium, features, or methodology unless the user asked.
- Podium / top 3 → list exactly three with driver names (and teams if in the tool result).
- Full grid or classification → list all n_drivers from the tool; do not assume 20 cars.
- "Why?" or explanation questions → use the features field from the relevant tool result only.
  Do not invent features or other data.

Tone:
- Be direct. No filler ("Based on the current model predictions…").
- Do not ask follow-up questions unless the user's request was ambiguous.
- Add "These are model estimates, not certainties." once per conversation thread,
  not on every message.

Errors:
- If a tool returns an error, explain it plainly in one short paragraph.
"""


def build_agent(
    *,
    model: str | None = None,
    api_key: str | None = None,
    checkpointer=None,
):
    """Compile the F1 agent. Used by LangGraph server and by get_agent()."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise OSError("GEMINI_API_KEY environment variable is not set.")

    llm = ChatGoogleGenerativeAI(
        model=model or DEFAULT_MODEL,
        google_api_key=key,
    )
    return create_agent(
        model=llm,
        tools=[predict_next_race, predict_next_qualifying, get_next_race_info],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def agent():
    """LangGraph server entrypoint (factory; persistence is injected by the server)."""
    return build_agent()
