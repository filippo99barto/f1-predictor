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

Rules:
- Always use the provided tools for predictions or schedule questions. Never invent race results.
- When asked who will win or about the podium, call predict_next_race.
- When asked about pole, the grid, or qualifying, call predict_next_qualifying.
- When asked when or where the next race is, call get_next_race_info.
- Present predictions clearly with driver names, predicted finishing positions, and race context.
- Mention that predictions are model estimates, not certainties.
- If a tool returns an error, explain it plainly to the user.
"""


def build_agent(*, model: str | None = None, api_key: str | None = None):
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
    )


def agent():
    """LangGraph server entrypoint (factory; persistence is injected by the server)."""
    return build_agent()
