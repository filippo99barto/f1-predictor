import json
import os
from typing import Any

from google import genai
from google.genai import types

from src.assistant.tools import FUNCTION_DECLARATIONS, handle_tool_call

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
- When asked when or where the next race is, call get_next_race_info.
- Present predictions clearly with driver names, predicted finishing positions, and race context.
- Mention that predictions are model estimates, not certainties.
- If a tool returns an error, explain it plainly to the user.
"""


def ask(question: str, *, model: str | None = None) -> str:
    """Ask a natural-language question; returns the assistant's final answer."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    resolved_model = model or DEFAULT_MODEL

    declarations = [
        types.FunctionDeclaration(**declaration)
        for declaration in FUNCTION_DECLARATIONS
    ]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=declarations)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=question)]),
    ]

    while True:
        response = client.models.generate_content(
            model=resolved_model,
            contents=contents,
            config=config,
        )
        candidate = response.candidates[0].content
        function_calls = [
            part.function_call
            for part in candidate.parts
            if part.function_call is not None
        ]

        if not function_calls:
            return response.text or ""

        contents.append(candidate)
        for function_call in function_calls:
            args = dict(function_call.args) if function_call.args else {}
            result = handle_tool_call(function_call.name, args)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=function_call.name,
                            response=result,
                        )
                    ],
                )
            )
