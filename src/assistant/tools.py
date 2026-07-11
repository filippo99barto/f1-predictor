from typing import Any

from src.inference.next_race import resolve_target_race
from src.models.cascade.predict import predict_next_race

FUNCTION_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "predict_next_race",
        "description": (
            "Predict qualifying and race finishing positions for the next upcoming "
            "F1 race or a specific season/round. Use for win, podium, or grid predictions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "season": {
                    "type": "integer",
                    "description": "Season year (optional; defaults to next race after latest results).",
                },
                "round": {
                    "type": "integer",
                    "description": "Round number (optional; requires season if set).",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top finishers to include (default 10).",
                },
            },
        },
    },
    {
        "name": "get_next_race_info",
        "description": (
            "Get schedule metadata for the next upcoming F1 race without running predictions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "season": {
                    "type": "integer",
                    "description": "Season year (optional).",
                },
                "round": {
                    "type": "integer",
                    "description": "Round number (optional).",
                },
            },
        },
    },
]


def handle_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "get_next_race_info":
        return _get_next_race_info(**arguments)
    if name == "predict_next_race":
        return _predict_next_race(**arguments)
    raise ValueError(f"Unknown tool: {name}")


def _get_next_race_info(
    season: int | None = None,
    round: int | None = None,
) -> dict[str, Any]:
    try:
        race = resolve_target_race(season=season, round_num=round)
    except ValueError as exc:
        return {"error": str(exc)}

    date = race.date
    if date is not None and hasattr(date, "isoformat"):
        date = date.isoformat()[:10]

    return {
        "season": race.season,
        "round": race.round,
        "race_name": race.race_name,
        "circuit_id": race.circuit_id,
        "date": date,
    }


def _predict_next_race(
    season: int | None = None,
    round: int | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    try:
        result = predict_next_race(season=season, round_num=round, mode="production")
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}

    return result.to_dict(top_n=top_n)
