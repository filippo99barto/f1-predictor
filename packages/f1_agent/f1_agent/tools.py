from typing import Any

from langchain.tools import tool

from f1_ml.inference.next_race import resolve_target_race
from f1_ml.models.cascade.predict import predict_next_race as run_predict_next_race


@tool
def get_next_race_info(
    season: int | None = None,
    round: int | None = None,
) -> dict[str, Any]:
    """Get schedule metadata for the next upcoming F1 race without running predictions.

    Args:
        season: Season year (optional).
        round: Round number (optional).
    """
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


@tool
def predict_next_race(
    season: int | None = None,
    round: int | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Predict qualifying and race finishing positions for the next upcoming F1 race or a specific season/round. Use for win, podium, or grid predictions.

    Args:
        season: Season year (optional; defaults to next race after latest results).
        round: Round number (optional; requires season if set).
        top_n: Number of top finishers to include (default 10).
    """
    try:
        result = run_predict_next_race(season=season, round_num=round, mode="production")
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}

    return result.to_dict(top_n=top_n)
