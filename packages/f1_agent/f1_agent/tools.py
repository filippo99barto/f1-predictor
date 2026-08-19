from typing import Any

from langchain.tools import tool
from mlflow.exceptions import MlflowException

from f1_ml.inference.next_race import resolve_target_race
from f1_ml.models.qualifying.predict import predict_next_qualifying as run_predict_next_qualifying
from f1_ml.models.race.predict import predict_next_race as run_predict_next_race


@tool
def get_next_race_info(
    season: int | None = None,
    round: int | None = None,
) -> dict[str, Any]:
    """Get schedule metadata for the next upcoming F1 race.

    Does not run predictions.

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
def predict_next_qualifying(
    season: int | None = None,
    round: int | None = None,
    top_n: int | None = None,
) -> dict[str, Any]:
    """Predict qualifying / grid positions for the next upcoming F1 race.

    Use for pole, grid, or qualifying questions. Does not predict the race.
    The result includes n_drivers, the full field size — display that many
    names for a complete grid. Do not assume a 20-car field. Each prediction
    includes features, the model inputs used for that driver. Use those values
    when asked why a driver is predicted where they are.

    Args:
        season: Season year (optional; defaults to next race after latest results).
        round: Round number (optional; requires season if set).
        top_n: Number of top qualifiers to include. Omit for the full field
            (see n_drivers in the result).
    """
    try:
        result = run_predict_next_qualifying(season=season, round_num=round)
    except (ValueError, FileNotFoundError, MlflowException) as exc:
        return {"error": str(exc)}

    return result.to_dict(top_n=top_n)


@tool
def predict_next_race(
    season: int | None = None,
    round: int | None = None,
    top_n: int | None = None,
) -> dict[str, Any]:
    """Predict race finishing positions for the next upcoming F1 race.

    Use for win, podium, or race-result questions. Uses Saturday's grid when
    qualifying results are already available; otherwise predicts qualifying
    internally. Do not call predict_next_qualifying first. The result includes
    n_drivers, the full field size — display that many names for a complete
    classification. Do not assume a 20-car field. Each prediction includes
    features, the model inputs used for that driver. Use those values when asked
    why a driver is predicted where they are.

    Args:
        season: Season year (optional; defaults to next race after latest results).
        round: Round number (optional; requires season if set).
        top_n: Number of top finishers to include. Omit for the full field
            (see n_drivers in the result).
    """
    try:
        result = run_predict_next_race(season=season, round_num=round)
    except (ValueError, FileNotFoundError, MlflowException) as exc:
        return {"error": str(exc)}

    return result.to_dict(top_n=top_n)
