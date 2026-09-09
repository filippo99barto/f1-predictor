import logging
from typing import Any

from langchain.tools import tool

from f1_ml.inference.next_race import resolve_target_race
from f1_ml.models.qualifying.predict import predict_next_qualifying as run_predict_next_qualifying
from f1_ml.models.race.predict import predict_next_race as run_predict_next_race

logger = logging.getLogger(__name__)


@tool
def get_next_race_info() -> dict[str, Any]:
    """Get schedule metadata for the next upcoming F1 race.

    Does not run predictions.
    """
    try:
        race = resolve_target_race()
    except Exception as exc:
        logger.warning("get_next_race_info failed: %s", exc)
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
def predict_next_qualifying(top_n: int | None = None) -> dict[str, Any]:
    """Predict qualifying / grid positions for the next upcoming F1 race.

    Use for pole, grid, or qualifying questions. Does not predict the race.
    The result includes n_drivers, the full field size — display that many
    names for a complete grid. Do not assume a 20-car field. Each prediction
    includes features, the model inputs used for that driver. Use those values
    when asked why a driver is predicted where they are.

    Args:
        top_n: Number of top qualifiers to include. Omit for the full field
            (see n_drivers in the result).
    """
    try:
        result = run_predict_next_qualifying()
    except Exception as exc:
        logger.warning("predict_next_qualifying failed: %s", exc)
        return {"error": str(exc)}

    return result.to_dict(top_n=top_n)


@tool
def predict_next_race(top_n: int | None = None) -> dict[str, Any]:
    """Predict race finishing positions for the next upcoming F1 race.

    Use for win, podium, or race-result questions. Uses Saturday's grid when
    qualifying results are already available; otherwise predicts qualifying
    internally. Do not call predict_next_qualifying first. The result includes
    n_drivers, the full field size — display that many names for a complete
    classification. Do not assume a 20-car field. Each prediction includes
    features, the model inputs used for that driver. Use those values when asked
    why a driver is predicted where they are.

    Args:
        top_n: Number of top finishers to include. Omit for the full field
            (see n_drivers in the result).
    """
    try:
        result = run_predict_next_race()
    except Exception as exc:
        logger.warning("predict_next_race failed: %s", exc)
        return {"error": str(exc)}

    return result.to_dict(top_n=top_n)
