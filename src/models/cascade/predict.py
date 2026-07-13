import logging
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from src.inference.next_race import (
    RaceInfo,
    build_inference_frames,
    resolve_target_race,
    target_race_mask,
)
from src.models.cascade.evaluate import MERGE_KEYS
from src.models.cascade.load import load_model
from src.models.qualifying_results.features import build_qualifying_results_features
from src.models.qualifying_results.train import (
    CONFIG as QUALIFYING_CONFIG,
    FEATURE_COLS as QUALIFYING_FEATURE_COLS,
    MODEL_NAME as QUALIFYING_MODEL_NAME,
)
from src.models.race_results.features import build_race_results_features
from src.models.race_results.train import (
    CONFIG as RACE_CONFIG,
    FEATURE_COLS as RACE_FEATURE_COLS,
    MODEL_NAME as RACE_MODEL_NAME,
)

logger = logging.getLogger(__name__)


def format_driver_id(driver_id: str) -> str:
    return driver_id.replace("_", " ").title()


@dataclass
class PredictionResult:
    season: int
    round: int
    race_name: str
    circuit_id: str
    predictions: pd.DataFrame
    winner: str
    podium: list[str]

    def to_dict(self, *, top_n: int | None = None) -> dict[str, Any]:
        preds = self.predictions.sort_values("predicted_race_position")
        if top_n is not None:
            preds = preds.head(top_n)

        rows = []
        for _, row in preds.iterrows():
            rows.append({
                "driver_id": row["driverId"],
                "driver_name": format_driver_id(row["driverId"]),
                "constructor_id": row["constructorId"],
                "predicted_qualifying_position": float(row["predicted_qualifying_position"]),
                "predicted_race_position": float(row["predicted_race_position"]),
            })

        return {
            "season": self.season,
            "round": self.round,
            "race_name": self.race_name,
            "circuit_id": self.circuit_id,
            "winner": {
                "driver_id": self.winner,
                "driver_name": format_driver_id(self.winner),
            },
            "podium": [
                {"driver_id": d, "driver_name": format_driver_id(d)}
                for d in self.podium
            ],
            "predictions": rows,
        }


def _predictable_rows(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    valid = df.dropna(subset=feature_cols)
    dropped = len(df) - len(valid)
    if dropped:
        logger.warning("Dropped %d drivers with incomplete features.", dropped)
    return valid


def predict_next_race(
    *,
    season: int | None = None,
    round_num: int | None = None,
    mode: Literal["dev", "production"] = "production",
) -> PredictionResult:
    """Predict qualifying and race finishing positions for the next (or specified) race."""
    race_info = resolve_target_race(season=season, round_num=round_num)
    merged = build_inference_frames(
        race_info.season,
        race_info.round,
        race_info.circuit_id,
    )

    qualy_model = load_model(
        QUALIFYING_MODEL_NAME, QUALIFYING_CONFIG.model_subdir, mode
    )
    race_model = load_model(RACE_MODEL_NAME, RACE_CONFIG.model_subdir, mode)

    qualy_features = build_qualifying_results_features(merged, for_inference=True)
    target_qualy = qualy_features[target_race_mask(
        qualy_features, race_info.season, race_info.round
    )]
    target_qualy = _predictable_rows(target_qualy, QUALIFYING_FEATURE_COLS, stage="qualifying")

    if target_qualy.empty:
        raise ValueError("No drivers with complete qualifying features for target race.")

    qualy_pred = qualy_model.predict(target_qualy[QUALIFYING_FEATURE_COLS])
    qualy_predictions = target_qualy[MERGE_KEYS].assign(
        predicted_qualifying_position=qualy_pred,
    )

    merged_for_race = merged.copy()
    target_mask = target_race_mask(merged_for_race, race_info.season, race_info.round)
    quali_lookup = qualy_predictions.set_index("driverId")["predicted_qualifying_position"]
    for driver_id, predicted_quali in quali_lookup.items():
        driver_mask = target_mask & (merged_for_race["driverId"] == driver_id)
        merged_for_race.loc[driver_mask, "qualifying_position"] = float(predicted_quali)

    race_features = build_race_results_features(merged_for_race, for_inference=True)
    target_race = race_features[target_race_mask(
        race_features, race_info.season, race_info.round
    )]
    target_race = _predictable_rows(target_race, RACE_FEATURE_COLS, stage="race")

    if target_race.empty:
        raise ValueError("No drivers with complete race features for target race.")

    race_x = target_race[RACE_FEATURE_COLS].copy()
    for col in RACE_FEATURE_COLS:
        race_x[col] = pd.to_numeric(race_x[col], errors="coerce")
    race_pred = race_model.predict(race_x)

    predictions = target_race[["driverId", "constructorId"]].copy()
    predictions = predictions.merge(
        qualy_predictions[["driverId", "predicted_qualifying_position"]],
        on="driverId",
        how="left",
    )
    predictions["predicted_race_position"] = race_pred
    predictions = predictions.sort_values("predicted_race_position").reset_index(drop=True)

    winner = predictions.iloc[0]["driverId"]
    podium = predictions.head(3)["driverId"].tolist()

    return PredictionResult(
        season=race_info.season,
        round=race_info.round,
        race_name=race_info.race_name,
        circuit_id=race_info.circuit_id,
        predictions=predictions,
        winner=winner,
        podium=podium,
    )
