from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from f1_ml.inference.next_race import build_inference_frames, resolve_target_race, target_race_mask
from f1_ml.models.common.load import load_model
from f1_ml.models.common.predict import drop_incomplete_feature_rows, format_driver_id
from f1_ml.models.qualifying.predict import predict_qualifying_frame
from f1_ml.models.race.features import build_race_results_features
from f1_ml.models.race.train import FEATURE_COLS, MODEL_NAME

MIN_ACTUAL_GRID = 18


@dataclass
class RacePredictionResult:
    season: int
    round: int
    race_name: str
    circuit_id: str
    predictions: pd.DataFrame
    winner: str
    podium: list[str]
    grid_source: Literal["actual", "predicted"]

    def to_dict(self, *, top_n: int | None = None) -> dict[str, Any]:
        preds = self.predictions.sort_values("predicted_race_position")
        if top_n is not None:
            preds = preds.head(top_n)

        rows = []
        for _, row in preds.iterrows():
            rows.append(
                {
                    "driver_id": row["driver_id"],
                    "driver_name": format_driver_id(row["driver_id"]),
                    "constructor_id": row["constructor_id"],
                    "predicted_qualifying_position": float(row["predicted_qualifying_position"]),
                    "predicted_race_position": float(row["predicted_race_position"]),
                }
            )

        return {
            "season": self.season,
            "round": self.round,
            "race_name": self.race_name,
            "circuit_id": self.circuit_id,
            "grid_source": self.grid_source,
            "n_drivers": int(len(self.predictions)),
            "winner": {
                "driver_id": self.winner,
                "driver_name": format_driver_id(self.winner),
            },
            "podium": [{"driver_id": d, "driver_name": format_driver_id(d)} for d in self.podium],
            "predictions": rows,
        }


def _has_actual_grid(merged: pd.DataFrame, season: int, round_num: int) -> bool:
    target = merged[target_race_mask(merged, season, round_num)]
    filled = pd.to_numeric(target["qualifying_position"], errors="coerce").notna().sum()
    return int(filled) >= MIN_ACTUAL_GRID


def _grid_from_actual(merged: pd.DataFrame, season: int, round_num: int) -> pd.DataFrame:
    target = merged[target_race_mask(merged, season, round_num)]
    grid = target.loc[
        pd.to_numeric(target["qualifying_position"], errors="coerce").notna(),
        ["driver_id", "constructor_id", "qualifying_position"],
    ].copy()
    return grid.rename(columns={"qualifying_position": "predicted_qualifying_position"})


def predict_next_race(
    *,
    season: int | None = None,
    round_num: int | None = None,
) -> RacePredictionResult:
    """Predict race finishing positions.

    Uses Saturday's qualifying results when they are already in silver; otherwise runs the
    qualifying model to fill grid features.
    """
    race_info = resolve_target_race(season=season, round_num=round_num)
    merged = build_inference_frames(
        race_info.season,
        race_info.round,
        race_info.circuit_id,
    )

    race_model = load_model(MODEL_NAME)
    merged_for_race = merged.copy()
    target_mask = target_race_mask(merged_for_race, race_info.season, race_info.round)

    if _has_actual_grid(merged_for_race, race_info.season, race_info.round):
        grid_source: Literal["actual", "predicted"] = "actual"
        qualy_predictions = _grid_from_actual(merged_for_race, race_info.season, race_info.round)
    else:
        grid_source = "predicted"
        qualy_predictions = predict_qualifying_frame(race_info, merged)
        quali_lookup = qualy_predictions.set_index("driver_id")["predicted_qualifying_position"]
        for driver_id, predicted_quali in quali_lookup.items():
            driver_mask = target_mask & (merged_for_race["driver_id"] == driver_id)
            merged_for_race.loc[driver_mask, "qualifying_position"] = float(predicted_quali)

    race_features = build_race_results_features(merged_for_race, for_inference=True)
    target_race = race_features[target_race_mask(race_features, race_info.season, race_info.round)]
    target_race = drop_incomplete_feature_rows(target_race, FEATURE_COLS, stage="race")

    if target_race.empty:
        raise ValueError("No drivers with complete race features for target race.")

    race_x = target_race[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        race_x[col] = pd.to_numeric(race_x[col], errors="coerce")
    race_pred = race_model.predict(race_x)

    predictions = target_race[["driver_id", "constructor_id"]].copy()
    predictions = predictions.merge(
        qualy_predictions[["driver_id", "predicted_qualifying_position"]],
        on="driver_id",
        how="left",
    )
    predictions["predicted_race_position"] = race_pred
    predictions = predictions.sort_values("predicted_race_position").reset_index(drop=True)

    winner = predictions.iloc[0]["driver_id"]
    podium = predictions.head(3)["driver_id"].tolist()

    return RacePredictionResult(
        season=race_info.season,
        round=race_info.round,
        race_name=race_info.race_name,
        circuit_id=race_info.circuit_id,
        predictions=predictions,
        winner=winner,
        podium=podium,
        grid_source=grid_source,
    )
