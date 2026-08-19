from dataclasses import dataclass
from typing import Any

import pandas as pd

from f1_ml.inference.next_race import (
    RaceInfo,
    build_inference_frames,
    resolve_target_race,
    target_race_mask,
)
from f1_ml.models.common.load import load_model
from f1_ml.models.common.predict import drop_incomplete_feature_rows, format_driver_id
from f1_ml.models.qualifying.features import build_qualifying_results_features
from f1_ml.models.qualifying.train import FEATURE_COLS, MODEL_NAME

ID_COLS = ["season", "round", "driver_id", "constructor_id"]


@dataclass
class QualifyingPredictionResult:
    season: int
    round: int
    race_name: str
    circuit_id: str
    predictions: pd.DataFrame
    pole: str

    def to_dict(self, *, top_n: int | None = None) -> dict[str, Any]:
        preds = self.predictions.sort_values("predicted_qualifying_position")
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
                }
            )

        return {
            "season": self.season,
            "round": self.round,
            "race_name": self.race_name,
            "circuit_id": self.circuit_id,
            "pole": {
                "driver_id": self.pole,
                "driver_name": format_driver_id(self.pole),
            },
            "predictions": rows,
        }


def predict_qualifying_frame(race_info: RaceInfo, merged: pd.DataFrame) -> pd.DataFrame:
    """Run the qualifying model on a prepared inference frame."""
    qualy_model = load_model(MODEL_NAME)
    qualy_features = build_qualifying_results_features(merged, for_inference=True)
    target_qualy = qualy_features[
        target_race_mask(qualy_features, race_info.season, race_info.round)
    ]
    target_qualy = drop_incomplete_feature_rows(target_qualy, FEATURE_COLS, stage="qualifying")

    if target_qualy.empty:
        raise ValueError("No drivers with complete qualifying features for target race.")

    qualy_pred = qualy_model.predict(target_qualy[FEATURE_COLS])
    return target_qualy[ID_COLS].assign(predicted_qualifying_position=qualy_pred)


def predict_next_qualifying(
    *,
    season: int | None = None,
    round_num: int | None = None,
) -> QualifyingPredictionResult:
    """Predict qualifying positions for the next (or specified) race."""
    race_info = resolve_target_race(season=season, round_num=round_num)
    merged = build_inference_frames(
        race_info.season,
        race_info.round,
        race_info.circuit_id,
    )
    predictions = predict_qualifying_frame(race_info, merged)
    predictions = predictions.sort_values("predicted_qualifying_position").reset_index(drop=True)

    return QualifyingPredictionResult(
        season=race_info.season,
        round=race_info.round,
        race_name=race_info.race_name,
        circuit_id=race_info.circuit_id,
        predictions=predictions,
        pole=predictions.iloc[0]["driver_id"],
    )
