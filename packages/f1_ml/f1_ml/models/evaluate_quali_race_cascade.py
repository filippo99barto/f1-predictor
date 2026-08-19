from typing import Literal

import mlflow
import pandas as pd

from f1_ml.models.common.load import load_model
from f1_ml.models.common.metrics import mae_slices
from f1_ml.models.common.splits import split_last_race_holdout, split_latest_season_fraction
from f1_ml.models.qualifying.train import FEATURE_COLS as QUALIFYING_FEATURE_COLS
from f1_ml.models.qualifying.train import MODEL_NAME as QUALIFYING_MODEL_NAME
from f1_ml.models.qualifying.train import load_training_data as load_qualifying_data
from f1_ml.models.race.train import FEATURE_COLS as RACE_FEATURE_COLS
from f1_ml.models.race.train import MODEL_NAME as RACE_MODEL_NAME
from f1_ml.models.race.train import load_training_data as load_race_data

EXPERIMENT_NAME = "f1-cascade-predictor"

ALIGN_KEYS = ["season", "round", "driver_id"]


def _get_holdout_test(
    df: pd.DataFrame,
    mode: Literal["dev", "production"],
    holdout_fraction: float,
) -> pd.DataFrame:
    if mode == "dev":
        _, test_df, _ = split_latest_season_fraction(df, holdout_fraction)
    else:
        _, test_df, _ = split_last_race_holdout(df)
    return test_df


def evaluate_quali_race_cascade(
    mode: Literal["dev", "production"] = "dev",
    holdout_fraction: float = 0.5,
) -> None:
    """Evaluate qualifying → race composition on the same holdout split used for training."""

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(name=EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    qualy_model = load_model(QUALIFYING_MODEL_NAME)
    race_model = load_model(RACE_MODEL_NAME)

    test_qualy = _get_holdout_test(load_qualifying_data(), mode, holdout_fraction)
    test_race = _get_holdout_test(load_race_data(), mode, holdout_fraction)

    qualy_pred = qualy_model.predict(test_qualy[QUALIFYING_FEATURE_COLS])
    quali_metrics = mae_slices(test_qualy["qualifying_position"], qualy_pred)

    qualy_predictions = test_qualy[ALIGN_KEYS].assign(
        predicted_qualifying_position=qualy_pred,
    )

    aligned = test_race.merge(qualy_predictions, on=ALIGN_KEYS, how="inner")
    race_x = aligned[RACE_FEATURE_COLS].copy()
    race_x["qualifying_position"] = aligned["predicted_qualifying_position"]

    race_pred = race_model.predict(race_x)
    race_metrics = mae_slices(aligned["position"], race_pred)

    with mlflow.start_run(run_name="quali-to-race-cascade"):
        mlflow.log_param("mode", mode)
        mlflow.log_param("qualy_model", QUALIFYING_MODEL_NAME)
        mlflow.log_param("race_model", RACE_MODEL_NAME)
        if mode == "dev":
            mlflow.log_param("holdout_fraction", holdout_fraction)

        mlflow.log_metrics(
            {
                "qualy_mae": quali_metrics["mae"],
                "qualy_mae_top3": quali_metrics["mae_top3"],
                "qualy_mae_top10": quali_metrics["mae_top10"],
                "qualy_mae_p11_plus": quali_metrics["mae_p11_plus"],
                "race_mae": race_metrics["mae"],
                "race_mae_top3": race_metrics["mae_top3"],
                "race_mae_top10": race_metrics["mae_top10"],
                "race_mae_p11_plus": race_metrics["mae_p11_plus"],
            }
        )


if __name__ == "__main__":
    evaluate_quali_race_cascade()
