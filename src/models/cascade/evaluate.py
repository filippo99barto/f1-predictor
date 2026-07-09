import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.config.paths import MLFLOW_RUNS_DIR
from src.models.qualifying_results.train import (
    FEATURE_COLS as QUALIFYING_FEATURE_COLS,
    HALF_2025_ROUND,
    MODEL_NAME as QUALIFYING_MODEL_NAME,
    load_training_data as load_qualifying_data,
    split_training_data as split_qualifying_data,
)
from src.models.race_results.train import (
    FEATURE_COLS as RACE_FEATURE_COLS,
    MODEL_NAME as RACE_MODEL_NAME,
    load_training_data as load_race_data,
    split_training_data as split_race_data,
)

EXPERIMENT_NAME = "f1-cascade-predictor"

MERGE_KEYS = ["season", "round", "driverId"]


def _mae_slices(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    top10_mask = y_true <= 10
    top3_mask = y_true <= 3
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mae_top3": float(mean_absolute_error(y_true[top3_mask], y_pred[top3_mask])),
        "mae_top10": float(mean_absolute_error(y_true[top10_mask], y_pred[top10_mask])),
        "mae_p11_plus": float(mean_absolute_error(y_true[~top10_mask], y_pred[~top10_mask])),
    }


def evaluate_cascade() -> None:
    """Evaluate quali → race cascade on the held-out 2025 test split."""

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=(MLFLOW_RUNS_DIR / EXPERIMENT_NAME).as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)

    qualy_model = mlflow.sklearn.load_model(f"models:/{QUALIFYING_MODEL_NAME}/latest")
    race_model = mlflow.sklearn.load_model(f"models:/{RACE_MODEL_NAME}/latest")

    # Each model reads from its own gold table on the same held-out split (2025 rounds > 15).
    _, test_qualy = split_qualifying_data(load_qualifying_data())
    _, test_race = split_race_data(load_race_data())

    # Stage 1: quali gold → quali model
    qualy_pred = qualy_model.predict(test_qualy[QUALIFYING_FEATURE_COLS])
    quali_metrics = _mae_slices(test_qualy["qualifying_position"], qualy_pred)


    # Only predicted grid crosses into the race path — not quali features.
    qualy_predictions = test_qualy[MERGE_KEYS].assign(
        predicted_qualifying_position=qualy_pred,
        predicted_starting_position=np.round(qualy_pred).clip(1, 20),
    )

    # Stage 2: race gold + predicted grid → race model
    aligned = test_race.merge(qualy_predictions, on=MERGE_KEYS, how="inner")
    race_x = aligned[RACE_FEATURE_COLS].copy()
    race_x["qualifying_position"] = aligned["predicted_qualifying_position"]
    race_x["starting_position"] = aligned["predicted_starting_position"]

    race_pred = race_model.predict(race_x)
    race_metrics = _mae_slices(aligned["position"], race_pred)

    with mlflow.start_run(run_name="quali-to-race-cascade"):
        mlflow.log_param("qualy_model", QUALIFYING_MODEL_NAME)
        mlflow.log_param("race_model", RACE_MODEL_NAME)
        mlflow.log_param("test_split", f"2025 rounds > {HALF_2025_ROUND}")

        mlflow.log_metrics({
            "qualy_mae": quali_metrics["mae"],
            "qualy_mae_top3": quali_metrics["mae_top3"],
            "qualy_mae_top10": quali_metrics["mae_top10"],
            "qualy_mae_p11_plus": quali_metrics["mae_p11_plus"],
            "race_mae": race_metrics["mae"],
            "race_mae_top3": race_metrics["mae_top3"],
            "race_mae_top10": race_metrics["mae_top10"],
            "race_mae_p11_plus": race_metrics["mae_p11_plus"],
        })


if __name__ == "__main__":
    evaluate_cascade()
