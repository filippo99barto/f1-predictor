from typing import Literal

import pandas as pd

from f1_ml.models.common.config import ModelTrainConfig
from f1_ml.models.common.trainer import load_training_data as _load_training_data
from f1_ml.models.common.trainer import train_model as _train_model

EXPERIMENT_NAME = "f1-race-results-predictor"
MODEL_NAME = "f1_position_predictor"

FEATURE_COLS = [
    "qualifying_position",
    "driver_last_race_position",
    "driver_median_position_last_3_races",
    "constructor_median_position_last_3_races",
    "driver_circuit_median_position_last_3_races",
    "driver_season_median_position",
    "driver_positions_gained_season_median",
    "driver_positions_gained_career_median",
    "driver_circuit_median_career_position",
    "constructor_median_season_position",
    "driver_qualifying_season_median",
]

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_child_weight": 100,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "objective": "reg:absoluteerror",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 10,
    "eval_metric": "mae",
    "reg_lambda": 1,
}

STATUS_FILTER = ["Finished", "Lapped", "+1 Lap", "+2 Laps"]

CONFIG = ModelTrainConfig(
    experiment_name=EXPERIMENT_NAME,
    model_name=MODEL_NAME,
    gold_schema="gold",
    gold_table="race_results",
    feature_cols=FEATURE_COLS,
    target_col="position",
    baseline_col="qualifying_position",
    xgb_params=XGB_PARAMS,
    train_row_filter=lambda df: df[df["status"].isin(STATUS_FILTER)],
)


def train_model(
    mode: Literal["dev", "production"] = "dev",
    holdout_fraction: float = 0.5,
    retrain_on_full_data: bool | None = None,
    register_model: bool | None = None,
) -> None:
    """Train the race results model."""
    _train_model(
        CONFIG,
        mode=mode,
        holdout_fraction=holdout_fraction,
        retrain_on_full_data=retrain_on_full_data,
        register_model=register_model,
    )


def load_training_data() -> pd.DataFrame:
    """Load the race training data."""
    return _load_training_data(CONFIG)
