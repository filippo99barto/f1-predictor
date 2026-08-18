from typing import Literal

import pandas as pd

from f1_ml.models.training.config import ModelTrainConfig
from f1_ml.models.training.trainer import load_training_data as _load_training_data
from f1_ml.models.training.trainer import train_model as _train_model

EXPERIMENT_NAME = "f1-qualifying-results-predictor"
MODEL_NAME = "f1_qualifying_predictor"

FEATURE_COLS = [
    "driver_positions_gained_season_median",
    "driver_positions_gained_career_median",
    "driver_last_qualifying_position",
    "last_qualifying_gap_to_pole",
    "driver_qualifying_season_median",
    "driver_qualifying_career_median",
    "driver_qualifying_circuit_median",
    "constructor_qualifying_season_median",
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

CONFIG = ModelTrainConfig(
    experiment_name=EXPERIMENT_NAME,
    model_name=MODEL_NAME,
    gold_path="gold/qualifying_results",
    model_subdir="qualifying_results",
    feature_cols=FEATURE_COLS,
    target_col="qualifying_position",
    baseline_col="driver_last_qualifying_position",
    xgb_params=XGB_PARAMS,
)


def train_model(
    mode: Literal["dev", "production"] = "dev",
    holdout_fraction: float = 0.5,
    retrain_on_full_data: bool | None = None,
    register_model: bool | None = None,
) -> None:
    """Train the qualifying results model."""
    _train_model(
        CONFIG,
        mode=mode,
        holdout_fraction=holdout_fraction,
        retrain_on_full_data=retrain_on_full_data,
        register_model=register_model,
    )


def load_training_data() -> pd.DataFrame:
    """Load the qualifying training data."""
    return _load_training_data(CONFIG)
