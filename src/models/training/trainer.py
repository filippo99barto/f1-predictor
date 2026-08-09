from typing import Literal

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config.paths import LOCAL_DATA_DIR, LOCAL_MODELS_DIR, MLFLOW_RUNS_DIR
from src.models.training.config import ModelTrainConfig
from src.models.training.metrics import baseline_mae, mae_slices
from src.models.training.splits import (
    SplitInfo,
    get_early_stopping_val_set,
    split_last_race_holdout,
    split_latest_season_fraction,
)
from src.storage.local_storage_backend import LocalStorageBackend

MODEL_TYPE = "XGBRegressor"
SKOPS_TRUSTED_TYPES = ["xgboost.sklearn.XGBRegressor", "xgboost.core.Booster"]


def load_training_data(config: ModelTrainConfig) -> pd.DataFrame:
    storage_backend = LocalStorageBackend(LOCAL_DATA_DIR)
    df = storage_backend.read(config.gold_path)
    return df.dropna(subset=config.feature_cols)


def _apply_train_filter(
    df: pd.DataFrame,
    config: ModelTrainConfig,
) -> pd.DataFrame:
    if config.train_row_filter is None:
        return df
    return config.train_row_filter(df)


def _fit_model(
    model: Pipeline,
    train_df: pd.DataFrame,
    config: ModelTrainConfig,
) -> None:
    fit_df, val_df = get_early_stopping_val_set(train_df)
    model.fit(
        fit_df[config.feature_cols],
        fit_df[config.target_col],
        reg__eval_set=[(val_df[config.feature_cols], val_df[config.target_col])],
        reg__verbose=False,
    )


def _pick_split(
    df: pd.DataFrame,
    mode: Literal["dev", "production"],
    holdout_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    if mode == "dev":
        return split_latest_season_fraction(df, holdout_fraction)
    return split_last_race_holdout(df)


def train_model(
    config: ModelTrainConfig,
    *,
    mode: Literal["dev", "production"] = "dev",
    holdout_fraction: float = 0.5,
    retrain_on_full_data: bool | None = None,
    register_model: bool | None = None,
) -> None:
    if retrain_on_full_data is None:
        retrain_on_full_data = mode == "production"
    if register_model is None:
        register_model = mode == "production"

    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            name=config.experiment_name,
            artifact_location=(MLFLOW_RUNS_DIR / config.experiment_name).as_uri(),
        )
    mlflow.set_experiment(config.experiment_name)

    df = load_training_data(config)
    train_df, test_df, split_info = _pick_split(df, mode, holdout_fraction)
    train_df = _apply_train_filter(train_df, config)

    x_test = test_df[config.feature_cols]
    y_test = test_df[config.target_col]

    model = Pipeline([("reg", XGBRegressor(**config.xgb_params))])

    with mlflow.start_run(run_name=f"{MODEL_TYPE}-{config.experiment_name}"):
        mlflow.log_param("mode", mode)
        mlflow.log_param("split_strategy", split_info.split_strategy)
        mlflow.log_param("train_split", split_info.train_desc)
        mlflow.log_param("test_split", split_info.test_desc)
        mlflow.log_param("holdout_season", split_info.holdout_season)
        mlflow.log_param("holdout_rounds", split_info.holdout_rounds)
        if mode == "dev":
            mlflow.log_param("holdout_fraction", holdout_fraction)
        mlflow.log_param("xgb_params", config.xgb_params)
        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param("features", config.feature_cols)
        mlflow.log_param("retrained_on_full_data", retrain_on_full_data)
        mlflow.log_param("registered", register_model)

        _fit_model(model, train_df, config)

        y_pred = model.predict(x_test)
        metrics = mae_slices(y_test, y_pred)
        metrics["baseline_mae"] = baseline_mae(y_test, x_test[config.baseline_col])
        metrics["best_iteration"] = float(model.named_steps["reg"].best_iteration)
        mlflow.log_metrics(metrics)

        if retrain_on_full_data:
            full_train_df = _apply_train_filter(df, config)
            _fit_model(model, full_train_df, config)
            mlflow.log_param("full_train_rows", len(full_train_df))

        model_dir = LOCAL_MODELS_DIR / config.model_subdir
        model_dir.mkdir(parents=True, exist_ok=True)

        if register_model:
            mlflow.sklearn.log_model(
                model,
                name=config.model_name,
                registered_model_name=config.model_name,
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            joblib.dump(model, model_dir / f"{config.model_name}.pkl")
        elif mode == "dev":
            joblib.dump(model, model_dir / f"{config.model_name}-dev.pkl")
