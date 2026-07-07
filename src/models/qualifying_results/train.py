import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from src.config.paths import LOCAL_MODELS_DIR, MLFLOW_RUNS_DIR

EXPERIMENT_NAME = "f1-qualifying-results-predictor"
MODEL_NAME = "f1_qualifying_predictor"
MODEL_TYPE = "XGBRegressor"

HALF_2025_ROUND = 15

FEATURE_COLS = [
    # "driver_last_race_position",
    # "driver_median_position_last_3_races",
    # "constructor_median_position_last_3_races",
    # "driver_circuit_median_position_last_3_races",
    # "driver_season_median_position",
    "driver_positions_gained_season_median",
    "driver_positions_gained_career_median",
    # "driver_circuit_median_career_position",
    # "constructor_median_season_position",

    #"driver_grid_season_median",
    #"qualifying_gap_to_pole", # THIS IS NOT POSSIBLE TO HAVE AS QUALIGYINF HASNT HAPPENED YET
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

def train_model() -> None:
    """Train the model."""

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=(MLFLOW_RUNS_DIR / EXPERIMENT_NAME).as_uri()
        )

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_training_data()
    train_df, test_df = split_training_data(df)

    x_test = test_df[FEATURE_COLS]
    y_test = test_df["grid_position"]

    model = Pipeline([("reg", XGBRegressor(**XGB_PARAMS))])

    with mlflow.start_run(run_name=f"{MODEL_TYPE}-{EXPERIMENT_NAME}"):
        mlflow.log_param("xgb_params", XGB_PARAMS)
        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param(f"train_split", f"seasons < 2025 + 2025 rounds 1-{HALF_2025_ROUND}")
        mlflow.log_param("features", FEATURE_COLS)

        val_df = train_df[(train_df["season"] == 2025) & (train_df["round"] >= 5)]
        fit_df = train_df[~train_df.index.isin(val_df.index)]

        model.fit(
            fit_df[FEATURE_COLS],
            fit_df["grid_position"],
            reg__eval_set=[(val_df[FEATURE_COLS], val_df["grid_position"])],
            reg__verbose=True
        )

        y_pred = model.predict(x_test)       


        
        baseline_model = mean_absolute_error(y_test, x_test["driver_last_qualifying_position"])
        best_iter = model.named_steps["reg"].best_iteration
        mae = mean_absolute_error(y_test, y_pred)
        top10_mask = y_test <= 10
        top3_mask = y_test <= 3
        mae_top3 = mean_absolute_error(y_test[top3_mask], y_pred[top3_mask])
        mae_top10 = mean_absolute_error(y_test[top10_mask], y_pred[top10_mask])
        mae_p11_plus = mean_absolute_error(y_test[~top10_mask], y_pred[~top10_mask])

        mlflow.log_metrics({
            "mae": float(mae),
            "mae_top3": float(mae_top3),
            "mae_top10": float(mae_top10),
            "mae_p11_plus": float(mae_p11_plus),
            "baseline_mae": float(baseline_model),
            "best_iteration": float(best_iter),
        })

        mlflow.sklearn.log_model(
            model, 
            name=MODEL_NAME, 
            registered_model_name=MODEL_NAME,
            skops_trusted_types=["xgboost.sklearn.XGBRegressor", "xgboost.core.Booster"]
        )

        joblib.dump(model, LOCAL_MODELS_DIR / "qualifying_results" / f"{MODEL_NAME}.pkl")


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the training data."""

    from src.storage.local_storage_backend import LocalStorageBackend
    from src.config.paths import LOCAL_DATA_DIR

    storage_backend = LocalStorageBackend(LOCAL_DATA_DIR)
    df = storage_backend.read("gold/race_results")

    df = df.dropna(subset=FEATURE_COLS)

    return df

def split_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training data into training and test sets."""  # rounds 1–12 train, 16–24 test

    train_df = df[(df["season"] < 2025) | ((df["season"] == 2025) & (df["round"] <= HALF_2025_ROUND))]
    test_df  = df[(df["season"] == 2025) & (df["round"] > HALF_2025_ROUND)]

    return train_df, test_df