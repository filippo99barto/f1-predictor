import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from src.config.paths import LOCAL_MODELS_DIR, MLFLOW_RUNS_DIR
import mlflow
import mlflow.sklearn

def train_model() -> None:
    """Train the model."""

    experiment_name = "f1-race-results-predictor"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=(MLFLOW_RUNS_DIR / experiment_name).as_uri()
        )

    mlflow.set_experiment(experiment_name)

    df = load_training_data()

    df = df[df["status"].isin(["Finished", "Lapped", "+1 Lap", "+2 Laps"])]

    features = df.columns.drop("position").tolist()

    train_df, test_df = split_training_data(df)

    x_train = train_df[features]
    y_train = train_df["position"]

    x_test = test_df[features]
    y_test = test_df["position"]

    baseline_model = mean_absolute_error(y_test, x_test["grid"])
    print(f"Baseline MAE (predict position = grid): {baseline_model:.2f}")

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [
            "driverId", 
            "constructorId",
            "circuitId"
            ]),
        # ("num", "passthrough", [
        #     "season",
        #     "round",
        #     "grid",
        #     "driver_last_race_position", 
        #     "driver_median_position_last_3_races",
        #     "constructor_median_position_last_3_races",
        #     "driver_circuit_median_position_last_3_races",
        #     "driver_season_median_position",
        #     "driver_positions_gained_season_median",
        #     "driver_positions_gained_career_median"]),
    ])

    HGB_PARAMS = {
        "max_iter": 500,        # like n_estimators for RF
        "learning_rate": 0.1,
        "max_depth": None,
        "random_state": 42,
    }

    model = Pipeline([
        ("prep", preprocessor),
        ("reg", HistGradientBoostingRegressor(**HGB_PARAMS)),
    ])

    with mlflow.start_run(run_name=f"race-results-predictor-v1"):
        # Log hyperparameters
        mlflow.log_param("max_iter", HGB_PARAMS["max_iter"])
        mlflow.log_param("learning_rate", HGB_PARAMS["learning_rate"])
        mlflow.log_param("model_type", "HistGradientBoostingRegressor")
        mlflow.log_param("version", "v1")
        mlflow.log_param("train_split", "seasons < 2025 + 2025 rounds 1-12")

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("baseline_mae", baseline_model)

        # Log the sklearn Pipeline (preprocessor + regressor)
        mlflow.sklearn.log_model(model, name=f"f1_position_predictor_")
        # Optional: keep joblib dump too during migration
        joblib.dump(model, LOCAL_MODELS_DIR / "race_results" / f"model.pkl")

        print(f"Model MAE: {mean_absolute_error(y_test, y_pred):.2f}")


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the training data."""

    from src.storage.local_storage_backend import LocalStorageBackend
    from src.config.paths import LOCAL_DATA_DIR

    storage_backend = LocalStorageBackend(LOCAL_DATA_DIR)
    
    df = storage_backend.read("gold/race_results")

    return df

def split_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training data into training and test sets."""

    HALF_2025_ROUND = 12  # rounds 1–12 train, 13–24 test

    train_df = df[(df["season"] < 2025) | ((df["season"] == 2025) & (df["round"] <= HALF_2025_ROUND))]
    test_df  = df[(df["season"] == 2025) & (df["round"] > HALF_2025_ROUND)]

    return train_df, test_df