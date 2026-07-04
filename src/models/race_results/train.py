import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
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
    df = df.dropna(subset=features)

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
        ("num", "passthrough", [
            "season",
            "round",
            "grid",
            "driver_last_race_position", 
            "driver_median_position_last_3_races",
            "constructor_median_position_last_3_races",
            "driver_circuit_median_position_last_3_races",
            "driver_season_median_position",
            "driver_positions_gained_season_median",
            "driver_positions_gained_career_median"]),
    ])

    HGB_PARAMS = {
        "max_iter": 400,        # like n_estimators for RF
        "learning_rate": 0.05,
        "max_depth": None,
        "random_state": 42,
        "min_samples_leaf": 50,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
    }

    model = Pipeline([
        ("prep", preprocessor),
        ("reg", HistGradientBoostingRegressor(**HGB_PARAMS)),
    ])

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("max_iter", HGB_PARAMS["max_iter"])
        mlflow.log_param("learning_rate", HGB_PARAMS["learning_rate"])
        mlflow.log_param("max_depth", HGB_PARAMS["max_depth"])
        mlflow.log_param("min_samples_leaf", HGB_PARAMS["min_samples_leaf"])
        mlflow.log_param("early_stopping", HGB_PARAMS["early_stopping"])
        mlflow.log_param("validation_fraction", HGB_PARAMS["validation_fraction"])
        mlflow.log_param("n_iter_no_change", HGB_PARAMS["n_iter_no_change"])
        mlflow.log_param("model_type", "HistGradientBoostingRegressor")
        mlflow.log_param("train_split", "seasons < 2025 + 2025 rounds 1-15")
        mlflow.log_param("status", "Finished, Lapped, +1 Lap, +2 Laps")
        mlflow.log_param("dropped_na_rows", "true")
        mlflow.log_param("dropped_na_features", "grid + rolling features")

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        n_iter = model.named_steps["reg"].n_iter_          # how many trees were actually used

        mae = mean_absolute_error(y_test, y_pred)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("baseline_mae", baseline_model)
        mlflow.log_metric("n_iter", n_iter)


        # Log the sklearn Pipeline (preprocessor + regressor)
        mlflow.sklearn.log_model(model, name=f"f1_position_predictor")
        # Optional: keep joblib dump too during migration
        joblib.dump(model, LOCAL_MODELS_DIR / "race_results" / f"model.pkl")

        print(f"Number of trees used: {n_iter}")
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

    HALF_2025_ROUND = 15  # rounds 1–15 train, 16–24 test

    train_df = df[(df["season"] < 2025) | ((df["season"] == 2025) & (df["round"] <= HALF_2025_ROUND))]
    test_df  = df[(df["season"] == 2025) & (df["round"] > HALF_2025_ROUND)]

    return train_df, test_df