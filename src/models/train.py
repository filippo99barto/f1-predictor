import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from src.models.features import add_features
import joblib
from src.config.paths import LOCAL_MODELS_DIR


def load_training_data():
    """Load the training data."""
    from src.storage.local_storage_backend import LocalStorageBackend
    from src.config.paths import LOCAL_DATA_DIR

    storage_backend = LocalStorageBackend(LOCAL_DATA_DIR)
    df = storage_backend.read("gold")

    df, features = add_features(df)

    HALF_2024_ROUND = 12  # rounds 1–12 train, 13–24 test

    train = df[(df["season"] < 2024) | ((df["season"] == 2024) & (df["round"] <= HALF_2024_ROUND))]
    test  = df[(df["season"] == 2024) & (df["round"] > HALF_2024_ROUND)]

    return train, test, features

def train_model():
    """Train the model."""
    train, test, features = load_training_data()

    train = train.dropna(subset=features)
    test = test.dropna(subset=features)

    train = train[train["status"].isin(["Finished", "Lapped", "+1 Lap", "+2 Laps"])]
    test = test[test["status"].isin(["Finished", "Lapped", "+1 Lap", "+2 Laps"])]

    x_train= train[features]
    y_train = train["position"]

    x_test = test[features]
    y_test = test["position"]

    # Baseline finishing position = grid position
    baseline_model = mean_absolute_error(y_test, x_test["grid"])
    print(f"Baseline MAE (predict position = grid): {baseline_model:.2f}")


    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore",), [
            "driverId", 
            "constructorId", 
            "circuitName",
            "season",
            "round"
            ]),
        ("num", "passthrough", [
            "grid",
            "driver_last_race_position", 
            "driver_median_position_last_3_races",
            "constructor_median_position_last_3_races",
            "driver_circuit_median_position_last_3_races",
            "driver_season_median_position",
            "driver_positions_gained_season_median",
            "driver_positions_gained_career_median"]),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("reg", RandomForestRegressor(n_estimators=500, random_state=42)),
    ])

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(f"Model MAE: {mean_absolute_error(y_test, y_pred):.2f}")

    joblib.dump(model, LOCAL_MODELS_DIR / "model_v2.pkl")