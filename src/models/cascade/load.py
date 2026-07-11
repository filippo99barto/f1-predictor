import joblib
import mlflow.sklearn
from typing import Literal

from src.config.paths import LOCAL_MODELS_DIR


def load_model(model_name: str, model_subdir: str, mode: Literal["dev", "production"]):
    if mode == "dev":
        path = LOCAL_MODELS_DIR / model_subdir / f"{model_name}-dev.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Dev model not found at {path}. Run train_model(mode='dev') first."
            )
        return joblib.load(path)
    return mlflow.sklearn.load_model(f"models:/{model_name}/latest")
