import os

from f1_data.config.paths import ROOT

LOCAL_MODELS_DIR = ROOT / "packages" / "f1_ml" / "f1_ml" / "models"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
