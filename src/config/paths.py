from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MLFLOW_RUNS_DIR = ROOT / "mlruns"
LOCAL_DATA_DIR = ROOT / "data"
LOCAL_MODELS_DIR = ROOT / "src" / "models"
