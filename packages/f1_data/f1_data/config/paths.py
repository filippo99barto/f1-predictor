import os
import subprocess
from pathlib import Path


def _find_repo_root() -> Path:
    env_root = os.environ.get("F1_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip()).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path(__file__).resolve().parents[4]


ROOT = _find_repo_root()
DATASETS_DIR = ROOT / "datasets"
LOCAL_DATA_DIR = DATASETS_DIR
