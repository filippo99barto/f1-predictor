from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


@dataclass
class ModelTrainConfig:
    experiment_name: str
    model_name: str
    gold_path: str
    model_subdir: str
    feature_cols: list[str]
    target_col: str
    baseline_col: str
    xgb_params: dict
    train_row_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None
