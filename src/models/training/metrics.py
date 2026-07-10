import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def mae_slices(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    top10_mask = y_true <= 10
    top3_mask = y_true <= 3
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mae_top3": float(mean_absolute_error(y_true[top3_mask], y_pred[top3_mask])),
        "mae_top10": float(mean_absolute_error(y_true[top10_mask], y_pred[top10_mask])),
        "mae_p11_plus": float(mean_absolute_error(y_true[~top10_mask], y_pred[~top10_mask])),
    }


def baseline_mae(y_true: pd.Series, baseline: pd.Series) -> float:
    return float(mean_absolute_error(y_true, baseline))
