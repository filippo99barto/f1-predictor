import logging

import pandas as pd

logger = logging.getLogger(__name__)


def format_driver_id(driver_id: str) -> str:
    return driver_id.replace("_", " ").title()


def drop_incomplete_feature_rows(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    stage: str,
) -> pd.DataFrame:
    missing = df[feature_cols].isna()
    incomplete = missing.any(axis=1)
    dropped_df = df.loc[incomplete]

    if not dropped_df.empty:
        for _, row in dropped_df.iterrows():
            null_cols = missing.loc[row.name]
            null_features = null_cols[null_cols].index.tolist()
            logger.warning(
                "Dropped driver %s (%s) at %s stage: missing %s",
                row["driver_id"],
                row.get("constructor_id", "unknown"),
                stage,
                null_features,
            )

    return df.loc[~incomplete]
