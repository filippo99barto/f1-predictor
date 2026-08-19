import pandas as pd

from f1_ml.models.common.features.constructors import (
    add_constructor_qualifying_season_median,
)
from f1_ml.models.common.features.drivers import (
    add_driver_positions_gained_career_median,
    add_driver_positions_gained_season_median,
    add_driver_previous_qualifying_gap_to_pole,
    add_driver_previous_qualifying_position,
    add_driver_qualifying_career_median,
    add_driver_qualifying_circuit_median,
    add_driver_qualifying_season_median,
)


def build_qualifying_results_features(
    df: pd.DataFrame,
    *,
    for_inference: bool = False,
) -> pd.DataFrame:
    """Build the qualifying results features."""

    if not for_inference:
        df = df.dropna(subset=["qualifying_position"])

    df = add_driver_positions_gained_season_median(df)
    df = add_driver_positions_gained_career_median(df)
    df = add_driver_qualifying_career_median(df)
    df = add_driver_previous_qualifying_position(df)
    df = add_driver_previous_qualifying_gap_to_pole(df)
    df = add_driver_qualifying_season_median(df)
    df = add_driver_qualifying_circuit_median(df)

    df = add_constructor_qualifying_season_median(df)

    return df
