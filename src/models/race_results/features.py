import pandas as pd
from src.features.drivers import (
    add_driver_previous_race_position,
    add_driver_median_position_previous_3_races,
    add_driver_circuit_median_position_previous_3_races,
    add_driver_circuit_median_career_position,
    add_driver_season_mediam_position,
    add_driver_positions_gained_season_median,
    add_driver_positions_gained_career_median,
    add_driver_qualifying_career_median,
    add_driver_qualifying_season_median,
)
from src.features.constructors import (
    add_constructor_median_season_position,
    add_constructor_median_position_previous_3_races,
)


def build_race_results_features(
    df: pd.DataFrame,
    *,
    for_inference: bool = False,
) -> pd.DataFrame:
    """Build the race results features."""

    if not for_inference:
        df = df.dropna(subset=["qualifying_position"])

    df = add_driver_previous_race_position(df)
    df = add_driver_median_position_previous_3_races(df)
    df = add_driver_circuit_median_position_previous_3_races(df)
    df = add_driver_circuit_median_career_position(df)
    df = add_driver_season_mediam_position(df)
    df = add_driver_positions_gained_season_median(df)
    df = add_driver_positions_gained_career_median(df)
    df = add_driver_qualifying_career_median(df)
    df = add_driver_qualifying_season_median(df)

    df = add_constructor_median_season_position(df)
    df = add_constructor_median_position_previous_3_races(df)

    return df
