import pandas as pd
from src.features.drivers import (
    add_driver_positions_gained_season_median, 
    add_driver_positions_gained_career_median, 
    add_driver_qualifying_career_median, 
    add_driver_previous_qualifying_position, 
    add_driver_previous_qualifying_gap_to_pole, 
    add_driver_qualifying_season_median, 
    add_driver_qualifying_circuit_median
    )
from src.features.constructors import (
    add_constructor_qualifying_season_median
    )

def build_qualifying_results_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the qualifying results features."""

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
