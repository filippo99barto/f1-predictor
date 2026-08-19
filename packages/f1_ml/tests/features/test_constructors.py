import math

import pandas as pd
import pytest

from f1_ml.models.common.features.constructors import (
    add_constructor_median_position_previous_3_races,
    add_constructor_median_season_position,
    add_constructor_qualifying_season_median,
)
from f1_ml.models.common.features.drivers import add_driver_qualifying_career_median


def team_value(df: pd.DataFrame, constructor_id: str, round_num: int, column: str):
    mask = (df["constructor_id"] == constructor_id) & (df["round"] == round_num)
    return df.loc[mask, column].iloc[0]


def test_add_constructor_median_position_previous_3_races(race_df: pd.DataFrame):
    result = add_constructor_median_position_previous_3_races(race_df)

    assert math.isnan(team_value(result, "red_bull", 1, "constructor_median_position_last_3_races"))
    assert team_value(result, "red_bull", 2, "constructor_median_position_last_3_races") == 2
    assert team_value(
        result, "red_bull", 3, "constructor_median_position_last_3_races"
    ) == pytest.approx(2.25)
    assert team_value(result, "ferrari", 2, "constructor_median_position_last_3_races") == 2


def test_add_constructor_median_season_position(race_df: pd.DataFrame):
    result = add_constructor_median_season_position(race_df)

    assert team_value(result, "red_bull", 1, "constructor_median_season_position") == pytest.approx(
        1.5
    )
    assert team_value(result, "red_bull", 2, "constructor_median_season_position") == 2
    assert team_value(result, "red_bull", 3, "constructor_median_season_position") == pytest.approx(
        2.25
    )
    assert team_value(result, "ferrari", 1, "constructor_median_season_position") == 3


def test_add_constructor_qualifying_season_median(quali_df: pd.DataFrame):
    seeded = add_driver_qualifying_career_median(quali_df)
    result = add_constructor_qualifying_season_median(seeded)

    assert math.isnan(team_value(result, "red_bull", 1, "constructor_qualifying_season_median"))
    assert team_value(
        result, "red_bull", 2, "constructor_qualifying_season_median"
    ) == pytest.approx(1.5)
    assert team_value(
        result, "red_bull", 3, "constructor_qualifying_season_median"
    ) == pytest.approx(1.75)
