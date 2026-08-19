import math

import pandas as pd
import pytest
from tests.conftest import row_value

from f1_ml.models.common.features.drivers import (
    add_driver_circuit_median_career_position,
    add_driver_circuit_median_position_previous_3_races,
    add_driver_median_position_previous_3_races,
    add_driver_positions_gained_career_median,
    add_driver_positions_gained_season_median,
    add_driver_previous_qualifying_gap_to_pole,
    add_driver_previous_qualifying_position,
    add_driver_previous_race_position,
    add_driver_qualifying_career_median,
    add_driver_qualifying_circuit_median,
    add_driver_qualifying_season_median,
    add_driver_season_mediam_position,
)


def test_add_driver_previous_race_position(race_df: pd.DataFrame):
    result = add_driver_previous_race_position(race_df)

    assert row_value(result, "driver_a", 1, "driver_last_race_position") == 2
    assert row_value(result, "driver_a", 2, "driver_last_race_position") == 3
    assert row_value(result, "driver_a", 3, "driver_last_race_position") == 1
    assert row_value(result, "driver_b", 3, "driver_last_race_position") == 4


def test_add_driver_median_position_previous_3_races(race_df: pd.DataFrame):
    result = add_driver_median_position_previous_3_races(race_df)

    assert math.isnan(row_value(result, "driver_a", 1, "driver_median_position_last_3_races"))
    assert row_value(result, "driver_a", 2, "driver_median_position_last_3_races") == 3
    assert row_value(result, "driver_a", 3, "driver_median_position_last_3_races") == 2
    assert row_value(result, "driver_b", 3, "driver_median_position_last_3_races") == pytest.approx(
        2.5
    )


def test_add_driver_season_mediam_position(race_df: pd.DataFrame):
    result = add_driver_season_mediam_position(race_df)

    assert row_value(result, "driver_a", 1, "driver_season_median_position") == 2
    assert row_value(result, "driver_a", 2, "driver_season_median_position") == 3
    assert row_value(result, "driver_a", 3, "driver_season_median_position") == 2


def test_add_driver_circuit_median_position_previous_3_races(race_df: pd.DataFrame):
    result = add_driver_circuit_median_position_previous_3_races(race_df)

    assert row_value(result, "driver_a", 1, "driver_circuit_median_position_last_3_races") == 2
    assert row_value(result, "driver_a", 3, "driver_circuit_median_position_last_3_races") == 3
    assert row_value(result, "driver_b", 3, "driver_circuit_median_position_last_3_races") == 1


def test_add_driver_positions_gained_season_median(race_df: pd.DataFrame):
    result = add_driver_positions_gained_season_median(race_df)

    assert row_value(result, "driver_a", 1, "driver_positions_gained_season_median") == 0
    assert row_value(result, "driver_a", 2, "driver_positions_gained_season_median") == -1
    assert row_value(
        result, "driver_a", 3, "driver_positions_gained_season_median"
    ) == pytest.approx(-0.5)


def test_add_driver_positions_gained_career_median(race_df: pd.DataFrame):
    result = add_driver_positions_gained_career_median(race_df)

    assert row_value(result, "driver_a", 1, "driver_positions_gained_career_median") == 0
    assert row_value(result, "driver_a", 2, "driver_positions_gained_career_median") == -1
    assert row_value(result, "driver_c", 2, "driver_positions_gained_career_median") == 1


def test_add_driver_circuit_median_career_position(race_df: pd.DataFrame):
    result = add_driver_circuit_median_career_position(race_df)

    assert row_value(result, "driver_a", 1, "driver_circuit_median_career_position") == 2
    assert row_value(result, "driver_a", 3, "driver_circuit_median_career_position") == 3
    assert row_value(result, "driver_b", 2, "driver_circuit_median_career_position") == 3


def test_add_driver_qualifying_career_median(quali_df: pd.DataFrame):
    result = add_driver_qualifying_career_median(quali_df)

    assert math.isnan(row_value(result, "driver_a", 1, "driver_qualifying_career_median"))
    assert row_value(result, "driver_a", 2, "driver_qualifying_career_median") == 2
    assert row_value(result, "driver_a", 3, "driver_qualifying_career_median") == pytest.approx(1.5)


def test_add_driver_previous_qualifying_position(quali_df: pd.DataFrame):
    seeded = add_driver_qualifying_career_median(quali_df)
    result = add_driver_previous_qualifying_position(seeded)

    assert math.isnan(row_value(result, "driver_a", 1, "driver_last_qualifying_position"))
    assert row_value(result, "driver_a", 2, "driver_last_qualifying_position") == 2
    assert row_value(result, "driver_a", 3, "driver_last_qualifying_position") == 1


def test_add_driver_previous_qualifying_gap_to_pole(quali_df: pd.DataFrame):
    result = add_driver_previous_qualifying_gap_to_pole(quali_df)

    assert row_value(result, "driver_a", 1, "last_qualifying_gap_to_pole") == 0
    assert row_value(result, "driver_a", 2, "last_qualifying_gap_to_pole") == 3
    assert row_value(result, "driver_a", 3, "last_qualifying_gap_to_pole") == 0


def test_add_driver_qualifying_season_median(quali_df: pd.DataFrame):
    seeded = add_driver_qualifying_career_median(quali_df)
    result = add_driver_qualifying_season_median(seeded)

    assert math.isnan(row_value(result, "driver_a", 1, "driver_qualifying_season_median"))
    assert row_value(result, "driver_a", 2, "driver_qualifying_season_median") == 2
    assert row_value(result, "driver_a", 3, "driver_qualifying_season_median") == pytest.approx(1.5)


def test_add_driver_qualifying_circuit_median(quali_df: pd.DataFrame):
    seeded = add_driver_qualifying_career_median(quali_df)
    result = add_driver_qualifying_circuit_median(seeded)

    assert math.isnan(row_value(result, "driver_a", 1, "driver_qualifying_circuit_median"))
    assert row_value(result, "driver_a", 2, "driver_qualifying_circuit_median") == 2
    assert row_value(result, "driver_a", 3, "driver_qualifying_circuit_median") == 2
