import pandas as pd
import pytest


@pytest.fixture
def race_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driver_id": "driver_a",
                "constructor_id": "red_bull",
                "circuit_id": "bahrain",
                "season": 2024,
                "round": 1,
                "position": 3,
                "starting_position": 2,
                "qualifying_position": 2,
            },
            {
                "driver_id": "driver_a",
                "constructor_id": "red_bull",
                "circuit_id": "jeddah",
                "season": 2024,
                "round": 2,
                "position": 1,
                "starting_position": 1,
                "qualifying_position": 1,
            },
            {
                "driver_id": "driver_a",
                "constructor_id": "red_bull",
                "circuit_id": "bahrain",
                "season": 2024,
                "round": 3,
                "position": 2,
                "starting_position": 3,
                "qualifying_position": 3,
            },
            {
                "driver_id": "driver_b",
                "constructor_id": "red_bull",
                "circuit_id": "bahrain",
                "season": 2024,
                "round": 1,
                "position": 1,
                "starting_position": 1,
                "qualifying_position": 1,
            },
            {
                "driver_id": "driver_b",
                "constructor_id": "red_bull",
                "circuit_id": "jeddah",
                "season": 2024,
                "round": 2,
                "position": 4,
                "starting_position": 5,
                "qualifying_position": 3,
            },
            {
                "driver_id": "driver_b",
                "constructor_id": "red_bull",
                "circuit_id": "bahrain",
                "season": 2024,
                "round": 3,
                "position": 5,
                "starting_position": 4,
                "qualifying_position": 4,
            },
            {
                "driver_id": "driver_c",
                "constructor_id": "ferrari",
                "circuit_id": "bahrain",
                "season": 2024,
                "round": 1,
                "position": 2,
                "starting_position": 3,
                "qualifying_position": 3,
            },
            {
                "driver_id": "driver_c",
                "constructor_id": "ferrari",
                "circuit_id": "jeddah",
                "season": 2024,
                "round": 2,
                "position": 3,
                "starting_position": 2,
                "qualifying_position": 2,
            },
        ]
    )


@pytest.fixture
def quali_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driver_id": "driver_a",
                "constructor_id": "red_bull",
                "circuit_id": "bahrain",
                "season": 2024,
                "round": 1,
                "qualifying_position": 2,
                "q1_seconds": 90.0,
                "q2_seconds": 89.0,
                "q3_seconds": 88.0,
                "position": 3,
                "starting_position": 2,
            },
            {
                "driver_id": "driver_a",
                "constructor_id": "red_bull",
                "circuit_id": "jeddah",
                "season": 2024,
                "round": 2,
                "qualifying_position": 1,
                "q1_seconds": 91.0,
                "q2_seconds": 90.0,
                "q3_seconds": 89.0,
                "position": 1,
                "starting_position": 1,
            },
            {
                "driver_id": "driver_a",
                "constructor_id": "red_bull",
                "circuit_id": "bahrain",
                "season": 2024,
                "round": 3,
                "qualifying_position": 3,
                "q1_seconds": 92.0,
                "q2_seconds": 91.0,
                "q3_seconds": 90.0,
                "position": 2,
                "starting_position": 3,
            },
            {
                "driver_id": "driver_b",
                "constructor_id": "red_bull",
                "circuit_id": "bahrain",
                "season": 2024,
                "round": 1,
                "qualifying_position": 1,
                "q1_seconds": 87.0,
                "q2_seconds": 86.0,
                "q3_seconds": 85.0,
                "position": 1,
                "starting_position": 1,
            },
            {
                "driver_id": "driver_b",
                "constructor_id": "red_bull",
                "circuit_id": "jeddah",
                "season": 2024,
                "round": 2,
                "qualifying_position": 3,
                "q1_seconds": 92.0,
                "q2_seconds": 91.0,
                "q3_seconds": 90.0,
                "position": 4,
                "starting_position": 5,
            },
        ]
    )


def row_value(df: pd.DataFrame, driver_id: str, round_num: int, column: str):
    mask = (df["driver_id"] == driver_id) & (df["round"] == round_num)
    return df.loc[mask, column].iloc[0]
