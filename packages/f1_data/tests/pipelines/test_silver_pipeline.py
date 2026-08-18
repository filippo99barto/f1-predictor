import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from f1_data.pipelines.silver.pipeline_race_results import SilverPipelineRaceResults
from f1_data.pipelines.silver.schemas import SilverSchemaRaceResults


def _race_row(
    *,
    season: int,
    round_num: int,
    driver_id: str,
    starting_position: int | str,
    position: int = 10,
) -> dict:
    return {
        "season": season,
        "round": round_num,
        "circuitId": "test_circuit",
        "position": position,
        "points": 0.0,
        "starting_position": starting_position,
        "status": "Finished",
        "driverId": driver_id,
        "constructorId": "test_team",
    }


def test_pit_lane_start_maps_to_field_size():
    rows = [
        _race_row(season=2022, round_num=4, driver_id=f"driver_{i}", starting_position=i)
        for i in range(1, 20)
    ]
    rows.append(
        _race_row(season=2022, round_num=4, driver_id="zhou", starting_position=0, position=15)
    )
    df = pd.DataFrame(rows)

    result = SilverPipelineRaceResults._resolve_starting_positions_pitlane_starts(df)

    assert result[df["driverId"] == "zhou"].iloc[0] == 20
    assert result.notna().all()


def test_pit_lane_start_accepts_string_grid_from_bronze():
    rows = [
        _race_row(season=2022, round_num=4, driver_id=f"driver_{i}", starting_position=str(i))
        for i in range(1, 20)
    ]
    rows.append(
        _race_row(season=2022, round_num=4, driver_id="zhou", starting_position="0", position=15)
    )
    df = pd.DataFrame(rows)

    result = SilverPipelineRaceResults._resolve_starting_positions_pitlane_starts(df)

    assert result[df["driverId"] == "zhou"].iloc[0] == 20


def test_multiple_pit_lane_starts_share_last_grid_slot():
    rows = [
        _race_row(season=2022, round_num=5, driver_id=f"driver_{i}", starting_position=i)
        for i in range(1, 19)
    ]
    rows.append(
        _race_row(season=2022, round_num=5, driver_id="stroll", starting_position=0, position=10)
    )
    rows.append(
        _race_row(season=2022, round_num=5, driver_id="vettel", starting_position=0, position=17)
    )
    df = pd.DataFrame(rows)

    result = SilverPipelineRaceResults._resolve_starting_positions_pitlane_starts(df)
    pit_starts = result[df["driverId"].isin(["stroll", "vettel"])]

    assert len(df) == 20
    assert pit_starts.tolist() == [20, 20]


def test_normal_grid_positions_unchanged():
    df = pd.DataFrame(
        [
            _race_row(
                season=2024,
                round_num=1,
                driver_id="max_verstappen",
                starting_position=1,
                position=1,
            ),
            _race_row(season=2024, round_num=1, driver_id="perez", starting_position=5, position=2),
        ]
    )

    result = SilverPipelineRaceResults._resolve_starting_positions_pitlane_starts(df)

    assert result[df["driverId"] == "max_verstappen"].iloc[0] == 1
    assert result[df["driverId"] == "perez"].iloc[0] == 5


def test_silver_schema_rejects_null_starting_position():
    df = pd.DataFrame(
        [
            {
                "season": 2024,
                "round": 1,
                "circuitId": "test_circuit",
                "position": 1,
                "points": 25.0,
                "starting_position": pd.NA,
                "status": "Finished",
                "driverId": "driver_a",
                "constructorId": "test_team",
            }
        ]
    )

    with pytest.raises((SchemaError, SchemaErrors), match="starting_position"):
        SilverSchemaRaceResults.validate(df)
