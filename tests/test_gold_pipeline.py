import pandas as pd
import pytest

from src.ingestion.gold_pipeline import GoldPipeline
from tests.fake_storage_backend import FakeStorageBackend


@pytest.fixture
def silver_season_df():
    df = pd.DataFrame([
        {
            "season": 2024,
            "round": 1,
            "raceName": "Bahrain Grand Prix",
            "circuitName": "Saudi Arabian Grand Prix",
            "locality": "Jeddah",
            "country": "Saudi Arabia",
            "date": "2024-03-01",
            "time": "15:00:00Z",
            "position": 3,
            "points": 16.0,
            "grid": 2,
            "status": "Finished",
            "driverId": "max_verstappen",
            "constructorId": "red_bull",
        },
        {
            "season": 2024,
            "round": 2,
            "raceName": "Bahrain Grand Prix",
            "circuitName": "Bahrain International Circuit",
            "locality": "Sakhir",
            "country": "Bahrain",
            "date": "2024-03-02",
            "time": "15:00:00Z",
            "position": 1,
            "points": 26.0,
            "grid": 1,
            "status": "Finished",
            "driverId": "max_verstappen",
            "constructorId": "red_bull",
        },
        {
            "season": 2024,
            "round": 2,
            "raceName": "Bahrain Grand Prix",
            "circuitName": "Bahrain International Circuit",
            "locality": "Sakhir",
            "country": "Bahrain",
            "date": "2024-03-02",
            "time": "15:00:00Z",
            "position": 2,
            "points": 18.0,
            "grid": pd.NA,
            "status": "Finished",
            "driverId": "perez",
            "constructorId": "red_bull",
        },
    ])

    return df.astype({"grid": "Int64"})


@pytest.fixture
def gold_season_df():
    df = pd.DataFrame([
        {
            "season": 2024,
            "round": 1,
            "raceName": "Bahrain Grand Prix",
            "circuitName": "Saudi Arabian Grand Prix",
            "locality": "Jeddah",
            "country": "Saudi Arabia",
            "position": 3,
            "points": 16.0,
            "grid": 2,
            "status": "Finished",
            "driverId": "max_verstappen",
            "constructorId": "red_bull",
            "race_datetime": pd.to_datetime("2024-03-01 15:00:00"),
            "driver_last_race_position": float("nan"),
            "driver_median_position_last_3_races": float("nan"),
        },
        {
            "season": 2024,
            "round": 2,
            "raceName": "Bahrain Grand Prix",
            "circuitName": "Bahrain International Circuit",
            "locality": "Sakhir",
            "country": "Bahrain",
            "position": 1,
            "points": 26.0,
            "grid": 1,
            "status": "Finished",
            "driverId": "max_verstappen",
            "constructorId": "red_bull",
            "race_datetime": pd.to_datetime("2024-03-02 15:00:00"),
            "driver_last_race_position": 3.0,
            "driver_median_position_last_3_races": 3.0,
        },
        {
            "season": 2024,
            "round": 2,
            "raceName": "Bahrain Grand Prix",
            "circuitName": "Bahrain International Circuit",
            "locality": "Sakhir",
            "country": "Bahrain",
            "position": 2,
            "points": 18.0,
            "grid": pd.NA,
            "status": "Finished",
            "driverId": "perez",
            "constructorId": "red_bull",
            "race_datetime": pd.to_datetime("2024-03-02 15:00:00"),
            "driver_last_race_position": float("nan"),
            "driver_median_position_last_3_races": float("nan"),
        },
    ])

    return df.astype({
        "grid": "Int64", 

        "race_datetime": "datetime64[ns]",
        })


@pytest.fixture
def fake_storage(silver_season_df):
    return FakeStorageBackend(
        reads={
            "silver": silver_season_df,
        }
    )


def test_build_gold_produces_the_correct_data(fake_storage, gold_season_df):
    pipeline = GoldPipeline(fake_storage)
    gold = pipeline.build_gold_data()

    print(gold.dtypes.compare(gold_season_df.dtypes))
    assert gold.equals(gold_season_df)
