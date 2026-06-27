import pandera.pandas as pa
import pandas as pd
import pytest

from src.pipelines.silver.pipeline import SilverPipeline
from tests.helpers.fake_storage_backend import FakeStorageBackend

@pytest.fixture
def bronze_races_df():
    return pd.DataFrame([
        {
            "season": 2024,
            "round": 1,
            "raceName": "Bahrain Grand Prix",
            "Circuit.circuitName": "Bahrain International Circuit",
            "Circuit.Location.locality": "Sakhir",
            "Circuit.Location.country": "Bahrain",
            "date": "2024-03-02",
            "time": "15:00:00Z",
        }
    ])


@pytest.fixture
def bronze_results_df():
    return pd.DataFrame([
        {
            "season": 2024,
            "round": 1,
            "position": 1,
            "points": 26.0,
            "grid": 1,
            "status": "Finished",
            "Driver.driverId": "max_verstappen",
            "Constructor.constructorId": "red_bull",
        },
        {
            "season": 2024,
            "round": 1,
            "position": 2,
            "points": 18.0,
            "grid": 0,
            "status": "Finished",
            "Driver.driverId": "perez",
            "Constructor.constructorId": "red_bull",
        },
    ])


@pytest.fixture
def silver_season_df():
    df = pd.DataFrame([
        {
            "season": 2024,
            "round": 1,
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
            "round": 1,
            "raceName": "Bahrain Grand Prix",
            "circuitName": "Bahrain International Circuit",
            "locality": "Sakhir",
            "country": "Bahrain",
            "date": "2024-03-02",
            "time": "15:00:00Z",
            "position":2,
            "points": 18.0,
            "grid": pd.NA,  
            "status": "Finished",
            "driverId": "perez",
            "constructorId": "red_bull",
        }
    ])

    return df.astype({ "grid": "Int64" })


@pytest.fixture
def fake_storage(bronze_races_df, bronze_results_df):
    return FakeStorageBackend(
        reads={
            "bronze/races": bronze_races_df,
            "bronze/results": bronze_results_df,
        }
    )


def test_build_silver_produces_the_correct_data(fake_storage, silver_season_df):
    pipeline = SilverPipeline(fake_storage)
    silver = pipeline.build_silver_data()

    assert silver.equals(silver_season_df)
