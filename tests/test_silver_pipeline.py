import pandas as pd
import pytest

from src.ingestion.silver_pipeline import SilverPipeline
from tests.fake_storage_backend import FakeStorageBackend

@pytest.fixture
def bronze_races_df():
    return pd.DataFrame([
        {
            "season": "2024",
            "round": "1",
            "raceName": "Bahrain Grand Prix",
            "Circuit.circuitName": "Bahrain International Circuit",
            "date": "2024-03-02",
            "time": "15:00:00Z",
        }
    ])

@pytest.fixture
def bronze_results_df():
    return pd.DataFrame([
        {
            "season": "2024", 
            "round": "1", 
            "position": "1", 
            "points": "26",
            "grid": "1", 
            "status": "Finished",
            "Driver.driverId": "max_verstappen",
            "Constructor.constructorId": "red_bull",
        },
        {
            "season": "2024", 
            "round": "1", 
            "position": "2", 
            "points": "18",
            "grid": "5", 
            "status": "Finished",
            "Driver.driverId": "perez",
            "Constructor.constructorId": "red_bull",
        },
    ])

@pytest.fixture
def silver_season_df():
    return pd.DataFrame([
        {
            "season": "2024",
            "round": "1",
            "raceName": "Bahrain Grand Prix",
            "Circuit.circuitName": "Bahrain International Circuit",
            "date": "2024-03-02",
            "time": "15:00:00Z",
            "position": "1",
            "points": "26",
            "grid": "1",
            "status": "Finished",
            "Driver.driverId": "max_verstappen",
            "Constructor.constructorId": "red_bull",
        },
        {
            "season": "2024",
            "round": "1",
            "raceName": "Bahrain Grand Prix",
            "Circuit.circuitName": "Bahrain International Circuit",
            "date": "2024-03-02",
            "time": "15:00:00Z",
            "position": "2",
            "points": "18",
            "grid": "5",
            "status": "Finished",
            "Driver.driverId": "perez",
            "Constructor.constructorId": "red_bull",
        }
    ])

@pytest.fixture
def fake_storage(bronze_races_df, bronze_results_df):
    from tests.fake_storage_backend import FakeStorageBackend 

    return FakeStorageBackend(
        reads={
            "bronze/races": bronze_races_df,
            "bronze/results": bronze_results_df,
        }
    )

def test_build_silver_merges_races_and_results(fake_storage, silver_season_df):
    pipeline = SilverPipeline(fake_storage)
    silver = pipeline.build_silver_data()
    
    assert silver.equals(silver_season_df)

