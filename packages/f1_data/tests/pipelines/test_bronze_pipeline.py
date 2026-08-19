from unittest.mock import patch

import pandas as pd

from f1_data.config.constants import BRONZE_CONSTRUCTORS_COLUMNS, BRONZE_RACES_COLUMNS
from f1_data.pipelines.bronze.pipeline import BronzePipeline


def test_backfill_start_year_defaults_to_constant():
    assert BronzePipeline._backfill_start_year(None) == 2022


def test_backfill_start_year_custom():
    assert BronzePipeline._backfill_start_year(2024) == 2024


def test_resolve_years_incremental_default():
    with patch("f1_data.pipelines.bronze.pipeline.datetime") as mock_dt:
        mock_dt.datetime.now.return_value.year = 2026
        years = BronzePipeline._resolve_years(backfill=False, start_backfill_year=None)
    assert years == [2026]


def test_resolve_years_backfill_uses_default_start():
    with patch("f1_data.pipelines.bronze.pipeline.datetime") as mock_dt:
        mock_dt.datetime.now.return_value.year = 2026
        years = BronzePipeline._resolve_years(backfill=True, start_backfill_year=None)
    assert years == [2022, 2023, 2024, 2025, 2026]


def test_resolve_years_backfill_custom_start():
    with patch("f1_data.pipelines.bronze.pipeline.datetime") as mock_dt:
        mock_dt.datetime.now.return_value.year = 2026
        years = BronzePipeline._resolve_years(backfill=True, start_backfill_year=2024)
    assert years == [2024, 2025, 2026]


def test_select_bronze_columns_casts_season_and_round_to_int():
    df = pd.DataFrame(
        {
            "season": ["2026"],
            "round": ["1"],
            "raceName": ["Australian Grand Prix"],
            "Circuit.circuitId": ["albert_park"],
            "Circuit.Location.locality": ["Melbourne"],
            "Circuit.Location.country": ["Australia"],
            "date": ["2026-03-08"],
            "time": ["04:00:00Z"],
        }
    )
    out = BronzePipeline._select_bronze_columns(df, BRONZE_RACES_COLUMNS)
    assert out["season"].dtype == int
    assert out["round"].dtype == int
    assert out["season"].iloc[0] == 2026
    assert out["round"].iloc[0] == 1


def test_select_bronze_columns_casts_season_when_round_absent():
    df = pd.DataFrame(
        {
            "season": ["2026"],
            "constructorId": ["ferrari"],
            "name": ["Ferrari"],
            "nationality": ["Italian"],
        }
    )
    out = BronzePipeline._select_bronze_columns(df, BRONZE_CONSTRUCTORS_COLUMNS)
    assert out["season"].dtype == int
    assert out["season"].iloc[0] == 2026
    assert "round" not in out.columns
