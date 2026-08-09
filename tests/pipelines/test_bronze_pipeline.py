from unittest.mock import patch

from src.pipelines.bronze.pipeline import BronzePipeline


def test_backfill_start_year_defaults_to_constant():
    assert BronzePipeline._backfill_start_year(None) == 2022


def test_backfill_start_year_custom():
    assert BronzePipeline._backfill_start_year(2024) == 2024


def test_resolve_years_incremental_default():
    with patch("src.pipelines.bronze.pipeline.datetime") as mock_dt:
        mock_dt.datetime.now.return_value.year = 2026
        years = BronzePipeline._resolve_years(backfill=False, start_backfill_year=None)
    assert years == [2026]


def test_resolve_years_backfill_uses_default_start():
    with patch("src.pipelines.bronze.pipeline.datetime") as mock_dt:
        mock_dt.datetime.now.return_value.year = 2026
        years = BronzePipeline._resolve_years(backfill=True, start_backfill_year=None)
    assert years == [2022, 2023, 2024, 2025, 2026]


def test_resolve_years_backfill_custom_start():
    with patch("src.pipelines.bronze.pipeline.datetime") as mock_dt:
        mock_dt.datetime.now.return_value.year = 2026
        years = BronzePipeline._resolve_years(backfill=True, start_backfill_year=2024)
    assert years == [2024, 2025, 2026]
