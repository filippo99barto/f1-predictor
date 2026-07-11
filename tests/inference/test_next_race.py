import pytest
import pandas as pd

from src.config.paths import LOCAL_MODELS_DIR
from src.inference.next_race import (
    RaceInfo,
    get_driver_lineup,
    load_race_schedule,
    resolve_target_race,
)
from src.models.qualifying_results.train import MODEL_NAME as QUALIFYING_MODEL_NAME
from src.models.race_results.train import MODEL_NAME as RACE_MODEL_NAME


def test_resolve_target_race_defaults_to_next_after_gold():
    race = resolve_target_race()
    assert race.season == 2026
    assert race.round == 10
    assert race.race_name == "Belgian Grand Prix"
    assert race.circuit_id == "spa"


def test_resolve_target_race_explicit_round():
    race = resolve_target_race(season=2026, round_num=10)
    assert race == RaceInfo(
        season=2026,
        round=10,
        race_name="Belgian Grand Prix",
        circuit_id="spa",
        date=race.date,
    )


def test_get_driver_lineup_returns_full_grid():
    lineup = get_driver_lineup(2026, 10)
    assert len(lineup) == 22
    assert "driverId" in lineup.columns
    assert "constructorId" in lineup.columns
    assert lineup["driverId"].is_unique


def test_load_race_schedule_filters_season():
    schedule = load_race_schedule(2026)
    assert schedule["season"].eq(2026).all()
    assert schedule["round"].max() >= 10


def _dev_models_available() -> bool:
    quali = LOCAL_MODELS_DIR / "qualifying_results" / f"{QUALIFYING_MODEL_NAME}-dev.pkl"
    race = LOCAL_MODELS_DIR / "race_results" / f"{RACE_MODEL_NAME}-dev.pkl"
    return quali.exists() and race.exists()


@pytest.mark.skipif(not _dev_models_available(), reason="Dev model artifacts not present")
def test_predict_next_race_smoke():
    from src.models.cascade.predict import predict_next_race

    result = predict_next_race(mode="dev")
    assert result.season == 2026
    assert result.round == 10
    assert len(result.predictions) >= 20
    assert result.winner in result.predictions["driverId"].values
    assert len(result.podium) == 3
