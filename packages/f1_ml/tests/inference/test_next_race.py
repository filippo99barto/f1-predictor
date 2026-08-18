from unittest.mock import patch

import pandas as pd
import pytest

from f1_ml.config.paths import LOCAL_MODELS_DIR
from f1_ml.inference.next_race import (
    RaceInfo,
    get_driver_lineup,
    load_race_schedule,
    resolve_target_race,
)
from f1_ml.models.qualifying_results.train import MODEL_NAME as QUALIFYING_MODEL_NAME
from f1_ml.models.race_results.train import MODEL_NAME as RACE_MODEL_NAME

TARGET_RACE = RaceInfo(
    season=2026,
    round=10,
    race_name="Belgian Grand Prix",
    circuit_id="spa",
    date="2026-08-03",
)

FAKE_SCHEDULE = pd.DataFrame(
    [
        {
            "season": 2026,
            "round": 9,
            "raceName": "Hungarian Grand Prix",
            "circuitId": "hungaroring",
            "date": "2026-07-27",
        },
        {
            "season": 2026,
            "round": 10,
            "raceName": "Belgian Grand Prix",
            "circuitId": "spa",
            "date": "2026-08-03",
        },
        {
            "season": 2026,
            "round": 11,
            "raceName": "Dutch Grand Prix",
            "circuitId": "zandvoort",
            "date": "2026-08-17",
        },
    ]
)

FAKE_SILVER_RACE_RESULTS = pd.DataFrame(
    [
        {
            "season": 2026,
            "round": 9,
            "driverId": f"driver_{i:02d}",
            "constructorId": f"team_{i % 11}",
            "circuitId": "hungaroring",
            "position": i + 1,
        }
        for i in range(22)
    ]
)


def _fake_inference_frames(season: int, round_num: int, circuit_id: str) -> pd.DataFrame:
    rows = []
    for race_round in range(1, round_num):
        for i in range(22):
            position = float((i % 20) + 1)
            rows.append(
                {
                    "season": season,
                    "round": race_round,
                    "driverId": f"driver_{i:02d}",
                    "constructorId": f"team_{i % 11}",
                    "circuitId": "hungaroring" if race_round % 2 else "monaco",
                    "position": position,
                    "starting_position": position,
                    "qualifying_position": position,
                    "q1_seconds": 90.0 + i * 0.1,
                    "q2_seconds": 89.0 + i * 0.1,
                    "q3_seconds": 88.0 + i * 0.1,
                    "points": 10.0,
                    "status": "Finished",
                }
            )

    for i in range(22):
        rows.append(
            {
                "season": season,
                "round": round_num,
                "driverId": f"driver_{i:02d}",
                "constructorId": f"team_{i % 11}",
                "circuitId": circuit_id,
                "position": float("nan"),
                "starting_position": float("nan"),
                "qualifying_position": float("nan"),
                "q1_seconds": float("nan"),
                "q2_seconds": float("nan"),
                "q3_seconds": float("nan"),
                "points": float("nan"),
                "status": pd.NA,
            }
        )

    return pd.DataFrame(rows)


@patch("f1_ml.inference.next_race.load_race_schedule")
@patch("f1_ml.inference.next_race.get_last_completed_race_from_gold")
def test_resolve_target_race_defaults_to_next_after_gold(
    mock_last_race,
    mock_schedule,
):
    mock_last_race.return_value = (2026, 9)
    mock_schedule.return_value = FAKE_SCHEDULE

    race = resolve_target_race()

    assert race == TARGET_RACE


@patch("f1_ml.inference.next_race.load_race_schedule")
def test_resolve_target_race_explicit_round(mock_schedule):
    mock_schedule.return_value = FAKE_SCHEDULE

    race = resolve_target_race(season=2026, round_num=10)

    assert race == TARGET_RACE


@patch("f1_ml.inference.next_race._storage")
def test_get_driver_lineup_returns_full_grid(mock_storage):
    mock_storage.return_value.read.return_value = FAKE_SILVER_RACE_RESULTS

    lineup = get_driver_lineup(2026, 10)

    assert len(lineup) == 22
    assert "driverId" in lineup.columns
    assert "constructorId" in lineup.columns
    assert lineup["driverId"].is_unique


@patch("f1_ml.inference.next_race._storage")
def test_load_race_schedule_filters_season(mock_storage):
    mock_storage.return_value.read.return_value = FAKE_SCHEDULE

    schedule = load_race_schedule(2026)

    assert schedule["season"].eq(2026).all()
    assert schedule["round"].max() == 11


def _dev_models_available() -> bool:
    quali = LOCAL_MODELS_DIR / "qualifying_results" / f"{QUALIFYING_MODEL_NAME}-dev.pkl"
    race = LOCAL_MODELS_DIR / "race_results" / f"{RACE_MODEL_NAME}-dev.pkl"
    return quali.exists() and race.exists()


@pytest.mark.skipif(not _dev_models_available(), reason="Dev model artifacts not present")
@patch("f1_ml.models.cascade.predict.build_inference_frames")
@patch("f1_ml.models.cascade.predict.resolve_target_race")
def test_predict_next_race_smoke(mock_resolve, mock_frames):
    from f1_ml.models.cascade.predict import predict_next_race

    mock_resolve.return_value = TARGET_RACE
    mock_frames.side_effect = _fake_inference_frames
    result = predict_next_race(mode="dev")

    assert result.season == 2026
    assert result.round == 10
    assert result.race_name == "Belgian Grand Prix"
    assert result.circuit_id == "spa"
    assert len(result.predictions) >= 20
    assert result.winner in result.predictions["driverId"].values
    assert len(result.podium) == 3
