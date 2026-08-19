from unittest.mock import patch

import numpy as np
import pandas as pd

from f1_ml.inference.next_race import (
    RaceInfo,
    get_driver_lineup,
    load_race_schedule,
    resolve_target_race,
)

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
            "race_name": "Hungarian Grand Prix",
            "circuit_id": "hungaroring",
            "date": "2026-07-27",
        },
        {
            "season": 2026,
            "round": 10,
            "race_name": "Belgian Grand Prix",
            "circuit_id": "spa",
            "date": "2026-08-03",
        },
        {
            "season": 2026,
            "round": 11,
            "race_name": "Dutch Grand Prix",
            "circuit_id": "zandvoort",
            "date": "2026-08-17",
        },
    ]
)

FAKE_SILVER_RACE_RESULTS = pd.DataFrame(
    [
        {
            "season": 2026,
            "round": 9,
            "driver_id": f"driver_{i:02d}",
            "constructor_id": f"team_{i % 11}",
            "circuit_id": "hungaroring",
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
                    "driver_id": f"driver_{i:02d}",
                    "constructor_id": f"team_{i % 11}",
                    "circuit_id": "hungaroring" if race_round % 2 else "monaco",
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
                "driver_id": f"driver_{i:02d}",
                "constructor_id": f"team_{i % 11}",
                "circuit_id": circuit_id,
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
    assert "driver_id" in lineup.columns
    assert "constructor_id" in lineup.columns
    assert lineup["driver_id"].is_unique


@patch("f1_ml.inference.next_race._storage")
def test_load_race_schedule_filters_season(mock_storage):
    mock_storage.return_value.read.return_value = FAKE_SCHEDULE

    schedule = load_race_schedule(2026)

    assert schedule["season"].eq(2026).all()
    assert schedule["round"].max() == 11


class _FakeModel:
    def predict(self, X):
        return np.arange(1, len(X) + 1, dtype=float)


@patch("f1_ml.models.race.predict.load_model", return_value=_FakeModel())
@patch("f1_ml.models.qualifying.predict.load_model", return_value=_FakeModel())
@patch("f1_ml.models.race.predict.build_inference_frames")
@patch("f1_ml.models.race.predict.resolve_target_race")
def test_predict_next_race_smoke(mock_resolve, mock_frames, _mock_qualy_load, _mock_race_load):
    from f1_ml.models.race.predict import predict_next_race

    mock_resolve.return_value = TARGET_RACE
    mock_frames.side_effect = _fake_inference_frames
    result = predict_next_race()
    payload = result.to_dict(top_n=3)

    assert result.season == 2026
    assert result.round == 10
    assert result.race_name == "Belgian Grand Prix"
    assert result.circuit_id == "spa"
    assert len(result.predictions) >= 20
    assert result.winner in result.predictions["driver_id"].values
    assert len(result.podium) == 3
    assert payload["winner"]["driver_id"] == result.winner
    assert len(payload["predictions"]) == 3


@patch("f1_ml.models.qualifying.predict.load_model", return_value=_FakeModel())
@patch("f1_ml.models.qualifying.predict.build_inference_frames")
@patch("f1_ml.models.qualifying.predict.resolve_target_race")
def test_predict_next_qualifying_smoke(mock_resolve, mock_frames, _mock_load):
    from f1_ml.models.qualifying.predict import predict_next_qualifying

    mock_resolve.return_value = TARGET_RACE
    mock_frames.side_effect = _fake_inference_frames
    result = predict_next_qualifying()
    payload = result.to_dict(top_n=3)

    assert result.season == 2026
    assert result.round == 10
    assert result.pole in result.predictions["driver_id"].values
    assert "predicted_qualifying_position" in result.predictions.columns
    assert "predicted_race_position" not in result.predictions.columns
    assert "constructor_id" in result.predictions.columns
    assert payload["pole"]["driver_id"] == result.pole
    assert len(payload["predictions"]) == 3
