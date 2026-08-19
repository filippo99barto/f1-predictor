from types import SimpleNamespace
from unittest.mock import patch

from f1_agent.tools import get_next_race_info, predict_next_qualifying, predict_next_race


def test_get_next_race_info_returns_schedule():
    race = SimpleNamespace(
        season=2026,
        round=14,
        race_name="Italian Grand Prix",
        circuit_id="monza",
        date=SimpleNamespace(isoformat=lambda: "2026-09-06T00:00:00"),
    )
    with patch("f1_agent.tools.resolve_target_race", return_value=race):
        assert get_next_race_info.invoke({}) == {
            "season": 2026,
            "round": 14,
            "race_name": "Italian Grand Prix",
            "circuit_id": "monza",
            "date": "2026-09-06",
        }


def test_get_next_race_info_returns_error_on_failure():
    with patch("f1_agent.tools.resolve_target_race", side_effect=ValueError("no race")):
        assert get_next_race_info.invoke({"season": 2099}) == {"error": "no race"}


def test_predict_next_qualifying_returns_grid():
    result = SimpleNamespace(
        to_dict=lambda top_n=None: {"pole": "VER", "n_drivers": 22, "top_n": top_n},
    )
    with patch("f1_agent.tools.run_predict_next_qualifying", return_value=result):
        assert predict_next_qualifying.invoke({"top_n": 3}) == {
            "pole": "VER",
            "n_drivers": 22,
            "top_n": 3,
        }


def test_predict_next_qualifying_returns_error_on_failure():
    with patch(
        "f1_agent.tools.run_predict_next_qualifying",
        side_effect=FileNotFoundError("missing model"),
    ):
        assert predict_next_qualifying.invoke({}) == {"error": "missing model"}


def test_predict_next_race_returns_results():
    result = SimpleNamespace(
        to_dict=lambda top_n=None: {"winner": "VER", "n_drivers": 22, "top_n": top_n},
    )
    with patch("f1_agent.tools.run_predict_next_race", return_value=result):
        assert predict_next_race.invoke({"top_n": 3}) == {
            "winner": "VER",
            "n_drivers": 22,
            "top_n": 3,
        }


def test_predict_next_race_returns_error_on_failure():
    with patch(
        "f1_agent.tools.run_predict_next_race",
        side_effect=ValueError("no drivers"),
    ):
        assert predict_next_race.invoke({}) == {"error": "no drivers"}
