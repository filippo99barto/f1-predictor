import pytest

from src.assistant.client import ask


def test_ask_raises_without_gemini_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        ask("Who will win the next race?")
