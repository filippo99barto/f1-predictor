from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from f1_agent import client
from f1_agent.agent import agent
from f1_agent.client import DEFAULT_THREAD_ID, ask, get_agent


@pytest.fixture(autouse=True)
def reset_agent():
    client._agent = None
    client._agent_model = None
    yield
    client._agent = None
    client._agent_model = None


def test_ask_raises_without_gemini_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(OSError, match="GEMINI_API_KEY"):
        ask("Who will win the next race?")


def test_get_agent_reuses_compiled_graph(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    compiled = MagicMock(name="compiled-agent")
    with (
        patch("f1_agent.agent.ChatGoogleGenerativeAI"),
        patch("f1_agent.agent.create_agent", return_value=compiled) as create,
    ):
        first = get_agent()
        second = get_agent()

    assert first is compiled
    assert second is compiled
    create.assert_called_once()
    checkpointer = create.call_args.kwargs["checkpointer"]
    assert isinstance(checkpointer, InMemorySaver)


def test_get_agent_rebuilds_when_model_changes(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with (
        patch("f1_agent.agent.ChatGoogleGenerativeAI"),
        patch(
            "f1_agent.agent.create_agent",
            side_effect=[MagicMock(), MagicMock()],
        ) as create,
    ):
        get_agent()
        get_agent(model="gemini-2.5-flash")

    assert create.call_count == 2


def test_langgraph_factory_builds_agent(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    compiled = MagicMock(name="compiled-agent")
    with (
        patch("f1_agent.agent.ChatGoogleGenerativeAI"),
        patch("f1_agent.agent.create_agent", return_value=compiled) as create,
    ):
        assert agent() is compiled
    assert create.call_args.kwargs["checkpointer"] is None


def test_ask_invokes_shared_agent(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    compiled = MagicMock()
    compiled.invoke.return_value = {"messages": [MagicMock(content="antonelli")]}
    with patch("f1_agent.client.get_agent", return_value=compiled):
        assert ask("Who will win?") == "antonelli"

    compiled.invoke.assert_called_once_with(
        {"messages": [{"role": "user", "content": "Who will win?"}]},
        config={"configurable": {"thread_id": DEFAULT_THREAD_ID}},
    )


def test_ask_keeps_thread_across_follow_ups(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    compiled = MagicMock()
    compiled.invoke.return_value = {"messages": [MagicMock(content="ok")]}
    with patch("f1_agent.client.get_agent", return_value=compiled):
        ask("Who will win?")
        ask("Why?")

    thread_ids = [
        call.kwargs["config"]["configurable"]["thread_id"]
        for call in compiled.invoke.call_args_list
    ]
    assert thread_ids == [DEFAULT_THREAD_ID, DEFAULT_THREAD_ID]


def test_ask_uses_explicit_thread_id(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    compiled = MagicMock()
    compiled.invoke.return_value = {"messages": [MagicMock(content="ok")]}
    with patch("f1_agent.client.get_agent", return_value=compiled):
        ask("Who will win?", thread_id="session-2")

    compiled.invoke.assert_called_once_with(
        {"messages": [{"role": "user", "content": "Who will win?"}]},
        config={"configurable": {"thread_id": "session-2"}},
    )
