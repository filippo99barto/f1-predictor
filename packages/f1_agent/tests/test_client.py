from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from f1_agent import client
from f1_agent.agent import agent
from f1_agent.client import DEFAULT_THREAD_ID, ask, get_agent


@pytest.fixture(autouse=True)
def reset_agent():
    client._agent = None
    yield
    client._agent = None


def test_ask_raises_without_gemini_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(OSError, match="GEMINI_API_KEY"):
        ask("Who will win the next race?")


def test_ask_raises_without_openai_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(OSError, match="OPENAI_API_KEY"):
        ask("Who will win the next race?")


def test_get_agent_reuses_compiled_graph(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    compiled = MagicMock(name="compiled-agent")
    with (
        patch("f1_agent.agent.ChatGoogleGenerativeAI"),
        patch("f1_agent.agent.ChatOpenAI"),
        patch("f1_agent.agent.create_agent"),
        patch("f1_agent.agent.StateGraph") as graph_cls,
    ):
        graph_cls.return_value.compile.return_value = compiled
        first = get_agent()
        second = get_agent()

    assert first is compiled
    assert second is compiled
    graph_cls.return_value.compile.assert_called_once()
    checkpointer = graph_cls.return_value.compile.call_args.kwargs["checkpointer"]
    assert isinstance(checkpointer, InMemorySaver)


def test_langgraph_factory_builds_agent(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    compiled = MagicMock(name="compiled-agent")
    with (
        patch("f1_agent.agent._enable_agent_tracking"),
        patch("f1_agent.agent.ChatGoogleGenerativeAI"),
        patch("f1_agent.agent.ChatOpenAI"),
        patch("f1_agent.agent.create_agent"),
        patch("f1_agent.agent.StateGraph") as graph_cls,
    ):
        graph_cls.return_value.compile.return_value = compiled
        assert agent() is compiled
    assert graph_cls.return_value.compile.call_args.kwargs["checkpointer"] is None


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
