from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from f1_agent.agent import build_agent

_agent = None

DEFAULT_THREAD_ID = "default"

__all__ = ["DEFAULT_THREAD_ID", "ask", "get_agent"]


def get_agent():
    """Return a process-wide compiled agent."""
    global _agent
    if _agent is None:
        _agent = build_agent(checkpointer=InMemorySaver())
    return _agent


def _last_ai_text(result: dict[str, Any]) -> str:
    message = result["messages"][-1]
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


def ask(
    question: str,
    *,
    thread_id: str = DEFAULT_THREAD_ID,
) -> str:
    """Ask a natural-language question; returns the assistant's final answer.

    Follow-up calls with the same ``thread_id`` keep prior messages (and tool
    results) in context. Pass a new ``thread_id`` to start a fresh conversation.
    """
    result = get_agent().invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return _last_ai_text(result)
