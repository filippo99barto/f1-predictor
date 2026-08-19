from typing import Any

from f1_agent.agent import DEFAULT_MODEL, SYSTEM_PROMPT, build_agent

_agent = None
_agent_model: str | None = None

__all__ = ["DEFAULT_MODEL", "SYSTEM_PROMPT", "ask", "get_agent"]


def get_agent(*, model: str | None = None):
    """Return a process-wide compiled agent, rebuilding only if the model changes."""
    global _agent, _agent_model
    resolved = model or DEFAULT_MODEL
    if _agent is None or _agent_model != resolved:
        _agent = build_agent(model=resolved)
        _agent_model = resolved
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


def ask(question: str, *, model: str | None = None) -> str:
    """Ask a natural-language question; returns the assistant's final answer."""
    result = get_agent(model=model).invoke({"messages": [{"role": "user", "content": question}]})
    return _last_ai_text(result)
