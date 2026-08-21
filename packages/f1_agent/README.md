# F1 Agent

LangChain + Gemini assistant that answers F1 prediction questions by calling the ML tools — it does not guess results.

The underlying quali and race models were tuned for **top‑3 and top‑10 MAE** (podium- and top‑10-style picks). The agent surfaces their predictions; it does not re-rank or override them.

## Table of contents

- [Dev container](#dev-container)
- [Layout](#layout)
- [Notebook / CLI](#notebook--cli)
- [Chat UI](#chat-ui)

## Dev container

If you open the repo in the dev container, Python deps (`uv sync`), Node, and pnpm are already installed. Postgres and MLflow are running for the ML tools. On container start, `.devcontainer/post-start.sh` also launches the LangGraph server (`:2024`) and chat UI (`:3000`) for you — open `http://localhost:3000` once the container is up.

Add your `GEMINI_API_KEY` to repo-root `.env` (copy from `.env.example`) — that is the only secret you need to set yourself.

## Layout

```
f1_agent/
  agent.py      System prompt and compiled LangGraph agent
  tools.py      get_next_race_info, predict_next_qualifying, predict_next_race
  client.py     ask() helper for notebooks / CLI (in-memory thread memory)
  ui/           Next.js chat UI (fork of [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui))
```

## Notebook / CLI

```python
from f1_agent.client import ask

answer = ask("Who will win the next race?")
why = ask("Why that driver?")  # same conversation; sees prior tool results
```

Follow-ups reuse an in-memory thread (`thread_id="default"`). Pass a new `thread_id` to start fresh.

Requires `GEMINI_API_KEY` in repo-root `.env` (see `.env.example`). In the dev container everything else is already configured.

## Chat UI

**Dev container:** LangGraph and the UI start automatically. Open `http://localhost:3000` (server at `:2024`, graph id `agent`). Logs: `/tmp/langgraph.log`, `/tmp/f1-agent-ui.log`.

**Manual setup** — from the repo root:

```bash
uv run langgraph dev
```

Then:

```bash
cd packages/f1_agent/ui
pnpm install
pnpm dev
```

See `packages/f1_agent/ui/README.md` for UI-only details.
