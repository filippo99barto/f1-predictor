# F1 Agent

LangChain + Gemini assistant that calls the F1 ML tools. The compiled graph is built once per process (`get_agent()`). `ask()` is a notebook/CLI helper; the chat UI talks to a LangGraph server.

## Notebook / CLI

```python
from f1_agent.client import ask

answer = ask("Who will win the next race?")
why = ask("Why that driver?")  # same conversation; sees prior tool results
```

Follow-ups reuse an in-memory thread (`thread_id="default"`). Pass a new `thread_id` to start a fresh conversation.

Requires `GEMINI_API_KEY` in the environment or repo-root `.env` (copy from `.env.example`).

## Chat UI

Start the persistent agent (from the repo root):

```bash
uv run langgraph dev
```

Then in another terminal:

```bash
cd packages/f1_agent/ui
pnpm install
pnpm dev
```

The UI is at `http://localhost:3000` and connects to the server at `http://localhost:2024` (graph id `agent`). Threads are stored by the server, so follow-up questions keep context.
