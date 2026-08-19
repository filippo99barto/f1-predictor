# F1 Agent Chat UI

Next.js chat frontend for the F1 prediction agent. It talks to the LangGraph server (not `ask()`), so threads and follow-ups persist.

## Run

From the repo root, start the agent server (needs `GEMINI_API_KEY` in `.env`):

```bash
uv run langgraph dev
```

In another terminal:

```bash
cd packages/f1_agent/ui
pnpm install
pnpm dev
```

Open `http://localhost:3000`. The UI defaults to `http://localhost:2024` and graph id `agent`.
