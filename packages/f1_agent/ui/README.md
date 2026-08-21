# F1 Agent Chat UI

Next.js chat frontend for the F1 prediction agent. It talks to the LangGraph server (not `ask()`), so threads and follow-ups persist.

Based on LangChain’s open-source [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui) — a generic Next.js app for chatting with any LangGraph server that exposes a `messages` key. This copy is wired to the local F1 agent (`graph` id `agent`).

## Dev container

On container start, `.devcontainer/post-start.sh` runs `langgraph dev` on `:2024` and `pnpm dev` on `:3000` for you. Open `http://localhost:3000` after the dev container is ready. Add `GEMINI_API_KEY` to repo-root `.env` (see `.env.example`).

## Manual run

From the repo root, start the agent server:

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
