#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspaces/f1_predictor
export PATH="${HOME}/.local/bin:${PATH}"

port_open() {
  bash -c "echo >/dev/tcp/127.0.0.1/$1" >/dev/null 2>&1
}

cd "$ROOT"

if port_open 2024; then
  echo "LangGraph already running on :2024"
else
  echo "Starting LangGraph on :2024"
  setsid uv run langgraph dev --no-browser --host 0.0.0.0 --port 2024 \
    </dev/null >/tmp/langgraph.log 2>&1 &
  disown || true
fi

if port_open 3000; then
  echo "Agent UI already running on :3000"
else
  echo "Starting agent UI on :3000"
  cd "$ROOT/packages/f1_agent/ui"
  pnpm install
  setsid pnpm dev --hostname 0.0.0.0 --port 3000 \
    </dev/null >/tmp/f1-agent-ui.log 2>&1 &
  disown || true
fi
