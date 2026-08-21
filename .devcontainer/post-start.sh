#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspaces/f1_predictor
UI_DIR="$ROOT/packages/f1_agent/ui"
UI_LOG=/tmp/f1-agent-ui.log
UI_PID=/tmp/f1-agent-ui.pid
export PATH="${HOME}/.local/bin:${PATH}"

port_open() {
  bash -c "echo >/dev/tcp/127.0.0.1/$1" >/dev/null 2>&1
}

process_alive() {
  local pid=$1
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

start_background() {
  local name=$1
  local logfile=$2
  shift 2
  echo "Starting $name (log: $logfile)"
  : >"$logfile"
  setsid nohup "$@" >>"$logfile" 2>&1 &
  echo $! >"${logfile}.pid"
  disown || true
}

wait_for_port() {
  local port=$1
  local logfile=$2
  local label=$3
  local timeout=${4:-90}

  for _ in $(seq 1 "$timeout"); do
    if port_open "$port"; then
      echo "$label ready on :$port"
      return 0
    fi
    sleep 1
  done

  echo "ERROR: $label did not start on :$port within ${timeout}s" >&2
  echo "--- tail of $logfile ---" >&2
  tail -40 "$logfile" >&2 || true
  return 1
}

wait_for_port_steady() {
  local port=$1
  local logfile=$2
  local label=$3
  local pidfile=$4
  local timeout=${5:-90}

  wait_for_port "$port" "$logfile" "$label" "$timeout" || return 1

  for _ in 1 2 3 4 5; do
    sleep 1
    if ! port_open "$port"; then
      echo "ERROR: $label stopped listening on :$port shortly after start" >&2
      tail -40 "$logfile" >&2 || true
      return 1
    fi
    if [[ -f "$pidfile" ]]; then
      local pid
      pid=$(cat "$pidfile")
      if ! process_alive "$pid"; then
        echo "ERROR: $label process $pid exited shortly after start" >&2
        tail -40 "$logfile" >&2 || true
        return 1
      fi
    fi
  done

  echo "$label still running on :$port"
}

cd "$ROOT"

if port_open 2024; then
  echo "LangGraph already running on :2024"
else
  start_background "LangGraph" /tmp/langgraph.log \
    uv run langgraph dev --no-browser --host 0.0.0.0 --port 2024
  wait_for_port 2024 /tmp/langgraph.log "LangGraph" 120 || true
fi

if port_open 3000; then
  echo "Agent UI already running on :3000"
else
  if [[ ! -d "$UI_DIR/node_modules/next" ]]; then
    echo "UI node_modules missing; running pnpm install"
    (cd "$UI_DIR" && pnpm install --frozen-lockfile >>"$UI_LOG" 2>&1)
  fi
  start_background "agent UI" "$UI_LOG" \
    bash -c "cd '$UI_DIR' && exec ./node_modules/.bin/next dev --hostname 0.0.0.0 --port 3000"
  cp "${UI_LOG}.pid" "$UI_PID"
  wait_for_port_steady 3000 "$UI_LOG" "Agent UI" "$UI_PID" 90 || true
fi
