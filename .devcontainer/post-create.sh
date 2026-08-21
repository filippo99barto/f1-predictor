#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UI_DIR="$ROOT/packages/f1_agent/ui"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --all-packages
uv run pre-commit install

echo "Installing agent UI dependencies"
(cd "$UI_DIR" && pnpm install --frozen-lockfile)
