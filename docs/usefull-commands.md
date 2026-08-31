# Useful commands

Commands to run **inside the Dev Container** terminal (`/workspaces/f1_predictor`). The Dev Container is the supported way to run this project: it provides Python and (on start) the LangGraph server and chat UI.

Backend depends on which config you opened:

- [Local setup](local-setup.md) — Compose Postgres, MinIO, and MLflow
- [AWS setup](aws-setup.md) — RDS, SageMaker MLflow, and S3 (no local Postgres/MinIO/MLflow)

See the [README](../README.md) for the local vs AWS chooser.

## Table of contents

- [One-time setup (inside the dev container)](#one-time-setup-inside-the-dev-container)
- [Train end-to-end](#train-end-to-end)
- [Predict](#predict)
- [Assistant](#assistant)
- [Chat UI](#chat-ui)
- [Tests](#tests)
- [Lint & pre-commit](#lint--pre-commit)
- [Environment (set in dev container)](#environment-set-in-dev-container)
- [Postgres](#postgres)

## One-time setup (inside the dev container)

On first open, `.devcontainer/post-create.sh` runs `uv sync --all-packages` and installs pre-commit hooks. If you pull dependency changes later:

```bash
uv sync --all-packages
```

Copy `.env.example` to `.env` at the repo root and set your key:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Train end-to-end

Run `notebooks/train.ipynb` (bronze → silver → gold → train → cascade eval).

Bronze API (defaults to current season only; silver reads all years already on disk):

```python
bronze_pipeline.extract_bronze_data()  # current season
bronze_pipeline.extract_bronze_data(backfill=True)  # DATA_START_BACKFILL → now
bronze_pipeline.extract_bronze_data(backfill=True, start_backfill_year=2024)
```

Uncomment production lines in the notebook to register models.

## Predict

```python
from f1_ml.models.race.predict import predict_next_race

result = predict_next_race()
print(result.winner, result.podium)
```

## Assistant

```python
from f1_agent.client import ask

answer = ask("Who will win the next race?")
```

`ask()` reuses a process-wide LangChain + Gemini agent (`get_agent()`) with in-memory short-term memory. Follow-up questions in the same process keep prior messages; pass `thread_id` to isolate conversations. See `notebooks/assistant.ipynb`.

## Chat UI

On container start, `.devcontainer/post-start.sh` launches LangGraph on `:2024` and the UI on `:3000`. Open `http://localhost:3000` from your browser (port forwarded from the dev container).

If you stopped them or need to restart manually:

```bash
uv run langgraph dev --no-browser --host 0.0.0.0 --port 2024
# other terminal
cd packages/f1_agent/ui && pnpm dev --hostname 0.0.0.0 --port 3000
```

The UI talks to the server at `http://localhost:2024` (graph id `agent`). Requires `GEMINI_API_KEY` in `.env`.

## Tests

```bash
uv run pytest                    # full suite
uv run pytest --cov=f1_data --cov=f1_ml --cov=f1_agent --cov-report=term-missing
uv run pytest packages/f1_ml/tests/features/     # unit tests only (no data/models)
uv run pytest packages/f1_ml/tests/inference/    # schedule/lineup + mocked predict smoke tests
```

HTML coverage report:

```bash
uv run pytest --cov=f1_data --cov=f1_ml --cov=f1_agent --cov-report=html
```

Output in `htmlcov/`.

## Lint & pre-commit

```bash
pre-commit install              # once per clone (post-create usually does this)
pre-commit run --all-files      # ruff, vulture, pytest with coverage
```

Config: `pyproject.toml` (`[tool.ruff]`, `[tool.vulture]`, `[tool.coverage.*]`). Hooks: `.pre-commit-config.yaml` (ruff check, ruff format, vulture, pytest with coverage). Pre-commit enforces a minimum coverage threshold via `fail_under` in `pyproject.toml`.

## Environment (set in dev container)

**Local** defaults are configured in `.devcontainer/docker-compose.yml` for the `app` service. Override via repo-root `.env` only when you need to (e.g. `GEMINI_API_KEY`). **AWS** uses `.devcontainer/.env` for `MLFLOW_TRACKING_URI`, `RDS_HOST`, and `F1_PREDICTOR_DB_PASSWORD` — see [AWS setup](aws-setup.md).

| Variable | Default (local stack) |
|----------|--------------------------|
| `GEMINI_API_KEY` | Set in `.env` (required for assistant / chat UI) |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` (UI on host: `http://localhost:5001`) |
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio:9000` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | `minioadmin` / `minioadmin` |
| `F1_DATABASE_URL` | `postgresql://f1_predictor:f1_predictor@postgres:5432/f1_predictor` |
| `F1_REPO_ROOT` | Optional override for repo root detection |

Backfill start year: `packages/f1_data/f1_data/config/constants.py` (`DATA_START_BACKFILL = 2022`, used when `extract_bronze_data(backfill=True)`).

**Generated locally (gitignored):** `.env`

## Postgres

**Local stack only.** Postgres hosts two databases on one instance: `mlflow` (tracking) and `f1_predictor` (bronze/silver/gold tables). Users, databases, and F1 DDL are created by `.devcontainer/init-postgres.sh` on a **fresh** volume only. After pulling a layout change, recreate the `postgres-data` Docker volume (or run that script by hand against an existing instance) so init runs again.

On AWS, `f1_predictor` lives on RDS (see [AWS setup](aws-setup.md)); there is no local `postgres` service.
