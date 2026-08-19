# F1 Predictor

Predicts Formula 1 qualifying and race finishing positions using XGBoost, exposed through a Gemini assistant that calls the models as tools — never by guessing.

Data comes from the [Jolpi Ergast API](https://api.jolpi.ca/ergast/f1) (2022–present). Two regressors run in a **qualifying → race cascade**: the quali model predicts grid order, that feeds the race model, and the assistant surfaces the results in natural language.

## Architecture

```mermaid
flowchart TB
    API[Jolpi Ergast API] --> Bronze
    Bronze --> Silver
    Silver --> Gold
    Gold --> QualiModel[Qualifying XGBoost]
    Gold --> RaceModel[Race XGBoost]
    QualiModel --> Cascade[Cascade Inference]
    RaceModel --> Cascade
    Cascade --> Agent[LangChain Agent]
    Agent --> Ask["ask()"]
    Agent --> Server[LangGraph server]
    Server --> UI[Chat UI]
```

At inference time grid order is unknown, so the quali model runs first. Its predictions are written into `qualifying_position`, then the race model predicts finishing positions. See `packages/f1_ml/f1_ml/models/qualifying/predict.py`, `packages/f1_ml/f1_ml/models/race/predict.py`, and `packages/f1_ml/f1_ml/models/evaluate.py`.

## Pipeline & models

| Stage | What it does |
|-------|----------------|
| **Bronze** | Raw API extracts; incremental (current season) or optional historical backfill |
| **Silver** | Cleaned tables, Pandera validation |
| **Gold** | Silver inner-join + engineered features for training |
| **Quali model** | 8 features → `qualifying_position` |
| **Race model** | 11 features → `position` (grid input: `qualifying_position` only) |
| **Cascade** | Quali predictions fed into race model at inference/eval |

Features live in `packages/f1_ml/f1_ml/models/common/features/` (shared primitives) and `packages/f1_ml/f1_ml/models/{qualifying,race}/features.py` (model recipes). All use lag/shift to avoid leakage; missing values on first circuit visits fall back to `qualifying_position`.

Both models are XGBoost regressors (`reg:absoluteerror`), logged to MLflow with MAE overall and by slice (top 3, top 10, P11+).

| Mode | Split | Output |
|------|-------|--------|
| `dev` | Last 50% of rounds in latest season | Metrics logged to MLflow (not registered) |
| `production` | Hold out most recent completed race | Retrain on all data, register in MLflow |

## Quick start

**Dev container (recommended)** — opens with dependencies via `uv sync`, Postgres + MinIO + MLflow sidecars. MLflow UI: `http://localhost:5001`, MinIO console: `http://localhost:9001` (login `minioadmin` / `minioadmin`).

Postgres hosts two databases on one instance: `mlflow` (tracking) and `f1_predictor` (bronze/silver/gold tables). Both users, databases, and F1 DDL are created by `.devcontainer/init-postgres.sh` on a **fresh** volume only. After pulling this layout, recreate the `postgres-data` volume (or run that script by hand against an existing instance) so the init runs again.

**Manual setup:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-packages
```

Optional `.env` for the assistant:

```
GEMINI_API_KEY=your_key_here
```

**Train end-to-end** — run `notebooks/train.ipynb` (bronze → silver → gold → train → cascade eval).

Bronze API (defaults to current season only; silver reads all years already on disk):

```python
bronze_pipeline.extract_bronze_data()  # current season
bronze_pipeline.extract_bronze_data(backfill=True)  # DATA_START_BACKFILL → now
bronze_pipeline.extract_bronze_data(backfill=True, start_backfill_year=2024)
```

Uncomment production lines in the notebook to register models.

**Predict:**

```python
from f1_ml.models.race.predict import predict_next_race

result = predict_next_race()
print(result.winner, result.podium)
```

**Assistant:**

```python
from f1_agent.client import ask

answer = ask("Who will win the next race?")
```

`ask()` reuses a process-wide LangChain + Gemini agent (`get_agent()`). See `notebooks/assistant.ipynb`.

**Chat UI** — run the LangGraph server, then the Next.js app in `packages/f1_agent/ui`:

```bash
uv run langgraph dev
# other terminal
cd packages/f1_agent/ui && pnpm install && pnpm dev
```

Open `http://localhost:3000`. The UI talks to the server at `http://localhost:2024` (graph id `agent`), so chat threads persist. Needs `GEMINI_API_KEY`.


**Tests:**

```bash
uv run pytest                    # full suite
uv run pytest --cov=f1_data --cov=f1_ml --cov=f1_agent --cov-report=term-missing
uv run pytest packages/f1_ml/tests/features/     # unit tests only (no data/models)
uv run pytest packages/f1_ml/tests/inference/    # schedule/lineup + mocked predict smoke tests
```

For a local HTML report: `uv run pytest --cov=f1_data --cov=f1_ml --cov=f1_agent --cov-report=html` (output in `htmlcov/`).

**Lint & pre-commit:**

```bash
uv sync --all-packages
pre-commit install        # once per clone — runs ruff + pytest on every commit
pre-commit run --all-files  # lint/format/test entire repo manually
```

Config: `pyproject.toml` (`[tool.ruff]`, `[tool.coverage.*]`). Hooks: `.pre-commit-config.yaml` (ruff check, format, pytest with coverage). Pre-commit enforces a minimum coverage threshold via `fail_under` in `pyproject.toml`.

## Project layout

```
f1_predictor/
├── .devcontainer/          devcontainer.json, docker-compose.yml (app + postgres + minio + mlflow)
├── packages/
│   ├── f1_data/            Bronze/silver pipelines, storage, config
│   ├── f1_ml/              Gold pipelines, qualifying/race models, inference
│   └── f1_agent/           LangChain agent, tools, chat UI (`ui/`)
├── langgraph.json          LangGraph server entry (`f1_agent.agent:agent`)
├── datasets/               Local JSON extracts (gitignored)
├── notebooks/              train.ipynb, assistant.ipynb
├── plans/                  Design notes and roadmaps
├── pyproject.toml          uv workspace root
└── uv.lock
```

## Configuration

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Required for `ask()` and the LangGraph server |
| `MLFLOW_TRACKING_URI` | Default `http://mlflow:5000` in devcontainer; `http://localhost:5001` when port-forwarded locally |
| `MLFLOW_S3_ENDPOINT_URL` | Default `http://minio:9000` in devcontainer (MinIO artifact store) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | MinIO credentials in devcontainer (`minioadmin` / `minioadmin`) |
| `F1_DATABASE_URL` | Postgres URL for F1 tables. Default `postgresql://f1_predictor:f1_predictor@postgres:5432/f1_predictor` in the devcontainer |
| `F1_REPO_ROOT` | Optional override for repo root detection (used by path config) |

Backfill start year: `packages/f1_data/f1_data/config/constants.py` (`DATA_START_BACKFILL = 2022`, used when `extract_bronze_data(backfill=True)`).

**Generated locally (gitignored):** `datasets/`, `mlruns/`, `mlflow.db`, `.env`

---

## README screenshots (optional)

Add images under `docs/screenshots/` and reference them here to make the project easier to skim. Suggested captures from MLflow UI (`http://localhost:5000`):

| Screenshot | Where in MLflow | Why include it |
|------------|-----------------|----------------|
| **Experiments overview** | Home → Experiments list | Shows the three experiments: `f1-qualifying-results-predictor`, `f1-race-results-predictor`, `f1-cascade-predictor` |
| **Single model run — metrics** | Open a quali or race run → Metrics tab | Chart `mae` vs `baseline_mae`; shows the model beats the naive baseline |
| **Slice metrics** | Same run → Metrics | `mae_top3`, `mae_top10`, `mae_p11_plus` — demonstrates performance across the grid, not just overall |
| **Run parameters** | Same run → Parameters | `features`, `mode`, `split_strategy`, `holdout_fraction` — documents what was trained |
| **Cascade eval run** | `f1-cascade-predictor` → latest run | All eight metrics (`qualy_mae_*`, `race_mae_*`) on one screen — end-to-end pipeline quality |
| **Model registry** | Models → Registered Models | Both `f1_qualifying_predictor` and `f1_position_predictor` with version history |

**Nice-to-have:**

- **Compare runs** — overlay two dev runs after a hyperparameter or feature change
- **Assistant output** — screenshot from `assistant.ipynb` showing a podium prediction (complements MLflow)

Example embed once captured:

```markdown
![MLflow cascade metrics](docs/screenshots/mlflow-cascade-metrics.png)
```
