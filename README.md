# F1 Predictor

ML-backed F1 race and qualifying predictions with a LangChain agent, built for a fantasy league where we compare model picks to human predictions across categories (podium, pole, and more).

Race and driver data: [Jolpi Ergast API](https://api.jolpi.ca/ergast/f1) (2022–present).

## Table of contents

- [Flowchart](#flowchart)
- [Project structure](#project-structure)
- [Screenshots](#screenshots)
- [Getting started](#getting-started)
- [Extra Documentation](#extra-documentation)

## Flowchart

```mermaid
flowchart LR
    API[Jolpi Ergast API] --> Bronze
    Bronze --> Silver
    Silver --> Gold
    Gold --> QualiModel[Qualifying XGBoost]
    Gold --> RaceModel[Race XGBoost]
    QualiModel --> Cascade[Cascade inference]
    RaceModel --> Cascade
    Cascade --> Agent[LangChain agent]
    Agent --> Ask["ask()"]
    Agent --> Server[LangGraph server]
    Server --> UI[Chat UI]
```

| Layer | Description |
|-------|-------------|
| **Bronze** | Raw API extracts (current season or historical backfill) → Postgres |
| **Silver** | Cleaned tables with Pandera validation |
| **Gold** | Feature-engineered tables for training |
| **Quali model** | Predicts `qualifying_position` (XGBoost, MAE) |
| **Race model** | Predicts `position`; uses real grid when available, else predicted quali |
| **Agent** | Gemini + tools for schedule, grid, and race predictions |

Both models are XGBoost regressors (`reg:absoluteerror`). **Features and hyperparameters were tuned with top‑3 and top‑10 MAE as the main targets** (`mae_top3`, `mae_top10` in MLflow), matching fantasy-league categories like podium and top‑10 picks — not overall grid MAE alone. Training also logs overall MAE, `mae_p11_plus`, and baseline comparisons.

Training modes (`dev` / `production`).

## Project structure

```
f1_predictor/
├── .devcontainer/       Dev Containers (local Compose stack + aws/ overlay)
├── packages/
│   ├── f1_data/         Bronze/silver pipelines, Postgres storage
│   ├── f1_ml/           Gold pipelines, models, inference
│   └── f1_agent/        LangChain agent, tools, chat UI (`ui/`)
├── notebooks/           train.ipynb, assistant.ipynb, eval.ipynb
├── docs/                setup walkthroughs, extra docs & resources
├── terraform/           optional AWS backend (S3, RDS, SageMaker MLflow)
├── langgraph.json       LangGraph server entrypoint
├── pyproject.toml       uv workspace root
└── uv.lock
```

## Screenshots

### Agent chat

Podium prediction and a follow-up in the chat UI (`http://localhost:3000`):

![Chat UI — podium prediction and follow-up](docs/resources/chat-ui-podium-and-followup.png)

Next race schedule from the agent (`get_next_race_info`):

![Chat UI — next race metadata](docs/resources/chat-ui-next-race-metadata.png)

### MLflow metrics

Holdout metrics for the qualifying model (`http://localhost:5001`):

![MLflow — qualifying results metrics](docs/resources/mlflow-qualifying-results-metrics.png)

Holdout metrics for the race model:

![MLflow — race results metrics](docs/resources/mlflow-race-results-metrics.png)

End-to-end cascade eval (predicted grid → race):

![MLflow — quali/race cascade metrics](docs/resources/mlflow-quali-race-cascade-metrics.png)

Agent Traces Evaluation:

![MLflow - Traces Evaluation](docs/resources/mlflow-agent-eval-traces.png)

## Getting started

Choose a backend, then follow the matching walkthrough.

| | Local | AWS |
|---|--------|-----|
| **Stack** | Compose Postgres, MinIO, MLflow | RDS Postgres, SageMaker MLflow, S3 |
| **Dev Container** | **F1 Predictor** | **F1 Predictor (AWS)** |
| **Guide** | [Local setup](docs/local-setup.md) | [AWS setup](docs/aws-setup.md) |

```mermaid
flowchart LR
  subgraph local [Local Dev Container]
    AppLocal[app]
    PG[postgres]
    Minio[minio]
    MLflowLocal[mlflow]
    AppLocal --> PG
    AppLocal --> MLflowLocal
    MLflowLocal --> Minio
  end
  subgraph aws [AWS Dev Container]
    AppAWS[app]
    RDS[RDS Postgres]
    Sage[SageMaker MLflow]
    S3[S3 artifacts]
    AppAWS --> RDS
    AppAWS --> Sage
    Sage --> S3
  end
```

Chat UI and LangGraph run in `app` in both modes ([localhost:3000](http://localhost:3000) / [localhost:2024](http://localhost:2024)). MLflow `:5001` and MinIO `:9001` are local-only.

## Extra Documentation

| Doc | Contents |
|-----|----------|
| [Local setup](docs/local-setup.md) | Docker Dev Container: Postgres, MinIO, MLflow |
| [AWS setup](docs/aws-setup.md) | Terraform + AWS Dev Container: RDS, SageMaker MLflow, S3 |
| [Useful commands](docs/usefull-commands.md) | Train, predict, assistant, tests, lint, env vars |
| [Limitations and next work](docs/limitations-and-next-work.md) | Known gaps (data, models, inference) and planned improvements |
| [f1_data](packages/f1_data/README.md) | Bronze/silver pipelines |
| [f1_ml](packages/f1_ml/README.md) | Gold, training, inference |
| [f1_agent](packages/f1_agent/README.md) | Agent tools and chat UI |
