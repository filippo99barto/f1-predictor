# Local setup

Run the project in a Dev Container with Compose-backed Postgres, MinIO, and MLflow on your machine. Chat UI and LangGraph still run in the `app` container.

For an optional AWS backend (RDS + SageMaker MLflow + S3), see [AWS setup](aws-setup.md).

## Table of contents

- [Prerequisites](#prerequisites)
- [Run the project](#run-the-project)
- [Local URLs](#local-urls)
- [Dev container stack](#dev-container-stack)

## Prerequisites

- **Docker** with Compose — [Docker Desktop](https://www.docker.com/products/docker-desktop/), [OrbStack](https://orbstack.dev/) (macOS), or any Engine with Compose v2
- **Editor with Dev Containers** — [VS Code](https://code.visualstudio.com/) or [Cursor](https://cursor.com/) ([Dev Containers](https://containers.dev/) extension; also [GitHub Codespaces](https://github.com/codespaces))

## Run the project

1. Clone the repo and open it in a Dev Container. Choose config **F1 Predictor** ([`.devcontainer/devcontainer.json`](../.devcontainer/devcontainer.json)) — *Dev Containers: Reopen in Container*.
2. Wait for `post-create` (deps + pre-commit) and `post-start` (LangGraph + chat UI).
3. Copy [`.env.example`](../.env.example) → `.env` at the repo root and set `GEMINI_API_KEY` for the assistant and chat UI.
4. **Load data and train models** — run [`notebooks/train.ipynb`](../notebooks/train.ipynb) end to end (bronze → silver → gold → train → eval). The Dev Container does not populate Postgres or register models on its own. On first setup, use bronze backfill if you need history beyond the current season ([Train end-to-end](usefull-commands.md#train-end-to-end)).

Bronze/silver/gold tables, MLflow, and Postgres are provided by the Dev Container stack — you still run the pipelines and training yourself via the notebook.

## Local URLs

| URL | Service |
|-----|---------|
| [localhost:5001](http://localhost:5001) | MLflow UI |
| [localhost:9001](http://localhost:9001) | MinIO console (`minioadmin` / `minioadmin`) |
| [localhost:3000](http://localhost:3000) | Agent chat UI |
| [localhost:2024](http://localhost:2024) | LangGraph API (graph id `agent`) |

## Dev container stack

| Service | Role |
|---------|------|
| **app** | Python 3.12 workspace (`uv`, notebooks, tests) |
| **postgres** | `f1_predictor` DB (bronze/silver/gold) + `mlflow` DB |
| **minio** | MLflow artifact store (S3-compatible) |
| **mlflow** | Experiment tracking and model registry |
| **LangGraph + UI** | Started in `app` on container boot |

Defined in [`.devcontainer/docker-compose.yml`](../.devcontainer/docker-compose.yml). Postgres schemas and tables are initialized on a fresh volume via [`.devcontainer/init-postgres.sh`](../.devcontainer/init-postgres.sh).
