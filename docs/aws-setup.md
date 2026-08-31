# AWS setup

Optional backend: RDS Postgres, SageMaker MLflow, and S3 instead of the local Compose Postgres / MinIO / MLflow stack. The `app` Dev Container still runs Chat UI and LangGraph locally (`:3000` / `:2024`).

For the default local stack, see [Local setup](local-setup.md).

## Table of contents

- [Prerequisites](#prerequisites)
- [Apply Terraform](#apply-terraform)
- [Map outputs into `.devcontainer/.env`](#map-outputs-into-devcontainerenv)
- [Open the AWS Dev Container](#open-the-aws-dev-container)
- [Train and run](#train-and-run)
- [Caveats](#caveats)

## Prerequisites

- **AWS CLI** with a named profile `f1-predictor` (credentials for the account Terraform will use)
- **Terraform** `>= 1.15`
- Region **`eu-west-2`** (hardcoded in [`terraform/provider.tf`](../terraform/provider.tf))
- **Docker** with Compose and an editor with [Dev Containers](https://containers.dev/) (same as [local setup](local-setup.md#prerequisites))

Confirm the profile:

```bash
aws sts get-caller-identity --profile f1-predictor
```

Use the Account ID from that command as `aws_account_id` in `terraform.tfvars`. Terraform refuses to apply against any other account.

## Apply Terraform

From the repo:

1. Copy [`terraform/terraform.tfvars.example`](../terraform/terraform.tfvars.example) → `terraform/terraform.tfvars`.
2. Set `aws_account_id` and `my_ip` (CIDR, e.g. `x.x.x.x/32`). Your current public IP: `curl -s https://checkip.amazonaws.com`.
3. From `terraform/`:

```bash
terraform init
terraform apply
```

This creates S3 artifacts, a public RDS instance allowlisted to `my_ip`, the `f1_predictor` database and role, and a SageMaker MLflow app.

## Map outputs into `.devcontainer/.env`

Compose interpolation for the AWS overlay reads `.devcontainer/.env` (gitignored). Copy [`.devcontainer/.env.example`](../.devcontainer/.env.example) → `.devcontainer/.env` and fill:

| Variable | Terraform output |
|----------|------------------|
| `MLFLOW_TRACKING_URI` | `mlflow_app_arn` |
| `RDS_HOST` | `rds_endpoint` |
| `F1_PREDICTOR_DB_PASSWORD` | `f1_predictor_role_password` |

```bash
cd terraform
terraform output -raw mlflow_app_arn
terraform output -raw rds_endpoint
terraform output -raw f1_predictor_role_password
```

Do not commit `.env` or `*.tfvars`.

You still need a repo-root `.env` with `GEMINI_API_KEY` (same as [local setup](local-setup.md#run-the-project)).

## Open the AWS Dev Container

Reopen in Container with config **F1 Predictor (AWS)** ([`.devcontainer/aws/devcontainer.json`](../.devcontainer/aws/devcontainer.json)).

That overlay starts only `app` (no local Postgres / MinIO / MLflow). It mounts `~/.aws` read-only and sets `AWS_PROFILE=f1-predictor`. `F1_DATABASE_URL` and `MLFLOW_TRACKING_URI` come from `.devcontainer/.env`.

Wait for `post-create` and `post-start` as in the local path.

## Train and run

Same notebook flow as local: run [`notebooks/train.ipynb`](../notebooks/train.ipynb) end to end. Data lands in RDS; experiment tracking and artifacts go to SageMaker MLflow / S3. Chat UI and LangGraph stay in `app` at [localhost:3000](http://localhost:3000) and [localhost:2024](http://localhost:2024).

MLflow `:5001` and MinIO `:9001` are local-stack only — they are not running in this config.

Commands: [Useful commands](usefull-commands.md).

## Caveats

- If your public IP changes, update `my_ip` and re-apply so RDS stays reachable.
- This is a personal/dev stack: RDS uses `skip_final_snapshot`, S3 uses `force_destroy`. Destroying the stack can delete data.
- Never commit `.env`, `.devcontainer/.env`, or `*.tfvars`.
