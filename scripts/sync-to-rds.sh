#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../terraform"

RDS_HOST=$(terraform output -raw rds_endpoint)
F1_PREDICTOR_PASSWORD=$(terraform output -raw f1_predictor_role_password)
MLFLOW_PASSWORD=$(terraform output -raw mlflow_role_password)

LOCAL_PGPASSWORD="postgres"

declare -A DB_PASSWORD=(
  [f1_predictor]="$F1_PREDICTOR_PASSWORD"
  [mlflow]="$MLFLOW_PASSWORD"
)

for DB in "${!DB_PASSWORD[@]}"; do
  echo "==> Dumping local $DB..."
  PGPASSWORD="$LOCAL_PGPASSWORD" pg_dump -h postgres -U postgres -d "$DB" \
    --no-owner --format=custom -f "/tmp/${DB}.dump"

  echo "==> Restoring $DB into RDS as $DB role..."
  PGPASSWORD="${DB_PASSWORD[$DB]}" pg_restore -h "$RDS_HOST" -U "$DB" -d "$DB" \
    --no-owner --clean --if-exists --single-transaction \
    "/tmp/${DB}.dump"

  echo "==> Done: $DB"
done