# F1 ML

Qualifying and race XGBoost models, gold feature pipelines, and inference for the next Grand Prix.

## Dev container

If you open the repo in the dev container, Python deps, Postgres (`F1_DATABASE_URL`), MLflow, and MinIO are already running and configured. Port-forward MLflow at `http://localhost:5001` to inspect runs. You still need trained models in the registry (run `notebooks/train.ipynb`) before `predict_*` works.

## Layout

```
f1_ml/
  pipelines/gold/       Silver → gold tables with engineered features
  models/
    common/             Training, splits, metrics, shared feature primitives
    qualifying/         Quali features, train, predict
    race/               Race features, train, predict (uses quali for grid)
    evaluate_quali_race_cascade.py   Joint holdout eval (predicted grid → race)
  inference/next_race.py   Calendar, lineup, history + scaffold for target weekend
```

## Usage

**Train** (via `notebooks/train.ipynb`):

```python
from f1_ml.models.qualifying.train import train_model as train_qualifying
from f1_ml.models.race.train import train_model as train_race
from f1_ml.models.evaluate_quali_race_cascade import evaluate_quali_race_cascade

train_qualifying(mode="dev")
train_race(mode="dev")
evaluate_quali_race_cascade(mode="dev")
```

**Predict** next race (loads models from MLflow registry):

```python
from f1_ml.models.qualifying.predict import predict_next_qualifying
from f1_ml.models.race.predict import predict_next_race

quali = predict_next_qualifying()
race = predict_next_race()  # uses Saturday grid if in silver, else predicted quali
```

Models register to MLflow in `production` mode only. Artifacts load from `models:/{name}/latest`.

## Depends on

- `f1-data` for Postgres reads (wired in the dev container)
- MLflow tracking URI and MinIO for model registry (also wired in the dev container)
