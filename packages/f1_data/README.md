# F1 Data

Extracts Formula 1 data from the [Jolpi Ergast API](https://api.jolpi.ca/ergast/f1) and stores it in Postgres as bronze and silver tables.

## Table of contents

- [Dev container](#dev-container)
- [Layout](#layout)
- [Usage](#usage)
- [Config](#config)

## Dev container

If you open the repo in the dev container, setup is already done: `uv sync`, Postgres, and `F1_DATABASE_URL` are configured via `.devcontainer/docker-compose.yml`. You can run the pipelines below without installing deps or wiring the database yourself.

## Layout

```
f1_data/
  config/           API URLs, column mappings, backfill start year
  pipelines/
    bronze/         Raw API extracts (races, results, qualifying, constructors)
    silver/         Cleaned, validated race and qualifying results
  storage/          Postgres backend (default) and local JSON backend
```

## Usage

```python
from f1_data.storage.postgres_storage_backend import get_storage_backend
from f1_data.pipelines.bronze.pipeline import BronzePipeline
from f1_data.pipelines.silver.pipeline_race_results import SilverPipelineRaceResults
from f1_data.pipelines.silver.pipeline_qualifying_results import SilverPipelineQualifyingResults

storage = get_storage_backend()

BronzePipeline(storage).extract_bronze_data()  # current season
# BronzePipeline(storage).extract_bronze_data(backfill=True)  # 2022 → now

SilverPipelineRaceResults(storage).build_silver_data()
SilverPipelineQualifyingResults(storage).build_silver_data()
```

Bronze holds the full calendar (including future rounds). Silver only has weekends that already have results.

## Config

| Variable | Purpose |
|----------|---------|
| `F1_DATABASE_URL` | Postgres connection (set automatically in the dev container) |

See `notebooks/train.ipynb` for the full bronze → silver flow.
