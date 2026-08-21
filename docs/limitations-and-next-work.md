# Limitations and next work

Known gaps in data, models, inference, and architecture — and planned improvements.

## Table of contents

- [Limitations](#limitations)
  - [Data and refresh](#data-and-refresh)
  - [Features](#features)
  - [Models](#models)
  - [Rookies, swaps, and lineup](#rookies-swaps-and-lineup)
  - [Inference and product](#inference-and-product)
  - [Architecture](#architecture)
- [Next work](#next-work)

## Limitations

### Data and refresh

- **Manual pipelines** — bronze → silver → gold → train → MLflow after each weekend; nothing auto-runs on container start.
- **Jolpi Ergast only, from 2022** — no pre-2022 history unless you change backfill.
- **Missing signals** — weather, tyres, practice pace, penalties, sprint format, news, injuries, driver swaps before quali, telemetry/sectors, fastest lap.
- **Sparse results** — race/quali positions, grid, status, Q-times; no strategy or live session data.

### Features

- **Backward-looking stats only** — last race, rolling medians, positions gained, circuit medians; no exogenous inputs.
- **Constructor = average driver finish** that weekend — not car performance or standings.

### Models

- **XGBoost MAE regression** — continuous positions, no win/podium probabilities.
- **Tuned for top-3 / top-10 MAE**, not full-grid or back-marker order.
- **Race training skips DNFs/retirements** — still predicts a full finishing order at inference.
- **Train/serve gap** — race model trained on **actual** quali; before Saturday silver data, uses **predicted** quali (errors compound).
- **Simple holdout** — last fraction of latest season (dev) or last race only (production); no multi-season CV.

### Rookies, swaps, and lineup

- **Less history → weaker features** — imputed with quali position, career medians, or `0`.
- **True rookies** can have **missing features** and be **dropped** from predictions.
- **Lineup guess** — quali grid if in silver, else **last race’s** driver/team pairs; misses mid-week announcements.
- **Team changes** — driver history spans old and new constructor; constructor stats may not match.

### Inference and product

- **Next race only** — not full-season or arbitrary-round forecasting.
- **Fantasy scope** — pole/grid and race order (win, podium, top 10 via sorting); no fastest lap etc.

### Architecture

- **Dev-container-first** — Postgres, MinIO, MLflow in Docker; no production deploy or scheduler.
- **On-demand inference** — loads from MLflow per tool call; no feature store or dedicated serving layer.
- **Hard-coded quali → race cascade** — no joint model or simulation.

## Next work

- Auto refresh + retrain after each race.
- Full fantasy categories + human vs model scoring.
- Weather, sprint weekends, practice, news/lineup feeds.
- Probabilities (not just MAE positions).
- Better rookie/substitute handling and lineup source.
- Production hardening (reliable UI start, optional cloud deploy).
