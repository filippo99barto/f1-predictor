#!/bin/bash
# Runs once on a fresh Postgres volume (docker-entrypoint-initdb.d).
# Recreate the postgres-data volume (or run this script by hand) to pick up changes.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER mlflow WITH PASSWORD 'mlflow';
    CREATE DATABASE mlflow OWNER mlflow;

    CREATE USER f1_predictor WITH PASSWORD 'f1_predictor';
    CREATE DATABASE f1_predictor OWNER f1_predictor;
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname f1_predictor <<-EOSQL
    CREATE SCHEMA bronze AUTHORIZATION f1_predictor;
    CREATE SCHEMA silver AUTHORIZATION f1_predictor;
    CREATE SCHEMA gold AUTHORIZATION f1_predictor;

    SET ROLE f1_predictor;

    CREATE TABLE bronze.races (
        season INTEGER NOT NULL,
        round INTEGER NOT NULL,
        race_name TEXT,
        circuit_id TEXT,
        locality TEXT,
        country TEXT,
        "date" TEXT,
        "time" TEXT,
        PRIMARY KEY (season, round)
    );

    CREATE TABLE bronze.race_results (
        season INTEGER NOT NULL,
        round INTEGER NOT NULL,
        position TEXT,
        points TEXT,
        grid TEXT,
        status TEXT,
        driver_id TEXT NOT NULL,
        constructor_id TEXT,
        PRIMARY KEY (season, round, driver_id)
    );

    CREATE TABLE bronze.qualifying_results (
        season INTEGER NOT NULL,
        round INTEGER NOT NULL,
        position TEXT,
        driver_id TEXT NOT NULL,
        constructor_id TEXT,
        q1 TEXT,
        q2 TEXT,
        q3 TEXT,
        PRIMARY KEY (season, round, driver_id)
    );

    CREATE TABLE bronze.constructors (
        season INTEGER NOT NULL,
        constructor_id TEXT NOT NULL,
        name TEXT,
        nationality TEXT,
        PRIMARY KEY (season, constructor_id)
    );

    CREATE TABLE silver.race_results (
        season INTEGER NOT NULL,
        round INTEGER NOT NULL,
        circuit_id TEXT,
        position INTEGER,
        points DOUBLE PRECISION,
        starting_position INTEGER,
        status TEXT,
        driver_id TEXT NOT NULL,
        constructor_id TEXT,
        PRIMARY KEY (season, round, driver_id)
    );

    CREATE TABLE silver.qualifying_results (
        season INTEGER NOT NULL,
        round INTEGER NOT NULL,
        circuit_id TEXT,
        qualifying_position INTEGER,
        driver_id TEXT NOT NULL,
        constructor_id TEXT,
        q1_seconds DOUBLE PRECISION,
        q2_seconds DOUBLE PRECISION,
        q3_seconds DOUBLE PRECISION,
        PRIMARY KEY (season, round, driver_id)
    );

    CREATE TABLE gold.race_results (
        position INTEGER,
        season INTEGER NOT NULL,
        round INTEGER NOT NULL,
        status TEXT,
        driver_id TEXT NOT NULL,
        constructor_id TEXT,
        starting_position INTEGER,
        qualifying_position INTEGER,
        q1_seconds DOUBLE PRECISION,
        q2_seconds DOUBLE PRECISION,
        q3_seconds DOUBLE PRECISION,
        driver_last_race_position INTEGER,
        driver_median_position_last_3_races DOUBLE PRECISION,
        constructor_median_position_last_3_races DOUBLE PRECISION,
        driver_circuit_median_position_last_3_races DOUBLE PRECISION,
        driver_season_median_position DOUBLE PRECISION,
        driver_positions_gained_season_median DOUBLE PRECISION,
        driver_positions_gained_career_median DOUBLE PRECISION,
        driver_circuit_median_career_position DOUBLE PRECISION,
        constructor_median_season_position DOUBLE PRECISION,
        driver_qualifying_season_median DOUBLE PRECISION,
        PRIMARY KEY (season, round, driver_id)
    );

    CREATE TABLE gold.qualifying_results (
        qualifying_position INTEGER,
        season INTEGER NOT NULL,
        round INTEGER NOT NULL,
        driver_id TEXT NOT NULL,
        constructor_id TEXT,
        driver_positions_gained_season_median DOUBLE PRECISION,
        driver_positions_gained_career_median DOUBLE PRECISION,
        driver_last_qualifying_position DOUBLE PRECISION,
        last_qualifying_gap_to_pole DOUBLE PRECISION,
        driver_qualifying_season_median DOUBLE PRECISION,
        driver_qualifying_career_median DOUBLE PRECISION,
        driver_qualifying_circuit_median DOUBLE PRECISION,
        constructor_qualifying_season_median DOUBLE PRECISION,
        PRIMARY KEY (season, round, driver_id)
    );
EOSQL
