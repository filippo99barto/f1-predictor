from pathlib import Path

DATA_START_BACKFILL = 2022  # default start year when bronze extract(backfill=True)

LOCAL_DATA_DIR = Path(__file__).resolve().parents[4] / "datasets"

JOLPI_API_URL = "https://api.jolpi.ca/ergast/f1"
JOLPI_API_LIMIT = 100

BRONZE_RACES_COLUMNS = {
    "season": "season",
    "round": "round",
    "raceName": "race_name",
    "Circuit.circuitId": "circuit_id",
    "Circuit.Location.locality": "locality",
    "Circuit.Location.country": "country",
    "date": "date",
    "time": "time",
}

BRONZE_RECE_RESULTS_COLUMNS = {
    "season": "season",
    "round": "round",
    "position": "position",
    "points": "points",
    "grid": "grid",
    "status": "status",
    "Driver.driverId": "driver_id",
    "Constructor.constructorId": "constructor_id",
}

BRONZE_QUALIFYING_RESULTS_COLUMNS = {
    "season": "season",
    "round": "round",
    "position": "position",
    "Driver.driverId": "driver_id",
    "Constructor.constructorId": "constructor_id",
    "Q1": "q1",
    "Q2": "q2",
    "Q3": "q3",
}

BRONZE_CONSTRUCTORS_COLUMNS = {
    "season": "season",
    "constructorId": "constructor_id",
    "name": "name",
    "nationality": "nationality",
}
