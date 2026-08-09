DATA_START_BACKFILL = 2022  # default start year when bronze extract(backfill=True)

JOLPI_API_URL = "https://api.jolpi.ca/ergast/f1"
JOLPI_API_LIMIT = 100

BRONZE_RACES_COLUMNS = [
    "season",
    "round",
    "raceName",
    "Circuit.circuitId",
    "Circuit.Location.locality",
    "Circuit.Location.country",
    "date",
    "time",
]

BRONZE_RECE_RESULTS_COLUMNS = [
    "season",
    "round",
    "position",
    "points",
    "grid",
    "status",
    "Driver.driverId",
    "Constructor.constructorId",
]

BRONZE_QUALIFYING_RESULTS_COLUMNS = [
    "season",
    "round",
    "position",
    "Driver.driverId",
    "Constructor.constructorId",
    "Q1",
    "Q2",
    "Q3",
]

BRONZE_CONSTRUCTORS_COLUMNS = [
    "season",
    "constructorId",
    "name",
    "nationality",
]
