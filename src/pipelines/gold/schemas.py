import pandera.pandas as pa
from pandera.typing import Series

class GoldSchemaRaceResults(pa.DataFrameModel):
    season: Series[int] = pa.Field(ge=1950)
    round: Series[int] = pa.Field(ge=1)
    circuitId: Series[str]
    position: Series[int] = pa.Field(ge=1)
    points: Series[float] = pa.Field(ge=0)
    grid: Series[int] = pa.Field(ge=1)
    status: Series[str]
    driverId: Series[str]
    constructorId: Series[str]
    driver_last_race_position: Series[int] = pa.Field(ge=1)
    driver_median_position_last_3_races: Series[float] = pa.Field(ge=1, nullable=True)
    constructor_median_position_last_3_races: Series[float] = pa.Field(ge=1, nullable=True)
    driver_circuit_median_position_last_3_races: Series[float] = pa.Field(ge=1, nullable=True)
    driver_season_median_position: Series[float] = pa.Field(ge=1)
    driver_positions_gained_season_median: Series[float]
    driver_positions_gained_career_median: Series[float]
    driver_circuit_median_career_position: Series[float] = pa.Field(ge=1)
    constructor_median_season_position: Series[float] = pa.Field(ge=1)
    driver_grid_season_median: Series[float] = pa.Field(ge=1)

    class Config:
        strict = True
        coerce = True