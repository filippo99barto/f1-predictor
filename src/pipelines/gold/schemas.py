import pandera.pandas as pa
from pandera.typing import Series


class GoldSchemaRaceResults(pa.DataFrameModel):
    #target
    position: Series[int] = pa.Field(ge=1)

    season: Series[int] = pa.Field(ge=1950)
    round: Series[int] = pa.Field(ge=1)
    status: Series[str]
    driverId: Series[str]
    constructorId: Series[str]
    #features
    starting_position: Series[int] = pa.Field(ge=1, nullable=True)
    qualifying_position: Series[int] = pa.Field(ge=1)
    q1_seconds: Series[float] = pa.Field(ge=1, nullable=True)
    q2_seconds: Series[float] = pa.Field(ge=1, nullable=True)
    q3_seconds: Series[float] = pa.Field(ge=1, nullable=True)

    driver_last_race_position: Series[int] = pa.Field(ge=1)
    driver_median_position_last_3_races: Series[float] = pa.Field(ge=1, nullable=True)
    constructor_median_position_last_3_races: Series[float] = pa.Field(ge=1, nullable=True)
    driver_circuit_median_position_last_3_races: Series[float] = pa.Field(ge=1, nullable=True)
    driver_season_median_position: Series[float] = pa.Field(ge=1)
    driver_positions_gained_season_median: Series[float]
    driver_positions_gained_career_median: Series[float]
    driver_circuit_median_career_position: Series[float] = pa.Field(ge=1, nullable=True)
    constructor_median_season_position: Series[float] = pa.Field(ge=1)
    driver_qualifying_season_median: Series[float] = pa.Field(ge=1, nullable=True)

    class Config:
        strict = True
        coerce = True

class GoldSchemaQualifyingResults(pa.DataFrameModel):
    #target
    qualifying_position: Series[int] = pa.Field(ge=1)

    season: Series[int] = pa.Field(ge=1950)
    round: Series[int] = pa.Field(ge=1)
    driverId: Series[str]
    constructorId: Series[str]
    #features
    driver_positions_gained_season_median: Series[float]
    driver_positions_gained_career_median: Series[float]
    driver_last_qualifying_position: Series[float] = pa.Field(ge=1, nullable=True)
    last_qualifying_gap_to_pole: Series[float] = pa.Field(ge=0)
    driver_qualifying_season_median: Series[float] = pa.Field(ge=1, nullable=True)
    driver_qualifying_career_median: Series[float] = pa.Field(ge=1, nullable=True)
    driver_qualifying_circuit_median: Series[float] = pa.Field(ge=1, nullable=True)
    constructor_qualifying_season_median: Series[float] = pa.Field(ge=1, nullable=True)

    class Config:
        strict = True
        coerce = True
