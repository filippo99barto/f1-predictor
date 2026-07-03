import pandera.pandas as pa
from pandera.typing import Series

class SilverSchemaRaceResults(pa.DataFrameModel):
    season: Series[int] = pa.Field(ge=1950)
    round: Series[int] = pa.Field(ge=1)
    circuitId: Series[str]
    position: Series[int] = pa.Field(ge=1)
    points: Series[float] = pa.Field(ge=0)
    grid: Series[int] = pa.Field(ge=0, nullable=True)
    status: Series[str]
    driverId: Series[str]
    constructorId: Series[str]

    class Config:
        strict = True
        coerce = True

class SilverSchemaQualifyingResults(pa.DataFrameModel):
    season: Series[int] = pa.Field(ge=1950)
    round: Series[int] = pa.Field(ge=1)
    circuitId: Series[str]
    grid_position: Series[int] = pa.Field(ge=1)
    driverId: Series[str]
    constructorId: Series[str]
    Q1: Series[str] = pa.Field(nullable=True)
    Q2: Series[str] = pa.Field(nullable=True)
    Q3: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True