import pandera.pandas as pa
from pandera.typing import Series

class SilverSchema(pa.DataFrameModel):
    season: Series[int] = pa.Field(ge=1950)
    round: Series[int] = pa.Field(ge=1)
    raceName: Series[str]
    circuitName: Series[str]
    locality: Series[str] = pa.Field(nullable=True)
    country: Series[str] = pa.Field(nullable=True)
    date: Series[str]
    time: Series[str] = pa.Field(nullable=True)
    position: Series[int] = pa.Field(ge=1)
    points: Series[float] = pa.Field(ge=0)
    grid: Series[int] = pa.Field(ge=0, nullable=True)
    status: Series[str]
    driverId: Series[str]
    constructorId: Series[str]

    class Config:
        strict = True
        coerce = True