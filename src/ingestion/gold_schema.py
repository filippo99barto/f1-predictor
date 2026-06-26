from pandera import DateTime
import pandera.pandas as pa
from pandera.typing import Series


class GoldSchema(pa.DataFrameModel):
    season: Series[int] = pa.Field(ge=1950)
    round: Series[int] = pa.Field(ge=1)
    raceName: Series[str]
    circuitName: Series[str]
    locality: Series[str] = pa.Field(nullable=True)
    country: Series[str] = pa.Field(nullable=True)
    position: Series[int] = pa.Field(ge=1)
    points: Series[float] = pa.Field(ge=0)
    grid: Series[int] = pa.Field(ge=0, nullable=True)
    status: Series[str]
    driverId: Series[str]
    constructorId: Series[str]
    race_datetime: Series[DateTime]
    driver_last_race_position: Series[int] = pa.Field(ge=1, nullable=True)
    driver_median_position_last_3_races: Series[int] = pa.Field(ge=1, nullable=True)

    class Config:
        strict = True
        coerce = True