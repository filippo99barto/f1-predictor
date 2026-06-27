import pandas as pd
from src.storage.storage_backend import StorageBackend
from src.pipelines.silver.schema import SilverSchema
import pandera.pandas as pa

class SilverPipeline:
    def __init__(self, storage_backend: StorageBackend):
        self.storage_backend = storage_backend

    def build_silver_data(self) -> pd.DataFrame:
        """
        Build the silver data from the bronze data.
        """
        df_races, df_results = self._read_bronze_data()

        # Transform
        df = df_races.merge(df_results, on=["season", "round"], how="inner")
        df = self._transform_bronze_data(df)

        df = self._validate_silver_schema(df)

        self._write_silver_data(df)
        return df

    def _read_bronze_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read the bronze data.
        """
        df_races = self.storage_backend.read("bronze/races")
        df_results = self.storage_backend.read("bronze/results")
        return df_races, df_results

    def _write_silver_data(self, df: pd.DataFrame) -> None:
        """
        Write the silver data.
        """
        self.storage_backend.write("silver/seasons", df)

    def _transform_bronze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the bronze data."""

        df = df.rename(columns=lambda col: col.split(".")[-1])
        df["grid"] = df["grid"].replace(0, pd.NA)
        return df


    def _validate_silver_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the silver schema."""
        validated = SilverSchema.validate(df, lazy=True)

        duplicate_keys = validated.duplicated(
            subset=["season", "round", "driverId"]
        )
        if duplicate_keys.any():
            raise pa.errors.SchemaError(
                schema=SilverSchema,
                data=validated,
                message="Duplicate (season, round, driverId) rows found",
            )

        return validated
