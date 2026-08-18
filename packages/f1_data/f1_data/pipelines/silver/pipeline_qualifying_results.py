import pandas as pd

from f1_data.pipelines.silver.schemas import SilverSchemaQualifyingResults
from f1_data.storage.storage_backend import StorageBackend


class SilverPipelineQualifyingResults:
    def __init__(self, storage_backend: StorageBackend):
        self.storage_backend = storage_backend

    def build_silver_data(self) -> pd.DataFrame:
        """
        Build the silver data from the bronze data.
        """
        df = self._read_bronze_data()
        df = self._transform_bronze_data(df)

        self._validate_silver_schema(df)
        self._write_silver_data(df)

        return df

    def _read_bronze_data(self) -> pd.DataFrame:
        """
        Read the bronze data.
        """
        df_races = self.storage_backend.read("bronze/races")
        df_qualifying = self.storage_backend.read("bronze/qualifying_results")

        return df_races.merge(df_qualifying, on=["season", "round"], how="inner")

    def _transform_bronze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the bronze data."""

        df = df.rename(columns=lambda col: col.split(".")[-1]).rename(
            columns={"position": "qualifying_position"}
        )

        df["q1_seconds"] = df["Q1"].apply(self.parse_lap_time)
        df["q2_seconds"] = df["Q2"].apply(self.parse_lap_time)
        df["q3_seconds"] = df["Q3"].apply(self.parse_lap_time)

        schema_columns = list(SilverSchemaQualifyingResults.to_schema().columns)
        return df[schema_columns]

    def _validate_silver_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the silver schema."""
        validated = SilverSchemaQualifyingResults.validate(df, lazy=True)
        return validated

    def _write_silver_data(self, df: pd.DataFrame) -> None:
        """Write the silver data."""
        self.storage_backend.write("silver/qualifying_results/data", df)

    @staticmethod
    def parse_lap_time(t: str) -> float | None:
        """Convert '1:31.471' to 91.471 seconds. Returns None for empty."""

        if not t or pd.isna(t):
            return None

        parts = t.split(":")
        minutes = int(parts[0])
        seconds = float(parts[1])

        return minutes * 60 + seconds
