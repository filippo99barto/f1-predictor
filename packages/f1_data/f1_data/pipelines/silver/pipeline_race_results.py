import pandas as pd

from f1_data.pipelines.silver.schemas import SilverSchemaRaceResults
from f1_data.storage.storage_backend import StorageBackend


class SilverPipelineRaceResults:
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
        df_results = self.storage_backend.read("bronze/race_results")
        return df_races.merge(df_results, on=["season", "round"], how="inner")

    def _transform_bronze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the bronze data."""

        df = df.rename(columns=lambda col: col.split(".")[-1]).rename(
            columns={"grid": "starting_position"}
        )
        df["starting_position"] = self._resolve_starting_positions_pitlane_starts(df)

        schema_columns = list(SilverSchemaRaceResults.to_schema().columns)

        return df[schema_columns]

    @staticmethod
    def _resolve_starting_positions_pitlane_starts(df: pd.DataFrame) -> pd.Series:
        grid = pd.to_numeric(df["starting_position"], errors="coerce")
        grid_field_size = df.groupby(["season", "round"])["driverId"].transform("size")
        return grid.where(grid != 0, grid_field_size).astype(int)

    def _validate_silver_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the silver schema."""
        validated = SilverSchemaRaceResults.validate(df, lazy=True)
        return validated

    def _write_silver_data(self, df: pd.DataFrame) -> None:
        """Write the silver data."""
        self.storage_backend.write("silver/race_results/data", df)
