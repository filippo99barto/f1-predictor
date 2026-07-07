import pandas as pd
from src.storage.storage_backend import StorageBackend
from src.pipelines.gold.schemas import GoldSchemaRaceResults
from src.models.race_results.features import add_features
from src.models.qualifying_results.features import add_features as add_features_qualifying_results

class GoldPipelineRaceResults:
    def __init__(self, storage_backend: StorageBackend):
        self.storage_backend = storage_backend

    def build_gold_data(self) -> pd.DataFrame:
        """
        Build the gold data from the silver data.
        """
        df = self._read_silver_data()
        df = self._transform_silver_data(df)
        df = self._validate_gold_schema(df)
        self._write_gold_data(df)
        
        return df
    
    def _read_silver_data(self) -> pd.DataFrame:
        """
        Read the silver data.
        """

        df_race_results = self.storage_backend.read("silver/race_results")
        df_qualifying_results = self.storage_backend.read("silver/qualifying_results")
        df = pd.merge(df_race_results, df_qualifying_results, on=["season", "round", "driverId", "constructorId", "circuitId"], how="inner")

        return df
    
    def _transform_silver_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the silver data.
        """
        df["q1_seconds"] = df["Q1"].apply(self.parse_lap_time)
        df["q2_seconds"] = df["Q2"].apply(self.parse_lap_time)
        df["q3_seconds"] = df["Q3"].apply(self.parse_lap_time)

        df = add_features(df)
        df = add_features_qualifying_results(df)

        schema_columns = list(GoldSchemaRaceResults.to_schema().columns)
        return df[schema_columns]

    def _write_gold_data(self, df: pd.DataFrame) -> None:
        """
        Write the gold data.
        """
        self.storage_backend.write("gold/race_results/data", df)
    
    def _validate_gold_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the gold schema.
        """
        return GoldSchemaRaceResults.validate(df, lazy=True)

    @staticmethod
    def parse_lap_time(t: str) -> float | None:
        """Convert '1:31.471' to 91.471 seconds. Returns None for empty."""

        if not t or pd.isna(t):
            return None

        parts = t.split(":")
        minutes = int(parts[0])
        seconds = float(parts[1])

        return minutes * 60 + seconds