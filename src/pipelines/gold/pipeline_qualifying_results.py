import pandas as pd
from src.storage.storage_backend import StorageBackend
from src.pipelines.gold.schemas import GoldSchemaQualifyingResults
from src.models.qualifying_results.features import build_qualifying_results_features

class GoldPipelineQualifyingResults:
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

        df = build_qualifying_results_features(df)

        schema_columns = list(GoldSchemaQualifyingResults.to_schema().columns)
        return df[schema_columns]

    def _write_gold_data(self, df: pd.DataFrame) -> None:
        """
        Write the gold data.
        """
        self.storage_backend.write("gold/qualifying_results/data", df)
    
    def _validate_gold_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the gold schema.
        """
        return GoldSchemaQualifyingResults.validate(df, lazy=True)