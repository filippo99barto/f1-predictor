import pandas as pd
from src.storage.storage_backend import StorageBackend
from src.pipelines.gold.schemas import GoldSchemaRaceResults
from src.models.race_results.features import add_features

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
        return self.storage_backend.read("silver/race_results")
    
    def _transform_silver_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the silver data.
        """
        df = add_features(df)

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