import pandas as pd
from src.storage.storage_backend import StorageBackend

class SilverPipeline:
    def __init__(self, storage_backend: StorageBackend):
        self.storage_backend = storage_backend

    def build_silver_data(self) -> pd.DataFrame:
        """
        Build the silver data from the bronze data.
        """
        df_races, df_results = self._read_bronze_data()

        df = df_races.merge(df_results, on=["season", "round"], how="left")
        
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