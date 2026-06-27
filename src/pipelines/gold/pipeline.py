from src.storage.storage_backend import StorageBackend
import pandas as pd
from src.pipelines.gold.schema import GoldSchema

class GoldPipeline:
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
        return self.storage_backend.read("silver")
    
    def _transform_silver_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the silver data.
        """
        # Create datetime column, drop date and time columns
        df["race_datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"])
        df = df.drop(columns=["date", "time"])

        return df

    def _write_gold_data(self, df: pd.DataFrame) -> None:
        """
        Write the gold data.
        """
        self.storage_backend.write("gold/ready_to_train", df)
    
    def _validate_gold_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the gold schema.
        """
        return GoldSchema.validate(df, lazy=True)