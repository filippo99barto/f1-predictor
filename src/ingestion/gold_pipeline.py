from src.storage.storage_backend import StorageBackend
import pandas as pd
from src.ingestion.gold_schema import GoldSchema

class GoldPipeline:
    def __init__(self, storage_backend: StorageBackend):
        self.storage_backend = storage_backend

    def build_gold_data(self) -> pd.DataFrame:
        """
        Build the gold data from the silver data.
        """
        df_silver = self._read_silver_data()
        df = self._transform_silver_data(df_silver)

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

        # Add driver last race position column
        df = df.sort_values(["driverId", "season", "round"])
        df["driver_last_race_position"] = df.groupby("driverId")["position"].shift(1)

        df = df.sort_values(["driverId", "season", "round"])
        df["driver_median_position_last_3_races"] = (
            df.groupby("driverId")["position"].
            transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
        )

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