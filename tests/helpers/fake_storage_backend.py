import pandas as pd
from src.storage.storage_backend import StorageBackend

class FakeStorageBackend(StorageBackend):
    def __init__(self, reads: dict[str, pd.DataFrame]):
        self.reads = reads
        self.writes: dict[str, pd.DataFrame] = {}

    def read(self, dataset: str) -> pd.DataFrame:
        if dataset not in self.reads:
            raise KeyError(f"Unknown dataset: {dataset}")
        return self.reads[dataset].copy()

    def write(self, dataset: str, df: pd.DataFrame) -> None:
        self.writes[dataset] = df.copy()