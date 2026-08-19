from abc import ABC, abstractmethod

import pandas as pd


class StorageBackend(ABC):
    @abstractmethod
    def read(self, schema: str, table: str) -> pd.DataFrame: ...

    @abstractmethod
    def write(self, schema: str, table: str, df: pd.DataFrame) -> None: ...
