# src/storage/base.py
from abc import ABC, abstractmethod

import pandas as pd


class StorageBackend(ABC):
    @abstractmethod
    def read(self, dataset: str) -> pd.DataFrame: ...

    @abstractmethod
    def write(self, dataset: str, df: pd.DataFrame) -> None: ...
