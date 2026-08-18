import pandas as pd

from f1_data.storage.storage_backend import StorageBackend


class LocalStorageBackend(StorageBackend):
    def __init__(self, root):
        self.root = root

    def read(self, dataset: str) -> pd.DataFrame:
        """
        Read a dataset from the local storage.
        """
        from glob import glob

        path_to_dataset = self.root / dataset
        glob_pattern = path_to_dataset / "*.json"
        files = glob(str(glob_pattern))

        if len(files) == 0:
            raise FileNotFoundError(f"No files found for dataset: {dataset}")
        else:
            df_final = pd.DataFrame()
            for file in files:
                df = pd.read_json(file, orient="records")
                df_final = pd.concat([df_final, df])

        return df_final

    def write(self, dataset: str, df: pd.DataFrame) -> None:
        df.to_json(f"{self.root}/{dataset}.json", orient="records", date_format="iso")
