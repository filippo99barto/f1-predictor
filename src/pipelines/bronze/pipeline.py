import pandas as pd
import requests
from src.storage.storage_backend import StorageBackend

class BronzePipeline:
    def __init__(self, storage_backend: StorageBackend):
        self.storage_backend = storage_backend

    def extract_bronze_data(self) -> None:
        """
        Build the bronze data from the raw data.
        """
        df_races = self._extract_races_data()
        df_results = self._extract_results_data(df_races)

        self._write_bronze_data("races", df_races)
        self._write_bronze_data("results", df_results)


    def _write_bronze_data(self, category: str, df: pd.DataFrame) -> None:
        """
        Write the bronze data to the storage backend.
        """
        self.storage_backend.write(f"bronze/{category}/{category}", df)

    def _extract_races_data(self) -> pd.DataFrame:
        """
        Extract the races data from the raw data.
        """
        df_final = pd.DataFrame()

        for year in range(2025, 2027):
            url = f"https://api.jolpi.ca/ergast/f1/{year}/races.json"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            races = data["MRData"]["RaceTable"]["Races"]
            df = pd.json_normalize(races)

            reduced_df = df[[
                "season",
                "round",
                "raceName",
                "Circuit.circuitName",
                "Circuit.Location.locality",
                "Circuit.Location.country",
                "date",
                "time",
            ]]

            df_final = pd.concat([df_final, reduced_df])

        return df_final

    def _extract_results_data(self, df_races: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the results data from the raw data.
        """
        rounds = len(df_races["round"].unique())
        print(f"total rounds: {rounds}")

        df_final = pd.DataFrame()

        for year, round in df_races[["season", "round"]].itertuples(index=False):
            url = f"https://api.jolpi.ca/ergast/f1/{year}/{round}/results.json"
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                races = data["MRData"]["RaceTable"]["Races"]

                df = pd.json_normalize(
                    races,
                    record_path="Results",
                    meta=["season", "round"],
                )

                reduced_df = df[[
                    "season",
                    "round",
                    "position",
                    "points",
                    "grid",
                    "status",
                    "Driver.driverId",
                    "Constructor.constructorId",
                ]]

                df_final = pd.concat([df_final, reduced_df])
            except requests.exceptions.HTTPError as e:
                print(f"Skipping {year} {round} because of HTTP error: {e}")
                continue
            except Exception as e:
                print(f"Skipping {year} {round} because of unknown error: {e}")
                continue

        return df_final