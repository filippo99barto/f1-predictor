import logging
import pandas as pd
import requests
from src.storage.storage_backend import StorageBackend
from src.config.constants import (
    DATA_START_BACKFILL,
    DATA_END_BACKFILL,
    JOLPI_API_URL,
    JOLPI_API_LIMIT,
    BRONZE_RACES_COLUMNS,
    BRONZE_RECE_RESULTS_COLUMNS,
    BRONZE_QUALIFYING_RESULTS_COLUMNS,
    BRONZE_CONSTRUCTORS_COLUMNS,
)

class BronzePipeline:
    def __init__(self, storage_backend: StorageBackend):
        self.storage_backend = storage_backend

    def extract_bronze_data(self) -> None:
        """
        Build the bronze data from the raw data.
        """
        logging.info(f"Extracting data from {DATA_START_BACKFILL} to {DATA_END_BACKFILL}")

        df_races = self._extract_races_data()
        df_results = self._extract_results_data()
        df_qualifying = self._extract_qualifying_results_data()
        df_constructors = self._extract_constructors_data()

    def _write_bronze_data(self, path: str, filename: str, df: pd.DataFrame) -> None:
        """
        Write the bronze data to the storage backend.
        """
        self.storage_backend.write(f"bronze/{path}/{filename}", df)

    def _extract_races_data(self) -> None:
        """
        Extract the races data from the raw data.
        """
        for year in range(DATA_START_BACKFILL, DATA_END_BACKFILL):
            logging.info(f"Extracting races data for {year}")

            url = f"{JOLPI_API_URL}/{year}/races.json"
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                races = data["MRData"]["RaceTable"]["Races"]
                df = pd.json_normalize(races)

                reduced_df = df[BRONZE_RACES_COLUMNS]

                self._write_bronze_data("races", f"{year}", reduced_df)
            except requests.exceptions.HTTPError as e:
                logging.error(f"HTTP Error extracting races data for {year}: {e}")
                raise e
            except Exception as e:
                logging.error(f"Error extracting races data for {year}: {e}")
                raise e

    def _extract_results_data(self) -> None:
        """
        Extract the results data from the raw data.
        """
        for year in range(DATA_START_BACKFILL, DATA_END_BACKFILL):
            logging.info(f"Extracting results data for {year}")
            
            url = f"{JOLPI_API_URL}/{year}/results/"
            try:
                limit = JOLPI_API_LIMIT
                offset = 0
                all_results = []

                while True:
                    resp = requests.get(
                        url,
                        params={"limit": limit, "offset": offset},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    md = data["MRData"]
                    all_results.extend(md["RaceTable"]["Races"])
                    total = int(md["total"])

                    offset += limit
                    if offset >= total:
                        break

                df = pd.json_normalize(
                    all_results,
                    record_path="Results",
                    meta=["season", "round"],
                )

                reduced_df = df[BRONZE_RECE_RESULTS_COLUMNS]
                self._write_bronze_data("race_results", f"{year}", reduced_df)

            except requests.exceptions.HTTPError as e:
                logging.error(f"HTTP Error extracting results data for {year}: {e}")
                raise e
            except Exception as e:
                logging.error(f"Error extracting results data for {year}: {e}")
                raise e

    def _extract_qualifying_results_data(self) -> None:
        """
        Extract the qualifying results data from the raw data.
        """
        for year in range(DATA_START_BACKFILL, DATA_END_BACKFILL):
            logging.info(f"Extracting qualifying results data for {year}")

            url = f"{JOLPI_API_URL}/{year}/qualifying/"
            try:
                limit = JOLPI_API_LIMIT
                offset = 0
                all_qualifying_results = []

                while True:
                    resp = requests.get(
                        url,
                        params={"limit": limit, "offset": offset},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    md = data["MRData"]
                    all_qualifying_results.extend(md["RaceTable"]["Races"])
                    total = int(md["total"])

                    offset += limit
                    if offset >= total:
                        break

                df = pd.json_normalize(
                    all_qualifying_results,
                    record_path="QualifyingResults",
                    meta=["season", "round"],
                )
                reduced_df = df[BRONZE_QUALIFYING_RESULTS_COLUMNS]
                self._write_bronze_data("qualifying_results", f"{year}", reduced_df)

            except requests.exceptions.HTTPError as e:
                logging.error(f"HTTP Error extracting qualifying results data for {year}: {e}")
                raise e
            except Exception as e:
                logging.error(f"Error extracting qualifying results data for {year}: {e}")
                raise e
    
    def _extract_constructors_data(self) -> None:
        """
        Extract the constructors data from the raw data.
        """
        for year in range(DATA_START_BACKFILL, DATA_END_BACKFILL):
            logging.info(f"Extracting constructors data for {year}")

            url = f"{JOLPI_API_URL}/{year}/constructors.json"
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                constructors = data["MRData"]["ConstructorTable"]
                df = pd.json_normalize(
                    constructors,
                    record_path="Constructors",
                    meta=["season"],
                )

                reduced_df = df[BRONZE_CONSTRUCTORS_COLUMNS]
                self._write_bronze_data("constructors", f"{year}", reduced_df)

            except requests.exceptions.HTTPError as e:
                logging.error(f"HTTP Error extracting constructors data for {year}: {e}")
                raise e
            except Exception as e:
                logging.error(f"Error extracting constructors data for {year}: {e}")
                raise e
