import datetime
import logging

import pandas as pd
import requests

from src.config.constants import (
    DATA_START_BACKFILL,
    JOLPI_API_URL,
    JOLPI_API_LIMIT,
    BRONZE_RACES_COLUMNS,
    BRONZE_RECE_RESULTS_COLUMNS,
    BRONZE_QUALIFYING_RESULTS_COLUMNS,
    BRONZE_CONSTRUCTORS_COLUMNS,
)
from src.storage.storage_backend import StorageBackend


class BronzePipeline:
    def __init__(self, storage_backend: StorageBackend):
        self.storage_backend = storage_backend
        self._extract_years: list[int] = []

    def extract_bronze_data(
        self,
        *,
        backfill: bool = False,
        start_backfill_year: int | None = None,
    ) -> None:
        """Extract bronze data from the Jolpi Ergast API.

        By default only the current season is fetched. Set ``backfill=True`` to
        refresh historical seasons from ``start_backfill_year`` through the
        current year (inclusive). Existing bronze files for other years are left
        on disk and are still picked up when silver is rebuilt.
        """
        self._extract_years = self._resolve_years(backfill, start_backfill_year)

        self._extract_races_data()
        self._extract_results_data()
        self._extract_qualifying_results_data()
        self._extract_constructors_data()

    @staticmethod
    def _backfill_start_year(start_backfill_year: int | None) -> int:
        return (
            start_backfill_year
            if start_backfill_year is not None
            else DATA_START_BACKFILL
        )

    @staticmethod
    def _resolve_years(
        backfill: bool,
        start_backfill_year: int | None,
    ) -> list[int]:
        current_year = datetime.datetime.now().year
        
        if backfill:
            start = BronzePipeline._backfill_start_year(start_backfill_year)
            years_to_extract = list(range(start, current_year + 1))
            logging.info("Backfill extract for seasons %s", years_to_extract)
            return years_to_extract

        logging.info("Extract for season %s", current_year)
        return [current_year]

    def _write_bronze_data(self, path: str, filename: str, df: pd.DataFrame) -> None:
        self.storage_backend.write(f"bronze/{path}/{filename}", df)

    def _extract_races_data(self) -> None:
        for year in self._extract_years:
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
        for year in self._extract_years:
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
        for year in self._extract_years:
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
        for year in self._extract_years:
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
