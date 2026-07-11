from dataclasses import dataclass

import pandas as pd

from src.config.paths import LOCAL_DATA_DIR
from src.models.race_results.train import load_training_data as load_race_gold
from src.models.training.splits import get_last_completed_race
from src.storage.local_storage_backend import LocalStorageBackend

MERGE_KEYS = ["season", "round", "driverId", "constructorId", "circuitId"]


@dataclass
class RaceInfo:
    season: int
    round: int
    race_name: str
    circuit_id: str
    date: str | None = None


def _storage() -> LocalStorageBackend:
    return LocalStorageBackend(LOCAL_DATA_DIR)


def load_race_schedule(season: int | None = None) -> pd.DataFrame:
    """Load and normalize the race calendar from bronze."""
    df = _storage().read("bronze/races")
    df = df.rename(columns=lambda col: col.split(".")[-1])
    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)
    if season is not None:
        df = df[df["season"] == season]
    return df.sort_values(["season", "round"]).reset_index(drop=True)


def get_last_completed_race_from_gold() -> tuple[int, int]:
    return get_last_completed_race(load_race_gold())


def resolve_target_race(
    season: int | None = None,
    round_num: int | None = None,
) -> RaceInfo:
    """Resolve the target race from explicit args or the next race after last completed."""
    last_season, last_round = get_last_completed_race_from_gold()

    if season is not None and round_num is not None:
        schedule = load_race_schedule(season)
        match = schedule[(schedule["season"] == season) & (schedule["round"] == round_num)]
        if match.empty:
            raise ValueError(f"Race {season} R{round_num} not found in schedule.")
        row = match.iloc[0]
        return RaceInfo(
            season=season,
            round=round_num,
            race_name=row["raceName"],
            circuit_id=row["circuitId"],
            date=row.get("date"),
        )

    schedule = load_race_schedule()
    upcoming = schedule[
        (schedule["season"] > last_season)
        | ((schedule["season"] == last_season) & (schedule["round"] > last_round))
    ]
    if upcoming.empty:
        raise ValueError(
            f"No upcoming race in schedule after {last_season} R{last_round}."
        )

    row = upcoming.iloc[0]
    return RaceInfo(
        season=int(row["season"]),
        round=int(row["round"]),
        race_name=row["raceName"],
        circuit_id=row["circuitId"],
        date=row.get("date"),
    )


def get_driver_lineup(season: int, round_num: int) -> pd.DataFrame:
    """Drivers from the most recent completed silver race before the target."""
    race_df = _storage().read("silver/race_results")
    race_df["season"] = race_df["season"].astype(int)
    race_df["round"] = race_df["round"].astype(int)

    completed = race_df[
        (race_df["season"] < season)
        | ((race_df["season"] == season) & (race_df["round"] < round_num))
    ]
    if completed.empty:
        raise ValueError("No completed race history available for lineup inference.")

    last_season = int(completed["season"].max())
    last_round = int(completed.loc[completed["season"] == last_season, "round"].max())
    latest = completed[(completed["season"] == last_season) & (completed["round"] == last_round)]

    return latest[["driverId", "constructorId"]].drop_duplicates().reset_index(drop=True)


def build_scaffold_rows(
    lineup: pd.DataFrame,
    season: int,
    round_num: int,
    circuit_id: str,
) -> pd.DataFrame:
    """Placeholder rows for a future race with unknown results."""
    scaffolds = lineup.copy()
    scaffolds["season"] = season
    scaffolds["round"] = round_num
    scaffolds["circuitId"] = circuit_id
    scaffolds["position"] = pd.NA
    scaffolds["points"] = pd.NA
    scaffolds["starting_position"] = pd.NA
    scaffolds["status"] = pd.NA
    scaffolds["qualifying_position"] = pd.NA
    scaffolds["q1_seconds"] = pd.NA
    scaffolds["q2_seconds"] = pd.NA
    scaffolds["q3_seconds"] = pd.NA
    return scaffolds


def build_inference_frames(
    season: int,
    round_num: int,
    circuit_id: str,
) -> pd.DataFrame:
    """Load silver history, merge, and append scaffold rows for the target race."""
    storage = _storage()
    race_df = storage.read("silver/race_results")
    quali_df = storage.read("silver/qualifying_results")

    merged = pd.merge(race_df, quali_df, on=MERGE_KEYS, how="inner", suffixes=("", "_quali"))
    merged = merged.drop(
        columns=[c for c in merged.columns if c.endswith("_quali")],
        errors="ignore",
    )

    lineup = get_driver_lineup(season, round_num)
    scaffolds = build_scaffold_rows(lineup, season, round_num, circuit_id)

    for col in merged.columns:
        if col not in scaffolds.columns:
            scaffolds[col] = pd.NA

    scaffolds = scaffolds[merged.columns]
    for col in merged.columns:
        if col in scaffolds.columns and col in merged.dtypes:
            if merged.dtypes[col] != scaffolds.dtypes[col]:
                scaffolds[col] = scaffolds[col].astype(merged.dtypes[col], errors="ignore")
    return pd.concat([merged, scaffolds], ignore_index=True)


def target_race_mask(df: pd.DataFrame, season: int, round_num: int) -> pd.Series:
    return (df["season"] == season) & (df["round"] == round_num)
