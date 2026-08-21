from dataclasses import dataclass

import pandas as pd

from f1_data.storage.postgres_storage_backend import get_storage_backend
from f1_ml.models.common.splits import get_last_completed_season_round
from f1_ml.models.race.train import load_training_data as load_race_gold

MERGE_KEYS = ["season", "round", "driver_id", "constructor_id", "circuit_id"]
QUALI_OVERLAY_COLS = ["qualifying_position", "q1_seconds", "q2_seconds", "q3_seconds"]


@dataclass
class RaceInfo:
    season: int
    round: int
    race_name: str
    circuit_id: str
    date: str | None = None


def load_race_schedule(season: int | None = None) -> pd.DataFrame:
    """Load and normalize the race calendar from bronze."""
    df = get_storage_backend().read("bronze", "races")
    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)
    if season is not None:
        df = df[df["season"] == season]
    return df.sort_values(["season", "round"]).reset_index(drop=True)


def resolve_target_race(
    season: int | None = None,
    round_num: int | None = None,
) -> RaceInfo:
    """Resolve the target race from explicit args or the next race after last completed."""
    if season is not None and round_num is not None:
        schedule = load_race_schedule(season)
        match = schedule[(schedule["season"] == season) & (schedule["round"] == round_num)]
        if match.empty:
            raise ValueError(f"Race {season} R{round_num} not found in schedule.")
        row = match.iloc[0]
        return RaceInfo(
            season=season,
            round=round_num,
            race_name=row["race_name"],
            circuit_id=row["circuit_id"],
            date=row.get("date"),
        )

    last_season, last_round = get_last_completed_season_round(load_race_gold())
    schedule = load_race_schedule()
    upcoming = schedule[
        (schedule["season"] > last_season)
        | ((schedule["season"] == last_season) & (schedule["round"] > last_round))
    ]
    if upcoming.empty:
        raise ValueError(f"No upcoming race in schedule after {last_season} R{last_round}.")

    row = upcoming.iloc[0]
    return RaceInfo(
        season=int(row["season"]),
        round=int(row["round"]),
        race_name=row["race_name"],
        circuit_id=row["circuit_id"],
        date=row.get("date"),
    )


def _filter_target_qualifying(quali_df: pd.DataFrame, season: int, round_num: int) -> pd.DataFrame:
    df = quali_df.copy()
    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)
    return df[(df["season"] == season) & (df["round"] == round_num)].copy()


def load_target_qualifying(season: int, round_num: int) -> pd.DataFrame:
    """Silver qualifying rows for the target race, if they have been extracted."""
    return _filter_target_qualifying(
        get_storage_backend().read("silver", "qualifying_results"), season, round_num
    )


def get_driver_lineup(
    season: int,
    round_num: int,
    *,
    target_quali: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Drivers from Saturday quali if present, otherwise the last completed silver race."""
    if target_quali is None:
        target_quali = load_target_qualifying(season, round_num)
    if not target_quali.empty:
        return (
            target_quali[["driver_id", "constructor_id"]].drop_duplicates().reset_index(drop=True)
        )

    race_df = get_storage_backend().read("silver", "race_results")
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

    return latest[["driver_id", "constructor_id"]].drop_duplicates().reset_index(drop=True)


def build_scaffold_rows(
    lineup: pd.DataFrame,
    season: int,
    round_num: int,
    circuit_id: str,
    *,
    target_quali: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Placeholder rows for a future race, overlaying real quali when available."""
    scaffolds = lineup.copy()
    scaffolds["season"] = season
    scaffolds["round"] = round_num
    scaffolds["circuit_id"] = circuit_id
    scaffolds["position"] = pd.NA
    scaffolds["points"] = pd.NA
    scaffolds["starting_position"] = pd.NA
    scaffolds["status"] = pd.NA
    scaffolds["qualifying_position"] = pd.NA
    scaffolds["q1_seconds"] = pd.NA
    scaffolds["q2_seconds"] = pd.NA
    scaffolds["q3_seconds"] = pd.NA

    if target_quali is None:
        target_quali = load_target_qualifying(season, round_num)
    if not target_quali.empty:
        overlay_cols = [c for c in QUALI_OVERLAY_COLS if c in target_quali.columns]
        overlay = target_quali[["driver_id", *overlay_cols]].drop_duplicates("driver_id")
        scaffolds = scaffolds.drop(columns=overlay_cols, errors="ignore")
        scaffolds = scaffolds.merge(overlay, on="driver_id", how="left")

    return scaffolds


def build_inference_frames(
    season: int,
    round_num: int,
    circuit_id: str,
) -> pd.DataFrame:
    """Load silver history, merge, and append scaffold rows for the target race."""
    storage = get_storage_backend()
    race_df = storage.read("silver", "race_results")
    quali_df = storage.read("silver", "qualifying_results")

    merged = pd.merge(race_df, quali_df, on=MERGE_KEYS, how="inner", suffixes=("", "_quali"))
    merged = merged.drop(
        columns=[c for c in merged.columns if c.endswith("_quali")],
        errors="ignore",
    )

    target_quali = _filter_target_qualifying(quali_df, season, round_num)
    lineup = get_driver_lineup(season, round_num, target_quali=target_quali)
    scaffolds = build_scaffold_rows(
        lineup, season, round_num, circuit_id, target_quali=target_quali
    )

    for col in merged.columns:
        if col not in scaffolds.columns:
            scaffolds[col] = pd.NA

    scaffolds = scaffolds[merged.columns]
    for col in merged.columns:
        if col in scaffolds.columns and merged.dtypes[col] != scaffolds.dtypes[col]:
            scaffolds[col] = scaffolds[col].astype(merged.dtypes[col], errors="ignore")
    return pd.concat([merged, scaffolds], ignore_index=True)


def target_race_mask(df: pd.DataFrame, season: int, round_num: int) -> pd.Series:
    return (df["season"] == season) & (df["round"] == round_num)
