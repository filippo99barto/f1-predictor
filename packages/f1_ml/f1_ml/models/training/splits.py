from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass
class SplitInfo:
    mode: Literal["dev", "production"]
    split_strategy: str
    train_desc: str
    test_desc: str
    holdout_season: int
    holdout_rounds: list[int]


def get_last_completed_race(df: pd.DataFrame) -> tuple[int, int]:
    """Return max (season, round) from gold data."""
    idx = df[["season", "round"]].apply(tuple, axis=1).idxmax()
    row = df.loc[idx]
    return int(row["season"]), int(row["round"])


def _max_season_rounds(df: pd.DataFrame, season: int) -> list[int]:
    return sorted(df.loc[df["season"] == season, "round"].unique())


def split_latest_season_fraction(
    df: pd.DataFrame,
    holdout_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    """Hold out the last fraction of rounds in the latest season (at least 1 round)."""
    max_season = int(df["season"].max())
    rounds = _max_season_rounds(df, max_season)
    n_holdout = max(1, round(len(rounds) * holdout_fraction))
    holdout_rounds = [int(r) for r in rounds[-n_holdout:]]
    cutoff = holdout_rounds[0]

    train_df = df[
        (df["season"] < max_season) | ((df["season"] == max_season) & (df["round"] < cutoff))
    ]
    test_df = df[(df["season"] == max_season) & (df["round"].isin(holdout_rounds))]

    split_info = SplitInfo(
        mode="dev",
        split_strategy="latest_season_fraction",
        train_desc=f"all seasons before {max_season} + {max_season} rounds 1-{cutoff - 1}",
        test_desc=f"{max_season} rounds {holdout_rounds[0]}-{holdout_rounds[-1]}",
        holdout_season=max_season,
        holdout_rounds=holdout_rounds,
    )
    return train_df, test_df, split_info


def split_last_race_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    """Hold out exactly the last (season, round) globally."""
    holdout_season, holdout_round = get_last_completed_race(df)

    train_df = df[
        (df["season"] < holdout_season)
        | ((df["season"] == holdout_season) & (df["round"] < holdout_round))
    ]
    test_df = df[(df["season"] == holdout_season) & (df["round"] == holdout_round)]

    split_info = SplitInfo(
        mode="production",
        split_strategy="last_race",
        train_desc=f"all data before {holdout_season} R{holdout_round}",
        test_desc=f"{holdout_season} R{holdout_round}",
        holdout_season=holdout_season,
        holdout_rounds=[holdout_round],
    )
    return train_df, test_df, split_info


def get_early_stopping_val_set(
    train_df: pd.DataFrame,
    n_rounds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (fit_df, val_df) using the last n rounds of the max season in train_df."""
    max_season = int(train_df["season"].max())
    season_rounds = _max_season_rounds(train_df, max_season)
    val_rounds = set(season_rounds[-min(n_rounds, len(season_rounds)) :])

    val_df = train_df[(train_df["season"] == max_season) & (train_df["round"].isin(val_rounds))]
    fit_df = train_df[~train_df.index.isin(val_df.index)]
    return fit_df, val_df
