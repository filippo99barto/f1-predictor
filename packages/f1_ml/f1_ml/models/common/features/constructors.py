import pandas as pd


def add_constructor_median_position_previous_3_races(df: pd.DataFrame) -> pd.DataFrame:
    """Add the constructor median position last 3 races feature."""

    df = df.sort_values(["constructor_id", "season", "round"])

    team_race = (
        df.groupby(["constructor_id", "season", "round"], as_index=False)
        .agg(team_position=("position", "mean"))  # or "mean"
        .sort_values(["constructor_id", "season", "round"])
    )

    team_race["constructor_median_position_last_3_races"] = team_race.groupby("constructor_id")[
        "team_position"
    ].transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())

    df = df.merge(
        team_race[
            ["constructor_id", "season", "round", "constructor_median_position_last_3_races"]
        ],
        on=["constructor_id", "season", "round"],
        how="left",
    )
    return df


def add_constructor_median_season_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add the constructor season median position feature."""

    df = df.sort_values(["constructor_id", "season", "round"])

    team_race = (
        df.groupby(["constructor_id", "season", "round"], as_index=False)
        .agg(
            team_position=("position", "mean"),
            team_starting_position=("starting_position", "mean"),
        )
        .sort_values(["constructor_id", "season", "round"])
    )

    team_race["constructor_median_season_position"] = (
        team_race.groupby("constructor_id")["team_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(team_race["team_starting_position"])  # fill na with team starting_position position
    )

    df = df.merge(
        team_race[["constructor_id", "season", "round", "constructor_median_season_position"]],
        on=["constructor_id", "season", "round"],
        how="left",
    )
    return df


def add_constructor_qualifying_season_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the constructor's qualifying season median feature."""

    df = df.sort_values(["constructor_id", "season", "round"])

    team_qualifying = (
        df.groupby(["constructor_id", "season", "round"], as_index=False)
        .agg(
            team_position=("qualifying_position", "mean"),
            team_career_median=("driver_qualifying_career_median", "mean"),
        )
        .sort_values(["constructor_id", "season", "round"])
    )

    team_qualifying["constructor_qualifying_season_median"] = (
        team_qualifying.groupby(["constructor_id", "season"])["team_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(team_qualifying["team_career_median"])
    )

    df = df.merge(
        team_qualifying[
            ["constructor_id", "season", "round", "constructor_qualifying_season_median"]
        ],
        on=["constructor_id", "season", "round"],
        how="left",
    )
    return df
