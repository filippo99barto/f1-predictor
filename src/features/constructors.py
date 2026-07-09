import pandas as pd

def build_all_constructor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add constructor features to the dataframe."""

    df = build_all_constructor_features_race_results(df)
    df = build_all_constructor_features_qualifying_results(df)

    return df

def build_all_constructor_features_race_results(df: pd.DataFrame) -> pd.DataFrame:
    """Add constructor features to the dataframe."""

    df = add_constructor_median_position_previous_3_races(df)
    df = add_constructor_median_season_position(df)

    return df

def build_all_constructor_features_qualifying_results(df: pd.DataFrame) -> pd.DataFrame:
    """Add constructor features to the dataframe."""

    df = add_constructor_qualifying_season_median(df)

    return df

def add_constructor_median_position_previous_3_races(df: pd.DataFrame) -> pd.DataFrame:
    """Add the constructor median position last 3 races feature."""

    df = df.sort_values(["constructorId", "season", "round"])

    team_race = (
        df.groupby(["constructorId", "season", "round"], as_index=False)
        .agg(team_position=("position", "mean"))  # or "mean"
        .sort_values(["constructorId", "season", "round"])
    )

    team_race["constructor_median_position_last_3_races"] = (
        team_race.groupby("constructorId")["team_position"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    )

    df = df.merge(
        team_race[["constructorId", "season", "round", "constructor_median_position_last_3_races"]],
        on=["constructorId", "season", "round"],
        how="left",
    )
    return df

def add_constructor_median_season_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add the constructor season median position feature."""

    df = df.sort_values(["constructorId", "season", "round"])

    team_race = (
        df.groupby(["constructorId", "season", "round"], as_index=False)
        .agg(
            team_position=("position", "mean"),
            team_starting_position=("starting_position", "mean")
        ) 
        .sort_values(["constructorId", "season", "round"])
    )

    team_race["constructor_median_season_position"] = (
        team_race.groupby("constructorId")["team_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(team_race["team_starting_position"]) # fill na with team starting_position position
    )

    df = df.merge(
        team_race[["constructorId", "season", "round", "constructor_median_season_position"]],
        on=["constructorId", "season", "round"],
        how="left",
    )
    return df

def add_constructor_qualifying_season_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the constructor's qualifying season median feature."""
    
    df = df.sort_values(["constructorId", "season", "round"])

    team_qualifying = (
        df.groupby(["constructorId", "season", "round"], as_index=False)
        .agg(
            team_position=("qualifying_position", "mean"),
            team_career_median=("driver_qualifying_career_median", "mean"),
        ) 
        .sort_values(["constructorId", "season", "round"])
    )

    team_qualifying["constructor_qualifying_season_median"] = (
        team_qualifying.groupby(["constructorId", "season"])["team_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(team_qualifying["team_career_median"])
    )

    df = df.merge(
        team_qualifying[["constructorId", "season", "round", "constructor_qualifying_season_median"]],
        on=["constructorId", "season", "round"],
        how="left",
    )
    return df