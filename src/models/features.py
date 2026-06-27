import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features to the dataframe."""
    
    df = add_driver_last_race_position(df)
    df = add_driver_median_position_last_3_races(df)
    df = add_constructor_median_position_last_3_races(df)
    df = add_driver_circuit_median_position_last_3_races(df)
    df = add_driver_season_mediam_position(df)
    df = add_driver_positions_gained_season_median(df)
    df = add_driver_positions_gained_career_median(df)

    features = [
        "grid",
        "season",
        "round",
        "circuitName",
        "driverId",
        "constructorId",
        "driver_last_race_position",
        "driver_median_position_last_3_races",
        "constructor_median_position_last_3_races",
        "driver_circuit_median_position_last_3_races",
        "driver_season_median_position",
        "driver_positions_gained_season_median",
        "driver_positions_gained_career_median"
    ]
    return df, features

def add_driver_last_race_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver last race position feature."""

    df = df.sort_values(["driverId", "season", "round"])
    df["driver_last_race_position"] = (
        df.groupby("driverId")["position"]
        .shift(1))
    return df

def add_driver_median_position_last_3_races(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver median position last 3 races feature."""

    df = df.sort_values(["driverId", "season", "round"])
    df["driver_median_position_last_3_races"] = (
        df.groupby("driverId")["position"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).median()))
    return df

def add_driver_season_mediam_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver season median position feature."""
    
    df = df.sort_values(["driverId", "season", "round"])
    df["driver_season_median_position"] = (
        df.groupby(["driverId", "season"])["position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median()))
    return df

def add_constructor_median_position_last_3_races(df: pd.DataFrame) -> pd.DataFrame:
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

def add_driver_circuit_median_position_last_3_races(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver circuit median position last 3 races feature."""

    df = df.sort_values(["driverId", "circuitName", "season", "round"])
    df["driver_circuit_median_position_last_3_races"] = (
        df.groupby(["driverId", "circuitName"])["position"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).median()))
    return df

def add_driver_positions_gained_season_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver positions gained season median feature."""

    df = df.sort_values(["driverId", "season", "round"])

    df["driver_positions_gained_season_median"] = (
        df.assign(position_minus_grid=df["grid"] - df["position"])
        .groupby(["driverId", "season"])["position_minus_grid"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
    )
    return df

def add_driver_positions_gained_career_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver positions gained career median feature."""

    df = df.sort_values(["driverId", "season", "round"])
    df["driver_positions_gained_career_median"] = (
        df.assign(position_minus_grid=df["grid"] - df["position"])
        .groupby(["driverId"])["position_minus_grid"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
    )
    return df