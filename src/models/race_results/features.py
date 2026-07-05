import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features to the dataframe."""
    
    df = transform_data(df)

    df = add_driver_last_race_position(df)

    df = add_driver_median_position_last_3_races(df)
    df = add_constructor_median_position_last_3_races(df)
    df = add_driver_circuit_median_position_last_3_races(df)

    df = add_driver_circuit_median_career_position(df)
    df = add_driver_season_mediam_position(df)
    df = add_driver_positions_gained_season_median(df)
    df = add_driver_positions_gained_career_median(df)
    df = add_constructor_median_season_position(df)
    df = add_driver_grid_season_median(df)

    return df

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the data."""

    df = df.dropna(subset=["grid"])
    #df = df[df["status"].isin(["Finished", "Lapped", "+1 Lap", "+2 Laps"])]

    return df

def add_driver_last_race_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver last race position feature."""

    df = df.sort_values(["driverId", "season", "round"])
    df["driver_last_race_position"] = (
        df.groupby("driverId")["position"]
        .shift(1)
        .fillna(df["grid"]) # fill na with grid position
    )
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
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(df["grid"]) # fill na with grid position
    )
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

    df = df.sort_values(["driverId", "circuitId", "season", "round"])
    df["driver_circuit_median_position_last_3_races"] = (
        df.groupby(["driverId", "circuitId"])["position"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).median()))
    return df

def add_driver_positions_gained_season_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver positions gained season median feature."""

    df = df.sort_values(["driverId", "season", "round"])

    df["driver_positions_gained_season_median"] = (
        df.assign(position_minus_grid=df["grid"] - df["position"])
        .groupby(["driverId", "season"])["position_minus_grid"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(0) # fill na with 0
    )
    return df

def add_driver_positions_gained_career_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver positions gained career median feature."""

    df = df.sort_values(["driverId", "season", "round"])
    df["driver_positions_gained_career_median"] = (
        df.assign(position_minus_grid=df["grid"] - df["position"])
        .groupby(["driverId"])["position_minus_grid"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(0) # fill na with 0
    )
    return df

def add_driver_circuit_median_career_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver circuit median career position feature."""

    df = df.sort_values(["driverId", "circuitId", "season", "round"])
    df["driver_circuit_median_career_position"] = (
        df.groupby(["driverId", "circuitId"])["position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(df["grid"]) # fill na with grid position
    )
    return df

def add_constructor_median_season_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add the constructor season median position feature."""

    df = df.sort_values(["constructorId", "season", "round"])

    team_race = (
        df.groupby(["constructorId", "season", "round"], as_index=False)
        .agg(
            team_position=("position", "mean"),
            team_grid=("grid", "mean")
        ) 
        .sort_values(["constructorId", "season", "round"])
    )

    team_race["constructor_median_season_position"] = (
        team_race.groupby("constructorId")["team_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(team_race["team_grid"]) # fill na with team grid position
    )

    df = df.merge(
        team_race[["constructorId", "season", "round", "constructor_median_season_position"]],
        on=["constructorId", "season", "round"],
        how="left",
    )
    return df

def add_driver_grid_season_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver grid season median feature."""

    df = df.sort_values(["driverId", "season", "round"])
    df["driver_grid_season_median"] = (
        df.groupby(["driverId", "season"])["grid"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(df["grid"]) # fill na with grid position
    )
    return df