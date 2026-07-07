import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features to the dataframe."""
    
    df = add_driver_qualifying_career_median(df)
    df = add_driver_last_qualifying_position(df)
    df = add_driver_last_qualifying_gap_to_pole(df)
    df = add_driver_qualifying_season_median(df)
    df = add_driver_qualifying_circuit_median(df)
    df = add_constructor_qualifying_season_median(df)

    return df

def add_driver_last_qualifying_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver's last qualifying position feature."""
    
    df = df.sort_values(["driverId", "season", "round"])
    df["driver_last_qualifying_position"] = (
        df.groupby("driverId")["grid_position"]
        .shift(1)
        .fillna(df["driver_qualifying_career_median"])
    )
    return df

def add_driver_last_qualifying_gap_to_pole(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver's last qualifying gap to pole position feature."""
    
    df = df.sort_values(["driverId", "season", "round"])
    df["best_q_seconds"] = df[["q1_seconds", "q2_seconds", "q3_seconds"]].min(axis=1)

    # Step 1: gap to pole for each race (grouped by race)
    df["qualifying_gap_to_pole"] = (
        df.groupby(["season", "round"])["best_q_seconds"]
        .transform(lambda s: s - s.min())
    )

    # Step 2: previous race's gap, per driver
    df = df.sort_values(["driverId", "season", "round"])
    df["last_qualifying_gap_to_pole"] = (
        df.groupby("driverId")["qualifying_gap_to_pole"]
        .shift(1)
    )

    return df

def add_driver_qualifying_season_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver's qualifying season median feature."""
    
    df = df.sort_values(["driverId", "season", "round"])
    df["driver_qualifying_season_median"] = (
        df.groupby(["driverId", "season"])["grid_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(df["driver_qualifying_career_median"])
    )
    return df

def add_driver_qualifying_career_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver's qualifying career median feature."""
    
    df = df.sort_values(["driverId", "season", "round"])
    df["driver_qualifying_career_median"] = (
        df.groupby("driverId")["grid_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
    )
    return df

def add_driver_qualifying_circuit_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver's qualifying circuit median feature."""
    
    df = df.sort_values(["driverId", "circuitId", "season", "round"])
    df["driver_qualifying_circuit_median"] = (
        df.groupby(["driverId", "circuitId"])["grid_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(df["driver_qualifying_career_median"])
    )
    return df

def add_constructor_qualifying_season_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the constructor's qualifying season median feature."""
    
    df = df.sort_values(["constructorId", "season", "round"])

    team_qualifying = (
        df.groupby(["constructorId", "season", "round"], as_index=False)
        .agg(
            team_position=("grid_position", "mean"),
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