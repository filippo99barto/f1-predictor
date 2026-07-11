import pandas as pd


def add_driver_previous_race_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver last race position feature."""

    df = df.sort_values(["driverId", "season", "round"])
    df["driver_last_race_position"] = (
        df.groupby("driverId")["position"]
        .shift(1)
        .fillna(df["starting_position"]) # fill na with starting_position position
    )
    return df

def add_driver_median_position_previous_3_races(df: pd.DataFrame) -> pd.DataFrame:
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
        .fillna(df["starting_position"]) # fill na with starting_position position
    )
    return df


def add_driver_circuit_median_position_previous_3_races(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver circuit median position last 3 races feature."""

    df = df.sort_values(["driverId", "circuitId", "season", "round"])
    df["driver_circuit_median_position_last_3_races"] = (
        df.groupby(["driverId", "circuitId"])["position"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
        .fillna(df["starting_position"]) # fill na with starting_position position
    )
    return df

def add_driver_positions_gained_season_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver positions gained season median feature."""

    df = df.sort_values(["driverId", "season", "round"])

    df["driver_positions_gained_season_median"] = (
        df.assign(position_minus_starting_position=df["starting_position"] - df["position"])
        .groupby(["driverId", "season"])["position_minus_starting_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(0) # fill na with 0
    )
    return df

def add_driver_positions_gained_career_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver positions gained career median feature."""

    df = df.sort_values(["driverId", "season", "round"])
    df["driver_positions_gained_career_median"] = (
        df.assign(position_minus_starting_position=df["starting_position"] - df["position"])
        .groupby(["driverId"])["position_minus_starting_position"]
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
        .fillna(df["starting_position"]) # fill na with starting_position position
    )
    return df

def add_driver_starting_position_season_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver starting_position season median feature."""

    df = df.sort_values(["driverId", "season", "round"])
    df["driver_starting_position_season_median"] = (
        df.groupby(["driverId", "season"])["starting_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(df["starting_position"]) # fill na with starting_position position
    )
    return df

def add_driver_previous_qualifying_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver's last qualifying position feature."""
    
    df = df.sort_values(["driverId", "season", "round"])
    df["driver_last_qualifying_position"] = (
        df.groupby("driverId")["qualifying_position"]
        .shift(1)
        .fillna(df["driver_qualifying_career_median"])
    )
    return df


def add_driver_previous_qualifying_gap_to_pole(df: pd.DataFrame) -> pd.DataFrame:
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
        .fillna(0)
    )

    return df

def add_driver_qualifying_season_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver's qualifying season median feature."""
    
    df = df.sort_values(["driverId", "season", "round"])
    df["driver_qualifying_season_median"] = (
        df.groupby(["driverId", "season"])["qualifying_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(df["driver_qualifying_career_median"])
    )
    return df

def add_driver_qualifying_career_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver's qualifying career median feature."""
    
    df = df.sort_values(["driverId", "season", "round"])
    df["driver_qualifying_career_median"] = (
        df.groupby("driverId")["qualifying_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
    )
    return df

def add_driver_qualifying_circuit_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver's qualifying circuit median feature."""
    
    df = df.sort_values(["driverId", "circuitId", "season", "round"])
    df["driver_qualifying_circuit_median"] = (
        df.groupby(["driverId", "circuitId"])["qualifying_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).median())
        .fillna(df["driver_qualifying_career_median"])
    )
    return df