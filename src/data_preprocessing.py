"""
data_preprocessing.py

Reads raw TfNSW traffic and BOM weather data, builds one merged hourly
dataset per traffic station, and computes the CO2 target variable.

Run from project root:
    python src/data_preprocessing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# --- Station configuration ---
# Maps each traffic station file to its road/lga and matching BOM weather station
STATIONS = {
    "50240 - Briens Road.csv": {
        "station_id": "50240", "road": "Briens Road", "lga": "Parramatta",
        "weather_station": "066124"
    },
    "50260 - Silverwater Road.csv": {
        "station_id": "50260", "road": "Silverwater Road", "lga": "Parramatta",
        "weather_station": "066124"
    },
    "7272 - Edgar Street.csv": {
        "station_id": "7272", "road": "Edgar Street", "lga": "Bankstown",
        "weather_station": "066137"
    },
}

# --- Emission factors (NGA Factors 2025, Table 9, Scope 1) ---
PETROL_EF = 2.31   # kg CO2-e per litre
DIESEL_EF = 2.72   # kg CO2-e per litre
LIGHT_FUEL_RATE = 11.1 / 100   # L per km (ABS Survey of Motor Vehicle Use)
HEAVY_FUEL_RATE = 28.0 / 100   # L per km (blended estimate, rigid+articulated)


def load_traffic_station(filename: str, meta: dict) -> pd.DataFrame:
    """Load one traffic station file, sum both directions, pivot to
    wide light/heavy columns, and return one row per hourly timestamp."""
    df = pd.read_csv(RAW_DIR / filename)
    hour_cols = [c for c in df.columns if c.startswith("hour_")]

    # Only keep Light/Heavy (drop "All Vehicles" - it's redundant)
    df = df[df["classification_seq"].isin(["Light Vehicles", "Heavy Vehicles"])]

    # Sum across both directions for total road volume per date/classification
    grouped = df.groupby(["date", "classification_seq", "public_holiday", "school_holiday"])[hour_cols].sum().reset_index()

    # Melt hour columns into long format
    long_df = grouped.melt(
        id_vars=["date", "classification_seq", "public_holiday", "school_holiday"],
        value_vars=hour_cols,
        var_name="hour_str",
        value_name="volume"
    )
    long_df["hour"] = long_df["hour_str"].str.replace("hour_", "").astype(int)
    long_df["timestamp"] = pd.to_datetime(long_df["date"]) + pd.to_timedelta(long_df["hour"], unit="h")

    # Pivot classification into columns: volume_light, volume_heavy
    pivoted = long_df.pivot_table(
        index=["timestamp", "public_holiday", "school_holiday"],
        columns="classification_seq",
        values="volume",
        aggfunc="sum"
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={
        "Light Vehicles": "volume_light",
        "Heavy Vehicles": "volume_heavy"
    })

    for col in ["volume_light", "volume_heavy"]:
        if col not in pivoted.columns:
            pivoted[col] = np.nan

    pivoted["station_id"] = meta["station_id"]
    pivoted["road"] = meta["road"]
    pivoted["lga"] = meta["lga"]
    pivoted["weather_station"] = meta["weather_station"]
    return pivoted


def load_weather(station_number: str) -> pd.DataFrame:
    """Load and merge max temperature + rainfall for a BOM station across
    all available years found in data/raw."""
    temp_files = sorted(RAW_DIR.glob(f"IDCJAC0010_{station_number}_*_Data.csv"))
    rain_files = sorted(RAW_DIR.glob(f"IDCJAC0009_{station_number}_*_Data.csv"))

    temp_df = pd.concat([pd.read_csv(f) for f in temp_files], ignore_index=True) if temp_files else pd.DataFrame()
    rain_df = pd.concat([pd.read_csv(f) for f in rain_files], ignore_index=True) if rain_files else pd.DataFrame()

    def make_date(df):
        return pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=df["Day"]))

    if not temp_df.empty:
        temp_df["date"] = make_date(temp_df)
        temp_df = temp_df[["date", "Maximum temperature (Degree C)"]].rename(
            columns={"Maximum temperature (Degree C)": "max_temp"})

    if not rain_df.empty:
        rain_df["date"] = make_date(rain_df)
        rain_df = rain_df[["date", "Rainfall amount (millimetres)"]].rename(
            columns={"Rainfall amount (millimetres)": "rainfall"})

    if temp_df.empty:
        weather = rain_df
    elif rain_df.empty:
        weather = temp_df
    else:
        weather = temp_df.merge(rain_df, on="date", how="outer")

    weather["weather_station"] = station_number
    return weather


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    df["month"] = df["timestamp"].dt.month
    df["date"] = df["timestamp"].dt.normalize()
    return df


def compute_co2(df: pd.DataFrame) -> pd.DataFrame:
    df["co2_estimate"] = (
        df["volume_light"].fillna(0) * LIGHT_FUEL_RATE * PETROL_EF
        + df["volume_heavy"].fillna(0) * HEAVY_FUEL_RATE * DIESEL_EF
    )
    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_stations = []
    for filename, meta in STATIONS.items():
        print(f"Loading traffic: {filename}")
        station_df = load_traffic_station(filename, meta)
        all_stations.append(station_df)
    traffic = pd.concat(all_stations, ignore_index=True)
    traffic = add_temporal_features(traffic)

    weather_stations = set(m["weather_station"] for m in STATIONS.values())
    weather_frames = []
    for ws in weather_stations:
        print(f"Loading weather: {ws}")
        weather_frames.append(load_weather(ws))
    weather = pd.concat(weather_frames, ignore_index=True)

    merged = traffic.merge(weather, on=["date", "weather_station"], how="left")
    merged = compute_co2(merged)

    cols = [
        "timestamp", "station_id", "road", "lga",
        "volume_light", "volume_heavy",
        "max_temp", "rainfall",
        "hour_of_day", "day_of_week", "is_weekend", "month",
        "public_holiday", "school_holiday",
        "co2_estimate"
    ]
    merged = merged[cols].sort_values(["station_id", "timestamp"])

    out_path = PROCESSED_DIR / "merged_dataset.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged):,} rows to {out_path}")
    print(f"Missing rainfall rows: {merged['rainfall'].isna().sum():,}")
    print(f"Missing max_temp rows: {merged['max_temp'].isna().sum():,}")
    print(f"\nPer-station row counts:")
    print(merged.groupby("station_id").size())


if __name__ == "__main__":
    main()