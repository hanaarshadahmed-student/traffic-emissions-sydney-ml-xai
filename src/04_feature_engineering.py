"""
CO2/NO2 Traffic-Emissions Capstone — Feature Engineering Script

Input:  data/processed/final_daily.csv
        data/processed/final_hourly.csv
Output: data/processed/features_daily.csv    (zero NaNs, model-ready)
        data/processed/features_hourly.csv   (zero NaNs, model-ready)
"""

import pandas as pd
import numpy as np
import os

IN_DIR = "data/processed"
OUT_DIR = "data/processed"


def classify_road(name):
    """Same classification used throughout the EDA notebook (Part 0.2)."""
    n = name.lower()
    if "highway" in n or "motorway" in n or "freeway" in n:
        return "Highway"
    elif "road" in n or "way" in n or "drive" in n or "parade" in n:
        return "Major Road"
    else:
        return "Local Street"


# ---------------------------------------------------------------------------
# STAGE 1 -- FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def add_calendar_features(df):
    """EDA 1.5: weekday/weekend NO2 difference is statistically significant
    (p << 0.001). Season is a natural extension of the temperature seasonality
    seen in Part 0.4."""
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon .. 6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month

    season_map = {12: "Summer", 1: "Summer", 2: "Summer",
                  3: "Autumn", 4: "Autumn", 5: "Autumn",
                  6: "Winter", 7: "Winter", 8: "Winter",
                  9: "Spring", 10: "Spring", 11: "Spring"}
    df["season"] = df["month"].map(season_map)
    df = pd.get_dummies(df, columns=["season"], prefix="season", dtype=int)
    return df


def add_road_type_features(df):
    """EDA 1.6: heavy-vehicle-% vs NO2 relationship changes sign depending on
    road type (Highway +0.12, Local Street +0.18, Major Road -0.10) -- pooling
    across road types hid this. road_type is one-hot encoded (only 3 categories,
    safe for any model)."""
    df["road_type"] = df["station_name"].apply(classify_road)
    df = pd.get_dummies(df, columns=["road_type"], prefix="road", dtype=int)
    return df


def add_traffic_features(df):
    """EDA 1.7: traffic_volume_total and traffic_volume_light are highly
    collinear (light vehicles dominate total volume) -- keep total + a ratio
    instead of all three raw columns.

    A small number of hourly rows (~2%, mostly overnight) have zero recorded
    vehicles of any class, which makes heavy_pct a 0/0 division. Those are
    filled with 0% rather than left undefined -- no traffic means no heavy
    vehicles either, so 0% is the sensible value, not a missing one."""
    df["heavy_pct"] = (df["traffic_volume_heavy"] / df["traffic_volume_total"] * 100)
    df.loc[df["traffic_volume_total"] == 0, "heavy_pct"] = 0
    df = df.drop(columns=["traffic_volume_light"])
    return df


def add_weather_features(df):
    """EDA 0.5: humidity_pct is not statistically significant (p=0.34, r~0.02)
    -- dropped. is_rainy captures the rain-washout effect found in EDA 0.5
    (NO2 significantly lower on rainy days, p~1.5e-05, with traffic itself NOT
    significantly different -- a genuine atmospheric effect, not a
    traffic-mediated one)."""
    df["is_rainy"] = (df["rain_mm"] > 1.0).astype(int)
    if "humidity_pct" in df.columns:
        df = df.drop(columns=["humidity_pct"])
    return df


def add_target_transform(df):
    """EDA 1.1: NO2 is right-skewed (skewness ~1.26) with heavy tails
    (kurtosis ~2.25). A log-transform is offered as an alternative target.
    A handful of NO2 readings are slightly negative (sensor noise near the
    detection floor, EDA 1.1) -- clipped to 0 before logging."""
    no2_clipped = df["no2_pphm"].clip(lower=0)
    df["log_no2_pphm"] = np.log1p(no2_clipped)
    return df


def add_lag_features(df, group_col="station_id", sort_col="date"):
    """Autoregressive features. Both use shift(1) so they only ever look at
    the PAST relative to each row -- no leakage. Optional: drop these before
    modeling if you want a model that explains NO2 from external drivers
    only, rather than partly from its own recent history."""
    df = df.sort_values([group_col, sort_col])
    df["traffic_volume_total_lag1"] = df.groupby(group_col)["traffic_volume_total"].shift(1)
    df["no2_pphm_lag1"] = df.groupby(group_col)["no2_pphm"].shift(1)
    df["traffic_volume_total_roll7"] = (
        df.groupby(group_col)["traffic_volume_total"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    return df


def add_hourly_cyclical_features(df):
    """Hour-of-day as a linear 1-24 number implies hour 24 and hour 1 are far
    apart, when they're actually adjacent. Cyclical sin/cos encoding fixes
    that. EDA 1.9: hour_of_day is essential for the hourly model -- the
    traffic-NO2 relationship reverses direction across the day (NOx
    photochemical cycle)."""
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_ending"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_ending"] / 24)
    return df


# ---------------------------------------------------------------------------
# LAG-FEATURE CLEANUP (the only cleanup that has to live here, not in 03 --
# both steps below depend on columns created earlier in THIS script)
# ---------------------------------------------------------------------------

def drop_lag_warmup_rows(df, group_col="station_id"):
    """The lag features (Stage 1) are undefined on each station's very first
    row -- there's no "previous day" to look back to. Rather than inventing a
    value, those rows are dropped: it's exactly one row per station (a
    handful in total), and any imputed value here would be fabricated, not
    estimated from data."""
    lag_cols = [c for c in df.columns if c.endswith("_lag1")]
    before = len(df)
    df = df.dropna(subset=lag_cols)
    dropped = before - len(df)
    print(f"  dropped {dropped} lag warm-up row(s) (first day per station, no prior value to lag from)")
    return df


def assert_no_nans_in_features(df, exclude_cols):
    """Final check: everything except identifier/metadata columns should now
    be NaN-free."""
    check_cols = [c for c in df.columns if c not in exclude_cols]
    nan_counts = df[check_cols].isna().sum()
    remaining = nan_counts[nan_counts > 0]
    if len(remaining):
        print("  WARNING -- NaNs remain in:", dict(remaining))
    else:
        print("  confirmed: zero NaNs in all feature columns")


# station_lat/lon, suburb/lga, weather_source_type, matched_aq_site, aq_match_flag,
# aq_quality_weight, speed_zone_type, speed_zone_match_dist_m are metadata/provenance
# columns, not predictive features -- posted_speed_kmh and aq_distance_km ARE kept as
# real numeric features (see 03_data_preprocessing.py docstring points 2-3).
NON_FEATURE_COLS = [
    "date", "station_id", "station_name", "station_lat", "station_lon",
    "station_suburb", "station_lga", "weather_source_type", "matched_aq_site",
    "aq_match_flag", "aq_quality_weight", "speed_zone_type", "speed_zone_match_dist_m",
    "has_no2_coverage",
]


def finish_lag_cleanup(df):
    print("Post-feature-creation cleanup (only what Stage 1 itself introduced):")
    df = drop_lag_warmup_rows(df)
    assert_no_nans_in_features(df, NON_FEATURE_COLS)
    return df


# ---------------------------------------------------------------------------

def build_daily():
    df = pd.read_csv(os.path.join(IN_DIR, "final_daily.csv"),
                      parse_dates=["date"], dtype={"station_id": str}, low_memory=False)

    df = add_calendar_features(df)
    df = add_road_type_features(df)
    df = add_traffic_features(df)
    df = add_weather_features(df)
    df = add_target_transform(df)
    df = add_lag_features(df)
    df = finish_lag_cleanup(df)

    out_path = os.path.join(OUT_DIR, "features_daily.csv")
    df.to_csv(out_path, index=False)
    print(f"[daily] {len(df)} rows, {len(df.columns)} columns -> {out_path}\n")
    return df


def build_hourly():
    df = pd.read_csv(os.path.join(IN_DIR, "final_hourly.csv"),
                      parse_dates=["date"], dtype={"station_id": str}, low_memory=False)

    df = add_calendar_features(df)
    df = add_road_type_features(df)
    df = add_traffic_features(df)
    df = add_weather_features(df)
    df = add_target_transform(df)
    df = add_hourly_cyclical_features(df)
    df["_sort_key"] = df["date"].astype(str) + "-" + df["hour_ending"].astype(str).str.zfill(2)
    df = add_lag_features(df, sort_col="_sort_key")
    df = df.drop(columns=["_sort_key"])
    df = finish_lag_cleanup(df)

    out_path = os.path.join(OUT_DIR, "features_hourly.csv")
    df.to_csv(out_path, index=False)
    print(f"[hourly] {len(df)} rows, {len(df.columns)} columns -> {out_path}")
    return df


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Building daily features ===")
    build_daily()
    print("=== Building hourly features ===")
    build_hourly()