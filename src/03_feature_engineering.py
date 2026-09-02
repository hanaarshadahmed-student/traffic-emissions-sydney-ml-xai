"""
CO2/NO2 Traffic-Emissions Capstone — Feature Engineering + Data Prep Script
=============================================================================
Two stages, run back to back:

STAGE 1 — FEATURE ENGINEERING: derive new columns from the clean preprocessed
data. Every feature added or column dropped here traces back to a specific
EDA finding -- see the comment above each block.

STAGE 2 — DATA PREP: encode categoricals and resolve every remaining missing
value, so the output file has zero NaNs and is genuinely ready to hand to a
model. This is the tail end of feature engineering, not a separate cleaning
pass -- it only exists because of decisions made in Stage 1 (e.g. lag
features create NaNs on each station's first day) or because a raw gap
(EDA Part 0.1) needs a modeling decision, not just a data-quality fix.

Input:  data/processed/preprocessed_daily.csv
        data/processed/preprocessed_hourly.csv
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
# STAGE 2 -- DATA PREP (encoding + imputation, so the output has zero NaNs)
# ---------------------------------------------------------------------------

def drop_redundant_weather_columns(df):
    """temp_max_c/temp_min_c only exist for BOM-sourced stations and were
    already averaged into temp_c during ingestion -- keeping all three would
    just be duplicating information temp_c already carries. Dropped, not
    imputed, since the information isn't missing, it's redundant."""
    return df.drop(columns=[c for c in ("temp_max_c", "temp_min_c") if c in df.columns])


def drop_unused_secondary_pollutants(df):
    """co_ppm and ozone_pphm are secondary pollutants, not the project's
    target (no2_pphm) and not used as a predictor -- dropped rather than
    imputed, since there's no modeling reason to keep them, and co_ppm alone
    is ~25% missing."""
    return df.drop(columns=[c for c in ("co_ppm", "ozone_pphm") if c in df.columns])


def impute_wind(df):
    """wind_speed_ms/wind_dir_deg are structurally absent for BOM (rural)
    stations -- EDA Part 0.1. Imputing with a single global value would
    misleadingly imply "average wind" for stations that never measured wind
    at all, so a has_wind_data flag is added first to preserve that
    distinction for any model or analysis that wants to use it. The actual
    NaNs are then filled with the median of the stations that DO have wind
    data, purely so the column contains no NaNs -- treat has_wind_data as
    the signal, wind_speed_ms/wind_dir_deg as a best-effort fallback value
    for stations without real readings."""
    df["has_wind_data"] = df["wind_speed_ms"].notna().astype(int)
    for col in ("wind_speed_ms", "wind_dir_deg"):
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def impute_remaining_small_gaps(df):
    """temp_c and rain_mm are only ~0.3% missing (EDA 0.1) -- small, close to
    random sensor gaps, not a structural pattern. Filled with the median."""
    for col in ("temp_c", "rain_mm"):
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


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


def one_hot_encode_station(df):
    """station_id is kept as-is (for grouping/reference in later notebooks)
    AND one-hot encoded (prefix stn_) for models that need purely numeric
    input. EDA found real station-level differences (traffic scale, NO2
    baseline, AQ-distance quality) that a model can't otherwise use."""
    dummies = pd.get_dummies(df["station_id"], prefix="stn", dtype=int)
    return pd.concat([df, dummies], axis=1)


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


NON_FEATURE_COLS = [
    "date", "station_id", "station_name", "station_lat", "station_lon",
    "station_suburb", "station_lga", "weather_source_type", "matched_aq_site",
    "has_no2_coverage",
]


def data_prep(df):
    print("Stage 2 -- data prep (encoding + imputation):")
    df = drop_redundant_weather_columns(df)
    df = drop_unused_secondary_pollutants(df)
    df = impute_wind(df)
    df = impute_remaining_small_gaps(df)
    df = drop_lag_warmup_rows(df)
    df = one_hot_encode_station(df)
    assert_no_nans_in_features(df, NON_FEATURE_COLS)
    return df


# ---------------------------------------------------------------------------

def build_daily():
    df = pd.read_csv(os.path.join(IN_DIR, "preprocessed_daily.csv"),
                      parse_dates=["date"], dtype={"station_id": str}, low_memory=False)

    df = add_calendar_features(df)
    df = add_road_type_features(df)
    df = add_traffic_features(df)
    df = add_weather_features(df)
    df = add_target_transform(df)
    df = add_lag_features(df)
    df = data_prep(df)

    out_path = os.path.join(OUT_DIR, "features_daily.csv")
    df.to_csv(out_path, index=False)
    print(f"[daily] {len(df)} rows, {len(df.columns)} columns -> {out_path}\n")
    return df


def build_hourly():
    df = pd.read_csv(os.path.join(IN_DIR, "preprocessed_hourly.csv"),
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
    df = data_prep(df)

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