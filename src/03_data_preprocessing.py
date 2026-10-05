"""
CO2/NO2 Traffic-Emissions Capstone -- Stage 03: Preprocessing

NO2 outlier report (reported, not removed), aq_quality_weight for
REVIEW-flagged stations, train-only median imputation, redundant-column
drops and station one-hot encoding. The reasoning behind each decision is
in docs/Data.md ("Data-quality decisions").

Input:  data/processed/preprocessed_daily.csv
        data/processed/preprocessed_hourly.csv
Output: data/processed/final_daily.csv      (zero NaNs in raw columns,
        data/processed/final_hourly.csv      encoded, weighted, ready for
                                              04's feature creation)
"""

import pandas as pd
import os

import sys
from pathlib import Path

# Repo root on the import path, so the shared code in scripts/ is importable
# whichever folder you run this from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paths  # noqa: E402
from scripts.split_utils import train_only_median  # noqa: E402

IN_DIR = str(paths.PROCESSED_DIR)
OUT_DIR = str(paths.PROCESSED_DIR)

REVIEW_WEIGHT = 0.5  # judgement call -- see docs/Data.md "aq_quality_weight". Adjust and justify in your report.


def log(msg):
    print(msg)


# ---------------------------------------------------------------------------
# Outliers + AQ-match quality
# ---------------------------------------------------------------------------

def report_no2_outliers(df, label):
    """Reproduces EDA 2.8's per-station IQR check on the data actually
    reaching this script, so the numbers are traceable to what 04 will
    train on -- NOT used to drop or cap anything, printed cross-check only."""
    def count_outliers(g):
        q1, q3 = g["no2_pphm"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return ((g["no2_pphm"] < lower) | (g["no2_pphm"] > upper)).sum()

    summary = df.groupby("station_id").apply(
        lambda g: pd.Series({"n_rows": len(g), "no2_outliers": count_outliers(g)})
    )
    summary["no2_outlier_pct"] = (summary["no2_outliers"] / summary["n_rows"] * 100).round(1)
    log(f"\n[{label}] NO2 outlier check (IQR, per station -- reported only, not removed):")
    log(summary.sort_values("no2_outlier_pct", ascending=False).to_string())
    return summary


def add_aq_quality_weight(df, label):
    """REVIEW-flagged stations get a lower sample weight rather than being
    dropped. Printed so the decision is visible and citable, not silent."""
    df["aq_quality_weight"] = df["aq_match_flag"].map({"OK": 1.0, "REVIEW": REVIEW_WEIGHT}).fillna(1.0)
    n_review_rows = (df["aq_quality_weight"] < 1.0).sum()
    review_stations = sorted(df.loc[df["aq_quality_weight"] < 1.0, "station_id"].unique())
    log(f"[{label}] aq_quality_weight = {REVIEW_WEIGHT} for {n_review_rows} rows "
        f"across REVIEW-flagged stations {review_stations} (all other rows = 1.0)")
    return df


# ---------------------------------------------------------------------------
# Missing values / redundant columns / encoding
# (raw columns only -- derived features are built in 04)
# ---------------------------------------------------------------------------

def drop_redundant_weather_columns(df):
    """temp_max_c/temp_min_c only exist for BOM-sourced stations and were
    already averaged into temp_c during ingestion -- dropped, not imputed,
    since the information isn't missing, it's redundant."""
    return df.drop(columns=[c for c in ("temp_max_c", "temp_min_c") if c in df.columns])


def drop_unused_secondary_pollutants(df):
    """co_ppm and ozone_pphm are secondary pollutants, not the project's
    target and not used as a predictor -- dropped rather than imputed,
    since co_ppm alone is ~25% missing and there's no modeling reason to
    keep either."""
    return df.drop(columns=[c for c in ("co_ppm", "ozone_pphm") if c in df.columns])


def impute_wind(df, label, time_cols):
    """wind_speed_ms/wind_dir_deg are structurally absent for BOM (rural)
    stations. has_wind_data preserves that distinction; the NaNs are then
    filled with a single value so the column contains no NaNs --
    has_wind_data is the real signal, the filled number is just a
    placeholder for models that can't take NaN.

    That placeholder is the median computed over TRAIN rows only (per
    05_train_test_split.py's global chronological split, via
    split_utils). Using the whole dataset's median here -- including rows
    that will later become val/test -- would leak those rows' distribution
    into every training row's wind features, and from there into every
    rolling/lag feature 04_feature_engineering.py builds on top of them."""
    df["has_wind_data"] = df["wind_speed_ms"].notna().astype(int)
    for col in ("wind_speed_ms", "wind_dir_deg"):
        if col in df.columns:
            fill_value = train_only_median(df, col, time_cols)
            df[col] = df[col].fillna(fill_value)
            log(f"[{label}] {col}: filled NaNs with TRAIN-only median {fill_value:.3f}")
    return df


def impute_remaining_small_gaps(df, label, time_cols):
    """temp_c and rain_mm are only ~0.3% missing -- small, close to random
    sensor gaps, not a structural pattern. Filled with the TRAIN-only
    median (see impute_wind's docstring for why train-only, not global)."""
    for col in ("temp_c", "rain_mm"):
        if col in df.columns:
            fill_value = train_only_median(df, col, time_cols)
            df[col] = df[col].fillna(fill_value)
            log(f"[{label}] {col}: filled NaNs with TRAIN-only median {fill_value:.3f}")
    return df


def one_hot_encode_station(df):
    """station_id is kept as-is (for grouping/reference downstream) AND
    one-hot encoded (prefix stn_) for models that need purely numeric
    input. EDA found real station-level differences (traffic scale, NO2
    baseline, AQ-distance quality) a model can't otherwise use."""
    dummies = pd.get_dummies(df["station_id"], prefix="stn", dtype=int)
    return pd.concat([df, dummies], axis=1)


def clean_raw_columns(df, label, time_cols):
    df = drop_redundant_weather_columns(df)
    df = drop_unused_secondary_pollutants(df)
    df = impute_wind(df, label, time_cols)
    df = impute_remaining_small_gaps(df, label, time_cols)
    df = one_hot_encode_station(df)
    raw_nan_cols = [c for c in ("temp_c", "rain_mm", "wind_speed_ms", "wind_dir_deg") if c in df.columns]
    remaining = df[raw_nan_cols].isna().sum()
    remaining = remaining[remaining > 0]
    if len(remaining):
        log(f"[{label}] WARNING -- NaNs remain after imputation: {dict(remaining)}")
    else:
        log(f"[{label}] confirmed: zero NaNs in raw weather columns after imputation")
    return df


def finalize(df, label, time_cols):
    log(f"\n{'='*60}\nFinal preprocessing [{label}]\n{'='*60}")
    report_no2_outliers(df, label)
    df = add_aq_quality_weight(df, label)
    log(f"[{label}] posted_speed_kmh present: {'posted_speed_kmh' in df.columns} "
        f"(kept as numeric feature -- see docs/Data.md \"Posted speed limit\")")
    df = clean_raw_columns(df, label, time_cols)
    return df


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    daily = pd.read_csv(os.path.join(IN_DIR, "preprocessed_daily.csv"),
                         parse_dates=["date"], dtype={"station_id": str}, low_memory=False)
    daily = finalize(daily, "DAILY", time_cols=["date"])
    daily.to_csv(os.path.join(OUT_DIR, "final_daily.csv"), index=False)
    log(f"\nSaved -> {OUT_DIR}/final_daily.csv ({len(daily)} rows, {len(daily.columns)} columns)")

    hourly = pd.read_csv(os.path.join(IN_DIR, "preprocessed_hourly.csv"),
                          parse_dates=["date"], dtype={"station_id": str}, low_memory=False)
    hourly = finalize(hourly, "HOURLY", time_cols=["date", "hour_ending"])
    hourly.to_csv(os.path.join(OUT_DIR, "final_hourly.csv"), index=False)
    log(f"\nSaved -> {OUT_DIR}/final_hourly.csv ({len(hourly)} rows, {len(hourly.columns)} columns)")

    return daily, hourly


if __name__ == "__main__":
    run()