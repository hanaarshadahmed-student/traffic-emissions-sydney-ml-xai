"""
Preprocessing: fills missing values (train-only medians), adds quality weights for flagged stations, drops redundant columns and one-hot encodes stations, ready for feature engineering. 

Usage:
    python src/03_data_preprocessing.py
    python run_pipeline.py --stage preprocess
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

REVIEW_WEIGHT = 0.5  # judgement call -- see data/Data.md "aq_quality_weight". Adjust and justify in your report.


def log(msg):
    print(msg)


# ---------------------------------------------------------------------------
# Outliers + AQ-match quality
# ---------------------------------------------------------------------------

def report_no2_outliers(df, label):
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
    df["aq_quality_weight"] = df["aq_match_flag"].map({"OK": 1.0, "REVIEW": REVIEW_WEIGHT}).fillna(1.0)
    n_review_rows = (df["aq_quality_weight"] < 1.0).sum()
    review_stations = sorted(df.loc[df["aq_quality_weight"] < 1.0, "station_id"].unique())
    log(f"[{label}] aq_quality_weight = {REVIEW_WEIGHT} for {n_review_rows} rows "
        f"across REVIEW-flagged stations {review_stations} (all other rows = 1.0)")
    return df


# ---------------------------------------------------------------------------
# Missing values / redundant columns / encoding
# ---------------------------------------------------------------------------

def drop_redundant_weather_columns(df):
    return df.drop(columns=[c for c in ("temp_max_c", "temp_min_c") if c in df.columns])


def drop_unused_secondary_pollutants(df):
    return df.drop(columns=[c for c in ("co_ppm", "ozone_pphm") if c in df.columns])


def impute_wind(df, label, time_cols):
    df["has_wind_data"] = df["wind_speed_ms"].notna().astype(int)
    for col in ("wind_speed_ms", "wind_dir_deg"):
        if col in df.columns:
            fill_value = train_only_median(df, col, time_cols)
            df[col] = df[col].fillna(fill_value)
            log(f"[{label}] {col}: filled NaNs with TRAIN-only median {fill_value:.3f}")
    return df


def impute_remaining_small_gaps(df, label, time_cols):
    for col in ("temp_c", "rain_mm"):
        if col in df.columns:
            fill_value = train_only_median(df, col, time_cols)
            df[col] = df[col].fillna(fill_value)
            log(f"[{label}] {col}: filled NaNs with TRAIN-only median {fill_value:.3f}")
    return df


def one_hot_encode_station(df):
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
        f"(kept as numeric feature -- see data/Data.md \"Posted speed limit\")")
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