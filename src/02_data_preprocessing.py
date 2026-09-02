"""
CO2/NO2 Traffic-Emissions Capstone — Preprocessing Script
============================================================
Takes both raw combined datasets from 01_data_ingestion.py and makes the
deliberate cleaning decisions needed before feature engineering / EDA.
Runs the same cleaning logic on both resolutions.

Every decision here is logged and printed so it can be explained/cited
in the report -- nothing is silently dropped.

Inputs:  data/processed/final_combined_dataset_daily.csv
         data/processed/final_combined_dataset_hourly.csv
Outputs: data/processed/preprocessed_daily.csv     (modeling-ready, all stations)
         data/processed/preprocessed_hourly.csv    (modeling-ready, metro-only stations)
         data/processed/excluded_stations_log_daily.csv
         data/processed/excluded_stations_log_hourly.csv
"""

import pandas as pd
import os

OUT_DIR = "data/processed"

# Stations with fewer days of traffic data than this are flagged (not dropped)
# as thin-coverage -- a judgement call, adjust and justify in your report.
MIN_TRAFFIC_DAYS = 90


def log(msg):
    print(msg)


def preprocess(df, label, min_days_col_scale=1):
    """Shared cleaning logic. min_days_col_scale=24 for hourly data, since
    'days of coverage' there means station-hours / 24."""
    excluded_records = []
    log(f"\n{'='*60}\nPreprocessing [{label}]\n{'='*60}")
    log(f"Loaded {len(df)} rows across {df['station_id'].nunique()} stations")

    # ---- Step 1: drop stations with zero NO2 coverage ----
    no_coverage = df.loc[~df["has_no2_coverage"], "station_id"].unique()
    for sid in no_coverage:
        n = len(df[df["station_id"] == sid])
        aq_site = df.loc[df["station_id"] == sid, "matched_aq_site"].iloc[0]
        excluded_records.append({
            "station_id": sid, "reason": "no NO2 coverage at matched AQ site",
            "matched_aq_site": aq_site, "rows_dropped": n,
        })
    before = len(df)
    df = df[df["has_no2_coverage"]].copy()
    log(f"Step 1 -- dropped {len(no_coverage)} stations with no NO2 coverage "
        f"({no_coverage.tolist()}): {before - len(df)} rows removed, {len(df)} remain")

    # ---- Step 2: sanity-check traffic volume ----
    bad_traffic = df["traffic_volume_total"] < 0
    if bad_traffic.any():
        log(f"Step 2 -- WARNING: {bad_traffic.sum()} rows have negative traffic_volume_total, dropping them")
        df = df[~bad_traffic].copy()
    else:
        log("Step 2 -- no negative traffic volumes found, nothing to drop")

    # ---- Step 3: sanity-check temperature ----
    bad_temp = df["temp_c"].notna() & ((df["temp_c"] < -10) | (df["temp_c"] > 50))
    if bad_temp.any():
        log(f"Step 3 -- WARNING: {bad_temp.sum()} rows have temp_c outside [-10, 50]C, setting to NaN")
        df.loc[bad_temp, "temp_c"] = pd.NA
    else:
        log("Step 3 -- no out-of-range temperatures found")

    # ---- Step 4: drop rows with missing NO2 (the target) ----
    before = len(df)
    missing_no2_by_station = (
        df[df["no2_pphm"].isna()].groupby("station_id").size().rename("rows_dropped_missing_no2")
    )
    df_clean = df.dropna(subset=["no2_pphm"]).copy()
    log(f"Step 4 -- dropped {before - len(df_clean)} rows with missing NO2 target: {len(df_clean)} rows remain")
    if len(missing_no2_by_station):
        log("           breakdown by station:")
        for sid, n in missing_no2_by_station.items():
            log(f"             {sid}: {n} rows")

    # ---- Step 5: flag (not drop) stations with thin coverage ----
    unit_counts = df_clean.groupby("station_id").size() / min_days_col_scale
    thin = unit_counts[unit_counts < MIN_TRAFFIC_DAYS]
    if len(thin):
        log(f"Step 5 -- NOTE: {len(thin)} station(s) have under {MIN_TRAFFIC_DAYS} equivalent days of data "
            f"after cleaning ({dict(thin.round(1))}). Kept but flagged.")
    else:
        log(f"Step 5 -- all remaining stations have at least {MIN_TRAFFIC_DAYS} equivalent days of data")

    # ---- Step 6: drop exact duplicate rows ----
    dedup_cols = ["station_id", "date"] + (["hour_ending"] if "hour_ending" in df_clean.columns else [])
    dup_mask = df_clean.duplicated(subset=dedup_cols, keep="first")
    if dup_mask.any():
        log(f"Step 6 -- dropping {dup_mask.sum()} duplicate {dedup_cols} rows")
        df_clean = df_clean[~dup_mask].copy()
    else:
        log(f"Step 6 -- no duplicate {dedup_cols} rows found")

    log(f"\nFinal [{label}] station list:")
    for sid in sorted(df_clean["station_id"].unique()):
        n = len(df_clean[df_clean["station_id"] == sid])
        log(f"  {sid}: {n} rows")

    return df_clean, pd.DataFrame(excluded_records)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    daily_raw = pd.read_csv(os.path.join(OUT_DIR, "final_combined_dataset_daily.csv"),
                             parse_dates=["date"], dtype={"station_id": str}, low_memory=False)
    daily_clean, daily_excluded = preprocess(daily_raw, "DAILY", min_days_col_scale=1)
    daily_clean.to_csv(os.path.join(OUT_DIR, "preprocessed_daily.csv"), index=False)
    daily_excluded.to_csv(os.path.join(OUT_DIR, "excluded_stations_log_daily.csv"), index=False)
    log(f"\nSaved -> {OUT_DIR}/preprocessed_daily.csv ({len(daily_clean)} rows)")

    hourly_raw = pd.read_csv(os.path.join(OUT_DIR, "final_combined_dataset_hourly.csv"),
                              parse_dates=["date"], dtype={"station_id": str}, low_memory=False)
    hourly_clean, hourly_excluded = preprocess(hourly_raw, "HOURLY", min_days_col_scale=24)
    hourly_clean.to_csv(os.path.join(OUT_DIR, "preprocessed_hourly.csv"), index=False)
    hourly_excluded.to_csv(os.path.join(OUT_DIR, "excluded_stations_log_hourly.csv"), index=False)
    log(f"\nSaved -> {OUT_DIR}/preprocessed_hourly.csv ({len(hourly_clean)} rows)")

    return daily_clean, hourly_clean


if __name__ == "__main__":
    run()