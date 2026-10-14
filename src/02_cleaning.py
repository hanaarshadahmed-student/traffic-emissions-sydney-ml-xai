"""
Data cleaning: removes stations and rows that can't be used (exclusions set in config.yaml) from the daily and hourly datasets, and logs every decision so it can be reported.

Usage:
    python src/02_cleaning.py
"""

import argparse
import os

import pandas as pd
import yaml

import sys
from pathlib import Path

# Repo root on the import path, so the shared code in scripts/ is importable
# whichever folder you run this from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paths  # noqa: E402
from scripts.stations import station_label  # noqa: E402

OUT_DIR = str(paths.PROCESSED_DIR)


def load_station_exclusions(config_path=paths.CONFIG_PATH) -> dict:
    
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        cfg = (yaml.safe_load(f) or {}).get("data", {}).get("station_exclusions") or {}
    if not cfg.get("enabled", False):
        return {}
    return {str(k): v for k, v in (cfg.get("stations") or {}).items()}

def load_clip_negative_no2(config_path=paths.CONFIG_PATH) -> bool:
    
    if not os.path.exists(config_path):
        return False
    with open(config_path) as f:
        return bool(((yaml.safe_load(f) or {}).get("data") or {}).get("clip_negative_no2", False))


# Stations with fewer days of traffic data than this are flagged (not dropped)
# as thin-coverage -- a judgement call, adjust and justify in your report.
MIN_TRAFFIC_DAYS = 90


def log(msg):
    print(msg)


def log_thin_hours(df):
    coverage = df.groupby("hour_ending")["no2_pphm"].apply(lambda s: s.notna().mean())
    thin = coverage[coverage < 0.6 * coverage.median()]
    if len(thin):
        log("Step 4 -- NOTE: NO2 is missing far more often at hour ending "
            + ", ".join(f"{h:02d}:00 ({c:.0%} of rows have NO2 vs {coverage.median():.0%} typical)"
                        for h, c in thin.items())
            + " -- a network-wide gap at that hour; values kept, few rows there")


def preprocess(df, label, min_days_col_scale=1, exclusions=None, clip_negative_no2=False):
    excluded_records = []
    steps = []  # one row per cleaning step -> data_quality_<grain>.csv (report table)

    def record(step, description, before, after_df):
        steps.append({"step": step, "description": description, "rows_before": before,
                      "rows_after": len(after_df), "rows_removed": before - len(after_df),
                      "stations_after": after_df["station_id"].nunique()})

    log(f"\n{'='*60}\nPreprocessing [{label}]\n{'='*60}")
    log(f"Loaded {len(df)} rows across {df['station_id'].nunique()} stations")
    record("0", "combined traffic + weather + NO2 data (all candidate stations)", len(df), df)

    # ---- Step 1: drop stations with zero NO2 coverage ----
    no_coverage = df.loc[~df["has_no2_coverage"], "station_id"].unique()
    for sid in no_coverage:
        n = len(df[df["station_id"] == sid])
        aq_site = df.loc[df["station_id"] == sid, "matched_aq_site"].iloc[0]
        excluded_records.append({
            "station_id": sid, "station": station_label(sid), "reason": "no NO2 coverage at matched AQ site",
            "matched_aq_site": aq_site, "rows_dropped": n,
        })
    before = len(df)
    df = df[df["has_no2_coverage"]].copy()
    log(f"Step 1 -- dropped {len(no_coverage)} stations with no NO2 coverage "
        f"({no_coverage.tolist()}): {before - len(df)} rows removed, {len(df)} remain")
    record("1", f"drop stations whose monitor has no NO2 ({', '.join(no_coverage.tolist()) or 'none'})", before, df)

    # ---- Step 1b: data-quality exclusions (config.yaml: data.station_exclusions) ----
    exclusions = exclusions or {}
    to_drop = [sid for sid in exclusions if sid in set(df["station_id"])]
    for sid in to_drop:
        excluded_records.append({
            "station_id": sid, "station": station_label(sid), "reason": exclusions[sid],
            "matched_aq_site": df.loc[df["station_id"] == sid, "matched_aq_site"].iloc[0],
            "rows_dropped": int((df["station_id"] == sid).sum()),
        })
    before = len(df)
    df = df[~df["station_id"].isin(to_drop)].copy()
    record("1b", f"drop stations excluded for data quality ({', '.join(to_drop) or 'none'})", before, df)
    if exclusions:
        log(f"Step 1b -- dropped {len(to_drop)} station(s) per config.yaml data.station_exclusions "
            f"({to_drop}): {before - len(df)} rows removed, {len(df)} remain")
    else:
        log("Step 1b -- station exclusions DISABLED (sensitivity run): all stations kept")

    # ---- Step 2: sanity-check traffic volume ----
    bad_traffic = df["traffic_volume_total"] < 0
    before = len(df)
    if bad_traffic.any():
        log(f"Step 2 -- WARNING: {bad_traffic.sum()} rows have negative traffic_volume_total, dropping them")
        df = df[~bad_traffic].copy()
    else:
        log("Step 2 -- no negative traffic volumes found, nothing to drop")
    record("2", "drop rows with negative traffic counts", before, df)

    # ---- Step 3: sanity-check temperature ----
    bad_temp = df["temp_c"].notna() & ((df["temp_c"] < -10) | (df["temp_c"] > 50))
    if bad_temp.any():
        log(f"Step 3 -- WARNING: {bad_temp.sum()} rows have temp_c outside [-10, 50]C, setting to NaN")
        df.loc[bad_temp, "temp_c"] = pd.NA
    else:
        log("Step 3 -- no out-of-range temperatures found")
    record("3", f"temperatures outside -10..50 C set to missing ({int(bad_temp.sum())} values; rows kept)",
           len(df), df)

    # ---- Step 4: drop rows with missing NO2 (the target) ----
    before = len(df)
    missing_no2_by_station = (
        df[df["no2_pphm"].isna()].groupby("station_id").size().rename("rows_dropped_missing_no2")
    )
    df_clean = df.dropna(subset=["no2_pphm"]).copy()
    log(f"Step 4 -- dropped {before - len(df_clean)} rows with missing NO2 target: {len(df_clean)} rows remain")
    record("4", "drop rows with no NO2 reading (the target is never filled in)", before, df_clean)
    if "hour_ending" in df.columns:
        log_thin_hours(df)
    if len(missing_no2_by_station):
        log("           breakdown by station:")
        for sid, n in missing_no2_by_station.items():
            log(f"             {sid}: {n} rows")

    # ---- Step 4b: negative NO2 readings (kept unless clip_negative_no2) ----
    negative = df_clean["no2_pphm"] < 0
    if clip_negative_no2:
        log(f"Step 4b -- CLIPPED {negative.sum()} negative NO2 values to 0 "
            f"(data.clip_negative_no2: true -- sensitivity run)")
        df_clean.loc[negative, "no2_pphm"] = 0.0
    else:
        by_station = df_clean.loc[negative].groupby("station_id").size()
        log(f"Step 4b -- kept {negative.sum()} negative NO2 values ({negative.mean():.1%} of rows; "
            f"monitor noise near zero -- see data/Data.md): {by_station.to_dict()}")
    record("4b", f"negative NO2 readings {'set to 0' if clip_negative_no2 else 'kept'} "
                 f"({int(negative.sum())} values, {negative.mean():.1%} of rows)", len(df_clean), df_clean)

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
    before = len(df_clean)
    if dup_mask.any():
        log(f"Step 6 -- dropping {dup_mask.sum()} duplicate {dedup_cols} rows")
        df_clean = df_clean[~dup_mask].copy()
    else:
        log(f"Step 6 -- no duplicate {dedup_cols} rows found")
    record("6", "drop duplicate station + time rows", before, df_clean)

    log(f"\nFinal [{label}] station list:")
    for sid in sorted(df_clean["station_id"].unique()):
        n = len(df_clean[df_clean["station_id"] == sid])
        log(f"  {station_label(sid):38s} {n} rows")

    steps_table = pd.DataFrame(steps)
    steps_table.insert(0, "grain", label.lower())
    return df_clean, pd.DataFrame(excluded_records), steps_table


# raw variables reported in missing_data_<grain>.csv, and how each gap is handled
MISSING_HANDLING = {
    "no2_pphm": "row dropped (02_cleaning.py) -- the target is never filled in",
    "traffic_volume_total": "kept as recorded; days/hours with no count are not in the traffic file",
    "traffic_volume_heavy": "kept as recorded",
    "temp_c": "filled with the train-only median (03_data_preprocessing.py)",
    "rain_mm": "filled with the train-only median; monitors without a rain gauge borrow rain "
               "from the nearest monitor that has one (01_data_ingestion.py)",
    "wind_speed_ms": "filled with the train-only median + has_wind_data flag; stations with no wind "
                     "sensor at all get 0 + flag, and traffic/wind uses the typical wind there",
    "wind_dir_deg": "filled with the train-only median + has_wind_data flag",
    "humidity_pct": "carried forward within the station, else 0 + humidity_data_available flag "
                    "(04_feature_engineering.py)",
}


def missing_data_table(raw, clean, label, exclusions):
    kept = raw[raw["has_no2_coverage"] & ~raw["station_id"].isin(list(exclusions or {}))]
    rows = []
    for column, handling in MISSING_HANDLING.items():
        if column not in raw.columns:
            continue
        rows.append({"grain": label.lower(), "variable": column,
                     "pct_missing_before_cleaning": round(100 * kept[column].isna().mean(), 2),
                     "pct_missing_after_cleaning": round(100 * clean[column].isna().mean(), 2),
                     "how_handled": handling})
    return pd.DataFrame(rows)


def run(config_path=paths.CONFIG_PATH):
    os.makedirs(OUT_DIR, exist_ok=True)
    exclusions = load_station_exclusions(config_path)
    clip = load_clip_negative_no2(config_path)

    daily_raw = pd.read_csv(os.path.join(OUT_DIR, "final_combined_dataset_daily.csv"),
                             parse_dates=["date"], dtype={"station_id": str}, low_memory=False)
    daily_clean, daily_excluded, daily_steps = preprocess(daily_raw, "DAILY", min_days_col_scale=1, exclusions=exclusions,
                                               clip_negative_no2=clip)
    daily_clean.to_csv(os.path.join(OUT_DIR, "preprocessed_daily.csv"), index=False)
    daily_excluded.to_csv(os.path.join(OUT_DIR, "excluded_stations_log_daily.csv"), index=False)
    daily_steps.to_csv(os.path.join(OUT_DIR, "data_quality_daily.csv"), index=False)
    missing_data_table(daily_raw, daily_clean, "DAILY", exclusions).to_csv(
        os.path.join(OUT_DIR, "missing_data_daily.csv"), index=False)
    log(f"\nSaved -> {OUT_DIR}/preprocessed_daily.csv ({len(daily_clean)} rows)")

    hourly_raw = pd.read_csv(os.path.join(OUT_DIR, "final_combined_dataset_hourly.csv"),
                              parse_dates=["date"], dtype={"station_id": str}, low_memory=False)
    hourly_clean, hourly_excluded, hourly_steps = preprocess(hourly_raw, "HOURLY", min_days_col_scale=24, exclusions=exclusions,
                                                 clip_negative_no2=clip)
    hourly_clean.to_csv(os.path.join(OUT_DIR, "preprocessed_hourly.csv"), index=False)
    hourly_excluded.to_csv(os.path.join(OUT_DIR, "excluded_stations_log_hourly.csv"), index=False)
    hourly_steps.to_csv(os.path.join(OUT_DIR, "data_quality_hourly.csv"), index=False)
    missing_data_table(hourly_raw, hourly_clean, "HOURLY", exclusions).to_csv(
        os.path.join(OUT_DIR, "missing_data_hourly.csv"), index=False)
    log(f"\nSaved -> {OUT_DIR}/preprocessed_hourly.csv ({len(hourly_clean)} rows)")

    return daily_clean, hourly_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=paths.CONFIG_PATH)
    run(parser.parse_args().config)