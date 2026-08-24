"""
02_data_preprocessing.py

Cleans the merged dataset produced by 01_data_ingestion.py
(data/processed/merged_dataset.csv):
- Validates dtypes and checks for duplicates
- Checks for impossible/invalid values (negative volumes, negative CO2, out-of-range weather)
- Reports genuine hourly coverage gaps per station (does NOT fill these in)
- Imputes missing weather values only (flagging every imputed row so it's traceable)

Run from project root:
    python src/02_data_preprocessing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "merged_dataset.csv"
OUTPUT_FILE = PROCESSED_DIR / "cleaned_dataset.csv"

# Plausible ranges for Sydney weather — anything outside this signals a real
# data problem worth investigating, not a normal reading.
TEMP_MIN, TEMP_MAX = -5, 50
RAINFALL_MIN, RAINFALL_MAX = 0, 400


def load_and_validate(path: Path) -> pd.DataFrame:
    print(f"Loading {path}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df):,} rows, {df['station_id'].nunique()} stations, "
          f"{df['timestamp'].min()} to {df['timestamp'].max()}")

    n_dupes = df.duplicated().sum()
    if n_dupes:
        print(f"WARNING: {n_dupes} exact duplicate rows found — dropping.")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found.")

    return df


def check_invalid_values(df: pd.DataFrame) -> None:
    """Report (don't silently fix) impossible values — these indicate a
    bug upstream, not something to patch here."""
    issues = []
    for col in ["volume_light", "volume_heavy", "co2_estimate"]:
        n_negative = (df[col] < 0).sum()
        if n_negative:
            issues.append(f"{n_negative} negative values in {col}")

    n_temp_bad = (~df["max_temp"].between(TEMP_MIN, TEMP_MAX) & df["max_temp"].notna()).sum()
    if n_temp_bad:
        issues.append(f"{n_temp_bad} max_temp values outside plausible range ({TEMP_MIN}-{TEMP_MAX}C)")

    n_rain_bad = (~df["rainfall"].between(RAINFALL_MIN, RAINFALL_MAX) & df["rainfall"].notna()).sum()
    if n_rain_bad:
        issues.append(f"{n_rain_bad} rainfall values outside plausible range ({RAINFALL_MIN}-{RAINFALL_MAX}mm)")

    if issues:
        print("INVALID VALUES FOUND — investigate before proceeding:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No invalid values found (volumes, CO2, temp, rainfall all in range).")


def report_coverage_gaps(df: pd.DataFrame) -> None:
    """Report genuine missing hourly timestamps per station. These are NOT
    filled in — fabricating rows here would create fake data a model would
    learn from as if it were real traffic. This just makes the known gaps
    (documented in DATA.md) visible and quantified from the actual data,
    rather than trusted from memory."""
    print("Hourly coverage gaps per station (documented in DATA.md, not filled here):")
    for station_id, group in df.groupby("station_id"):
        road = group["road"].iloc[0]
        full_range = pd.date_range(group["timestamp"].min(), group["timestamp"].max(), freq="h")
        actual = set(group["timestamp"])
        missing = len(full_range) - len(actual)
        pct_missing = missing / len(full_range) * 100
        print(f"  {station_id} ({road}): {missing:,} missing hours of {len(full_range):,} "
              f"({pct_missing:.1f}%)")


def impute_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing weather values per station, in chronological order,
    using forward-fill then backward-fill. Every imputed row is flagged
    so it stays traceable rather than silently blended into real readings."""
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    df["max_temp_imputed"] = df["max_temp"].isna()
    df["rainfall_imputed"] = df["rainfall"].isna()

    for col in ["max_temp", "rainfall"]:
        n_missing_before = df[col].isna().sum()
        df[col] = df.groupby("station_id")[col].transform(lambda s: s.ffill().bfill())
        n_missing_after = df[col].isna().sum()
        print(f"{col}: {n_missing_before} missing -> {n_missing_after} missing after impute "
              f"({n_missing_before - n_missing_after} filled)")

    return df


def main():
    df = load_and_validate(INPUT_FILE)
    print()

    print("Checking for invalid values...")
    check_invalid_values(df)
    print()

    report_coverage_gaps(df)
    print()

    print("Imputing missing weather values...")
    df = impute_weather(df)
    print()

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df):,} rows to {OUTPUT_FILE}")
    print(f"Imputed rows: {df['max_temp_imputed'].sum()} max_temp, {df['rainfall_imputed'].sum()} rainfall")


if __name__ == "__main__":
    main()