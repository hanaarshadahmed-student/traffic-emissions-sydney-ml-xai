"""
data_preprocessing.py

Cleans the merged dataset produced by ingestion.py (data/processed/merged_dataset.csv):
- Validates dtypes and checks for duplicates
- Checks for impossible/invalid values (negative volumes, negative CO2, out-of-range weather)
- Imputes missing weather values (flagging every imputed row so it's traceable)
- Leaves genuine coverage gaps (missing timestamps) alone — does not fabricate rows

Run from project root:
    python src/data_preprocessing.py
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
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

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
    print(f"Loading {INPUT_FILE}")
    df = load_and_validate(INPUT_FILE)
    print(f"Loaded {len(df):,} rows\n")

    print("Checking for invalid values...")
    check_invalid_values(df)
    print()

    print("Imputing missing weather values...")
    df = impute_weather(df)
    print()

    # Note: genuine coverage gaps (missing hourly timestamps entirely) are
    # left as-is intentionally — see DATA.md for documented gap percentages
    # per station. These are not filled here.

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df):,} rows to {OUTPUT_FILE}")
    print(f"Imputed rows: {df['max_temp_imputed'].sum()} max_temp, {df['rainfall_imputed'].sum()} rainfall")


if __name__ == "__main__":
    main()