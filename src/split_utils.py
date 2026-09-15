"""
CO2/NO2 Traffic-Emissions Capstone -- Shared chronological split logic

Both 03_data_preprocessing.py and 05_train_test_split.py need to know
which rows belong to "train" for a given station, and they MUST agree:

- 05 uses it to make the actual train/val/test files and to fit the
  StandardScaler on train only.
- 03 needs it too, earlier in the pipeline, to fit imputation values
  (medians for wind/temp/rain) on train-only rows -- otherwise the
  median baked into the imputed values would be computed across rows
  that later become val/test, leaking their distribution back into
  training features (and, downstream, into every rolling/lag feature
  04 builds on top of temp_c/wind_speed_ms/rain_mm).

Importing one function in both places means there's exactly one
definition of "train" to keep in sync, instead of two copies that could
silently drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test gets the remainder (~0.15)


def assign_chronological_split(df: pd.DataFrame, time_cols) -> pd.Series:
    """Per-station chronological split. Each station's own rows are sorted
    by time and cut at the train/val boundary and the val/test boundary
    independently -- a station observed for 6 months and one observed for
    2 years both end up with ~70/15/15 of their own timeline in each
    split, rather than one global date cutoff which would leave short-
    history stations entirely in one split.

    time_cols: a column name or list of column names to sort by within
    each station (e.g. "date" for daily data, ["date", "hour_ending"]
    for hourly, where hour_ending alone isn't a full ordering).
    """
    if isinstance(time_cols, str):
        time_cols = [time_cols]
    sort_cols = ["station_id"] + time_cols
    split = pd.Series(index=df.index, dtype=object)
    for station_id, group in df.sort_values(sort_cols).groupby("station_id"):
        n = len(group)
        train_end = int(np.floor(n * TRAIN_FRAC))
        val_end = train_end + int(np.floor(n * VAL_FRAC))
        labels = np.full(n, "test", dtype=object)
        labels[:train_end] = "train"
        labels[train_end:val_end] = "val"
        split.loc[group.index] = labels
    return split


def train_only_median(df: pd.DataFrame, column: str, time_cols) -> float:
    """Median of `column` computed over the TRAIN rows only (per the same
    per-station chronological split used downstream), so an imputed value
    never carries information from a row that ends up in val/test."""
    split = assign_chronological_split(df, time_cols)
    return df.loc[split.eq("train"), column].median()
