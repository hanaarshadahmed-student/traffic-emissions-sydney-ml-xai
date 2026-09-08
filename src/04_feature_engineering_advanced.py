from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TRAFFIC_DIR = ROOT_DIR / "data" / "raw" / "traffic"
STATION_REFERENCE_PATH = TRAFFIC_DIR / "station_reference.csv"

DAILY_INPUT = PROCESSED_DIR / "final_daily.csv"
HOURLY_INPUT = PROCESSED_DIR / "final_hourly.csv"
DAILY_OUTPUT = PROCESSED_DIR / "features_daily_advanced.csv"
HOURLY_OUTPUT = PROCESSED_DIR / "features_hourly_advanced.csv"
MANIFEST_OUTPUT = PROCESSED_DIR / "advanced_feature_manifest.json"

TARGET_COLUMNS = ["no2_pphm", "target_no2_log1p", "target_no2_sqrt"]
SAMPLE_WEIGHT_COLUMN = "aq_quality_weight"
BASE_METADATA_COLUMNS = [
    "date",
    "timestamp",
    "school_holiday",
    "station_id",
    "station_name",
    "station_suburb",
    "station_lga",
    "weather_source_type",
    "matched_aq_site",
    "aq_match_flag",
    "speed_zone_type",
    "road_name",
    "rms_region",
    "device_type",
    "road_on_type",
    "road_classification_type",
    "road_classification_admin",
]


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    numerator_values = numerator.to_numpy(dtype=float)
    denominator_values = denominator.to_numpy(dtype=float)
    output = np.full(len(numerator), default, dtype=float)
    np.divide(
        numerator_values,
        denominator_values,
        out=output,
        where=denominator_values != 0,
    )
    return pd.Series(output, index=numerator.index)


def slug(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return normalized.strip("_") or "unknown"


def add_cyclical_features(
    df: pd.DataFrame,
    column: str,
    period: float,
    prefix: str,
    offset: float = 0.0,
) -> pd.DataFrame:
    angle = 2.0 * np.pi * (df[column].astype(float) - offset) / period
    df[f"{prefix}_sin"] = np.sin(angle)
    df[f"{prefix}_cos"] = np.cos(angle)
    return df


def validate_source(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    required = [
        "date",
        "station_id",
        "station_name",
        "traffic_volume_total",
        "traffic_volume_light",
        "traffic_volume_heavy",
        "public_holiday",
        "school_holiday",
        "temp_c",
        "humidity_pct",
        "wind_speed_ms",
        "wind_dir_deg",
        "rain_mm",
        "no2_pphm",
        "aq_distance_km",
        "posted_speed_kmh",
    ]
    if grain == "hourly":
        required.append("hour_ending")
    require_columns(df, required, grain)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    df["station_id"] = df["station_id"].astype(str)

    key_columns = ["station_id", "date"]
    if grain == "hourly":
        if not df["hour_ending"].between(1, 24).all():
            raise ValueError("hour_ending must be between 1 and 24")
        key_columns.append("hour_ending")

    if df.duplicated(key_columns).any():
        raise ValueError(f"{grain} input contains duplicate time keys")
    if df["no2_pphm"].isna().any():
        raise ValueError(f"{grain} input contains missing target values")
    if (df[["traffic_volume_total", "traffic_volume_light", "traffic_volume_heavy"]] < 0).any().any():
        raise ValueError(f"{grain} input contains negative traffic values")

    traffic_difference = (
        df["traffic_volume_total"]
        - df["traffic_volume_light"]
        - df["traffic_volume_heavy"]
    )
    if not np.allclose(traffic_difference, 0.0, atol=1e-8):
        raise ValueError("traffic total does not equal light plus heavy traffic")

    if grain == "hourly":
        df["_timestamp"] = df["date"] + pd.to_timedelta(df["hour_ending"], unit="h")
    else:
        df["_timestamp"] = df["date"]

    return df.sort_values(["station_id", "_timestamp"]).reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    # Calendar cycles
    timestamp = df["_timestamp"]
    df["calendar_year"] = timestamp.dt.year
    df["calendar_quarter"] = timestamp.dt.quarter
    df["calendar_month"] = timestamp.dt.month
    df["calendar_week"] = timestamp.dt.isocalendar().week.astype(int)
    df["calendar_day_of_year"] = timestamp.dt.dayofyear
    df["calendar_day_of_month"] = timestamp.dt.day
    df["calendar_day_of_week"] = timestamp.dt.dayofweek
    df["is_weekend"] = (df["calendar_day_of_week"] >= 5).astype(np.int8)
    df["is_working_day"] = (
        (df["calendar_day_of_week"] < 5) & df["public_holiday"].eq(0)
    ).astype(np.int8)
    df["days_since_start"] = (timestamp - timestamp.min()).dt.total_seconds() / 86400.0

    df = add_cyclical_features(df, "calendar_month", 12.0, "month", offset=1.0)
    df = add_cyclical_features(df, "calendar_day_of_week", 7.0, "weekday")
    df = add_cyclical_features(df, "calendar_day_of_year", 365.25, "year_day", offset=1.0)

    if grain == "hourly":
        df["hour_of_day"] = df["hour_ending"].mod(24)
        df["hour_of_week"] = df["calendar_day_of_week"] * 24 + df["hour_of_day"]
        df = add_cyclical_features(df, "hour_of_day", 24.0, "hour")
        df = add_cyclical_features(df, "hour_of_week", 168.0, "week_hour")
        df["is_am_peak"] = df["hour_of_day"].between(7, 9).astype(np.int8)
        df["is_pm_peak"] = df["hour_of_day"].between(16, 18).astype(np.int8)
        df["is_peak_hour"] = (df["is_am_peak"] | df["is_pm_peak"]).astype(np.int8)
        df["is_overnight"] = df["hour_of_day"].between(0, 5).astype(np.int8)

    return df


def load_station_metadata() -> pd.DataFrame:
    columns = [
        "station_id",
        "road_name",
        "rms_region",
        "distance_to_intersection",
        "device_type",
        "permanent_station",
        "vehicle_classifier",
        "road_on_type",
        "road_classification_type",
        "road_classification_admin",
    ]
    reference = pd.read_csv(
        STATION_REFERENCE_PATH,
        usecols=columns,
        dtype={"station_id": str},
        low_memory=False,
    )
    reference = reference.drop_duplicates("station_id", keep="first")
    reference = reference.rename(
        columns={"distance_to_intersection": "intersection_distance_m"}
    )
    reference["intersection_distance_m"] = pd.to_numeric(
        reference["intersection_distance_m"], errors="coerce"
    )
    return reference


def add_station_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    reference = load_station_metadata()
    df = df.merge(reference, on="station_id", how="left", validate="many_to_one")

    if df["road_name"].isna().any():
        missing_ids = sorted(df.loc[df["road_name"].isna(), "station_id"].unique())
        raise ValueError(f"station metadata is missing for: {missing_ids}")

    df["intersection_distance_log1p"] = np.log1p(df["intersection_distance_m"].clip(lower=0))
    df["aq_distance_log1p"] = np.log1p(df["aq_distance_km"].clip(lower=0))

    category_columns = [
        "speed_zone_type",
        "weather_source_type",
        "rms_region",
        "device_type",
        "road_on_type",
        "road_classification_type",
        "road_classification_admin",
    ]
    for column in category_columns:
        if column not in df.columns:
            continue
        values = df[column].fillna("Unknown").astype(str)
        dummies = pd.get_dummies(values, dtype=np.int8)
        dummies.columns = [f"{slug(column)}_{slug(value)}" for value in dummies.columns]
        df = pd.concat([df, dummies], axis=1)

    return df


def traffic_file_map() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in TRAFFIC_DIR.glob("*.csv"):
        if " - " not in path.name:
            continue
        station_id = path.name.split(" - ", 1)[0]
        mapping[station_id] = path
    return mapping


def summarize_directional_traffic(
    long_df: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    long_df = (
        long_df.groupby(key_columns + ["cardinal_direction_seq"], as_index=False)["direction_volume"]
        .sum()
    )
    grouped = long_df.groupby(key_columns)["direction_volume"]
    summary = grouped.agg(
        direction_volume_total="sum",
        direction_volume_max="max",
        direction_volume_min="min",
        observed_direction_count="count",
    ).reset_index()
    summary["is_bidirectional_observed"] = (
        summary["observed_direction_count"] >= 2
    ).astype(np.int8)
    summary["dominant_direction_share"] = safe_divide(
        summary["direction_volume_max"], summary["direction_volume_total"]
    )
    summary["directional_imbalance"] = np.where(
        summary["observed_direction_count"] >= 2,
        safe_divide(
            summary["direction_volume_max"] - summary["direction_volume_min"],
            summary["direction_volume_total"],
        ),
        0.0,
    )

    direction_pivot = long_df.pivot_table(
        index=key_columns,
        columns="cardinal_direction_seq",
        values="direction_volume",
        aggfunc="sum",
        fill_value=0.0,
    )
    direction_pivot = direction_pivot.div(direction_pivot.sum(axis=1), axis=0)
    direction_pivot.columns = [f"direction_share_{slug(value)}" for value in direction_pivot.columns]
    direction_pivot = direction_pivot.reset_index()

    summary = summary.drop(columns=["direction_volume_max", "direction_volume_min"])
    return summary.merge(direction_pivot, on=key_columns, how="left", validate="one_to_one")


def load_directional_features(station_ids: list[str], grain: str) -> pd.DataFrame:
    # Directional flow from the raw counters
    file_map = traffic_file_map()
    frames: list[pd.DataFrame] = []

    for station_id in station_ids:
        if station_id not in file_map:
            raise FileNotFoundError(f"traffic source file not found for station {station_id}")

        path = file_map[station_id]
        header = pd.read_csv(path, nrows=0).columns.tolist()
        hour_columns = [column for column in header if re.fullmatch(r"hour_\d{2}", column)]
        use_columns = [
            "date",
            "cardinal_direction_seq",
            "classification_seq",
            *hour_columns,
        ]
        raw = pd.read_csv(path, usecols=use_columns, low_memory=False)
        raw = raw.loc[raw["classification_seq"].eq("All Vehicles")].copy()
        raw["date"] = pd.to_datetime(raw["date"], errors="raise").dt.normalize()
        raw["station_id"] = station_id

        if grain == "daily":
            raw["direction_volume"] = raw[hour_columns].sum(axis=1)
            long_df = raw[
                ["station_id", "date", "cardinal_direction_seq", "direction_volume"]
            ]
            keys = ["station_id", "date"]
        else:
            long_df = raw.melt(
                id_vars=["station_id", "date", "cardinal_direction_seq"],
                value_vars=hour_columns,
                var_name="hour_column",
                value_name="direction_volume",
            )
            long_df["hour_ending"] = (
                long_df["hour_column"].str.removeprefix("hour_").astype(int) + 1
            )
            long_df = long_df.drop(columns="hour_column")
            keys = ["station_id", "date", "hour_ending"]

        frames.append(summarize_directional_traffic(long_df, keys))

    combined = pd.concat(frames, ignore_index=True)
    direction_share_columns = [
        column for column in combined.columns if column.startswith("direction_share_")
    ]
    combined[direction_share_columns] = combined[direction_share_columns].fillna(0.0)
    return combined


def add_directional_features(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    keys = ["station_id", "date"]
    if grain == "hourly":
        keys.append("hour_ending")

    directional = load_directional_features(sorted(df["station_id"].unique()), grain)
    df = df.merge(directional, on=keys, how="left", validate="one_to_one")

    if df["direction_volume_total"].isna().any():
        raise ValueError(f"{grain} directional traffic features are incomplete")
    if not np.allclose(
        df["traffic_volume_total"], df["direction_volume_total"], atol=1e-8
    ):
        raise ValueError(f"{grain} directional traffic totals do not reconcile")

    return df.drop(columns="direction_volume_total")


def add_traffic_features(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    total = df["traffic_volume_total"].astype(float)
    heavy = df["traffic_volume_heavy"].astype(float)

    df["heavy_vehicle_share"] = safe_divide(heavy, total)
    df["traffic_total_log1p"] = np.log1p(total)
    df["traffic_heavy_log1p"] = np.log1p(heavy)
    df["traffic_total_sqrt"] = np.sqrt(total)
    df["traffic_heavy_sqrt"] = np.sqrt(heavy)
    df["has_traffic"] = total.gt(0).astype(np.int8)
    df["has_heavy_traffic"] = heavy.gt(0).astype(np.int8)
    df["traffic_total_x_working_day"] = total * df["is_working_day"]
    df["traffic_heavy_x_working_day"] = heavy * df["is_working_day"]
    df["traffic_total_x_weekend"] = total * df["is_weekend"]

    if grain == "hourly":
        df["traffic_total_x_am_peak"] = total * df["is_am_peak"]
        df["traffic_total_x_pm_peak"] = total * df["is_pm_peak"]
        df["traffic_heavy_x_peak"] = heavy * df["is_peak_hour"]

    for column in ["traffic_volume_total", "traffic_volume_heavy"]:
        grouped = df.groupby("station_id", sort=False)[column]
        prior_mean = grouped.transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )
        prior_std = grouped.transform(
            lambda values: values.shift(1).expanding(min_periods=2).std(ddof=0)
        )
        prefix = column.removeprefix("traffic_volume_")
        df[f"traffic_{prefix}_prior_baseline_available"] = prior_mean.notna().astype(np.int8)
        df[f"traffic_{prefix}_vs_prior_mean"] = safe_divide(
            df[column], prior_mean, default=1.0
        ).where(prior_mean.gt(0), 1.0)
        df[f"traffic_{prefix}_difference_from_prior_mean"] = (
            df[column] - prior_mean
        ).fillna(0.0)
        df[f"traffic_{prefix}_prior_zscore"] = safe_divide(
            df[column] - prior_mean,
            prior_std,
        ).where(prior_std.gt(0), 0.0)

    return df.drop(columns="traffic_volume_light")


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df["humidity_data_available"] = df["humidity_pct"].notna().astype(np.int8)
    df["humidity_pct"] = (
        df.groupby("station_id", sort=False)["humidity_pct"].ffill().fillna(0.0)
    )

    wind_available = df.get("has_wind_data", pd.Series(1, index=df.index)).eq(1)
    df["wind_speed_ms"] = df["wind_speed_ms"].where(wind_available, 0.0)
    wind_direction = df["wind_dir_deg"].where(wind_available, 0.0).mod(360.0)
    wind_angle = np.deg2rad(wind_direction)
    df["wind_direction_sin"] = np.where(wind_available, np.sin(wind_angle), 0.0)
    df["wind_direction_cos"] = np.where(wind_available, np.cos(wind_angle), 0.0)
    df["wind_u_ms"] = np.where(
        wind_available,
        -df["wind_speed_ms"] * np.sin(wind_angle),
        0.0,
    )
    df["wind_v_ms"] = np.where(
        wind_available,
        -df["wind_speed_ms"] * np.cos(wind_angle),
        0.0,
    )
    df["is_calm_wind"] = (
        wind_available & df["wind_speed_ms"].le(0.5)
    ).astype(np.int8)
    df["is_rainy"] = df["rain_mm"].gt(1.0).astype(np.int8)
    df["is_heavy_rain"] = df["rain_mm"].gt(10.0).astype(np.int8)
    df["rain_log1p"] = np.log1p(df["rain_mm"].clip(lower=0))
    df["temperature_squared"] = df["temp_c"].pow(2)
    df["wind_speed_squared"] = df["wind_speed_ms"].pow(2)
    df["temperature_x_humidity"] = df["temp_c"] * df["humidity_pct"]
    df["traffic_dispersion_proxy"] = safe_divide(
        df["traffic_total_log1p"], 1.0 + df["wind_speed_ms"]
    )
    df["heavy_traffic_dispersion_proxy"] = safe_divide(
        df["traffic_heavy_log1p"], 1.0 + df["wind_speed_ms"]
    )
    df["traffic_total_x_rainy"] = df["traffic_volume_total"] * df["is_rainy"]
    df["traffic_total_x_temperature"] = df["traffic_total_log1p"] * df["temp_c"]
    return df.drop(columns="wind_dir_deg")


def add_exact_lag(
    df: pd.DataFrame,
    value_column: str,
    delta: pd.Timedelta,
    suffix: str,
) -> pd.DataFrame:
    feature_name = f"{value_column}_lag_{suffix}"
    source = df[["station_id", "_timestamp", value_column]].copy()
    source["_timestamp"] = source["_timestamp"] + delta
    source = source.rename(columns={value_column: feature_name})
    df = df.merge(
        source,
        on=["station_id", "_timestamp"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    df[f"{feature_name}_available"] = df[feature_name].notna().astype(np.int8)
    df[feature_name] = df[feature_name].fillna(0.0)
    return df


def add_exact_lag_features(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    # Exact timestamp lags preserve missing intervals
    if grain == "daily":
        specifications = {
            "traffic_volume_total": [(1, "1d"), (7, "7d"), (14, "14d")],
            "traffic_volume_heavy": [(1, "1d"), (7, "7d")],
            "no2_pphm": [(1, "1d"), (7, "7d")],
            "temp_c": [(1, "1d"), (7, "7d")],
            "wind_speed_ms": [(1, "1d")],
            "rain_mm": [(1, "1d")],
        }
        unit = "D"
    else:
        specifications = {
            "traffic_volume_total": [(1, "1h"), (3, "3h"), (24, "24h"), (168, "168h")],
            "traffic_volume_heavy": [(1, "1h"), (24, "24h"), (168, "168h")],
            "no2_pphm": [(1, "1h"), (3, "3h"), (24, "24h"), (168, "168h")],
            "temp_c": [(1, "1h"), (24, "24h")],
            "wind_speed_ms": [(1, "1h"), (24, "24h")],
            "rain_mm": [(1, "1h"), (24, "24h")],
        }
        unit = "h"

    for value_column, lags in specifications.items():
        for amount, suffix in lags:
            df = add_exact_lag(
                df,
                value_column,
                pd.to_timedelta(amount, unit=unit),
                suffix,
            )

    short_suffix = "1d" if grain == "daily" else "1h"
    long_suffix = "7d" if grain == "daily" else "24h"
    traffic_short = f"traffic_volume_total_lag_{short_suffix}"
    no2_short = f"no2_pphm_lag_{short_suffix}"
    no2_long = f"no2_pphm_lag_{long_suffix}"

    df[f"traffic_total_change_from_{short_suffix}"] = np.where(
        df[f"{traffic_short}_available"].eq(1),
        df["traffic_volume_total"] - df[traffic_short],
        0.0,
    )
    df[f"traffic_total_ratio_to_{short_suffix}"] = np.where(
        df[f"{traffic_short}_available"].eq(1),
        safe_divide(df["traffic_volume_total"], df[traffic_short], default=1.0),
        1.0,
    )
    df[f"no2_prior_change_{short_suffix}_to_{long_suffix}"] = np.where(
        df[f"{no2_short}_available"].eq(1) & df[f"{no2_long}_available"].eq(1),
        df[no2_short] - df[no2_long],
        0.0,
    )
    return df


def rolling_by_time(
    df: pd.DataFrame,
    value_column: str,
    window: str,
    aggregation: str,
    min_periods: int,
) -> pd.Series:
    output = pd.Series(np.nan, index=df.index, dtype=float)
    for indices in df.groupby("station_id", sort=False).groups.values():
        group = df.loc[indices].sort_values("_timestamp")
        series = pd.Series(
            group[value_column].to_numpy(dtype=float),
            index=pd.DatetimeIndex(group["_timestamp"]),
        )
        rolling = series.rolling(window=window, closed="left", min_periods=min_periods)
        if aggregation == "std":
            values = rolling.std(ddof=0)
        else:
            values = getattr(rolling, aggregation)()
        output.loc[group.index] = values.to_numpy()
    return output


def add_rolling_features(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    # Past-only time windows
    if grain == "daily":
        specifications = [
            ("traffic_volume_total", "7D", "mean", 2),
            ("traffic_volume_total", "7D", "std", 2),
            ("traffic_volume_total", "28D", "mean", 7),
            ("traffic_volume_heavy", "7D", "mean", 2),
            ("no2_pphm", "7D", "mean", 2),
            ("no2_pphm", "7D", "std", 2),
            ("no2_pphm", "28D", "mean", 7),
            ("temp_c", "7D", "mean", 2),
            ("wind_speed_ms", "7D", "mean", 2),
            ("rain_mm", "7D", "sum", 2),
        ]
    else:
        specifications = [
            ("traffic_volume_total", "24h", "mean", 3),
            ("traffic_volume_total", "24h", "std", 3),
            ("traffic_volume_total", "168h", "mean", 24),
            ("traffic_volume_total", "168h", "std", 24),
            ("traffic_volume_heavy", "24h", "mean", 3),
            ("no2_pphm", "24h", "mean", 3),
            ("no2_pphm", "24h", "std", 3),
            ("no2_pphm", "168h", "mean", 24),
            ("temp_c", "24h", "mean", 3),
            ("temp_c", "24h", "std", 3),
            ("wind_speed_ms", "24h", "mean", 3),
            ("rain_mm", "24h", "sum", 3),
        ]

    for value_column, window, aggregation, min_periods in specifications:
        window_name = window.lower()
        feature_name = f"{value_column}_rolling_{window_name}_{aggregation}"
        values = rolling_by_time(
            df,
            value_column,
            window,
            aggregation,
            min_periods,
        )
        df[f"{feature_name}_available"] = values.notna().astype(np.int8)
        df[feature_name] = values.fillna(0.0)

    return df


def add_target_transforms(df: pd.DataFrame) -> pd.DataFrame:
    clipped = df["no2_pphm"].clip(lower=0.0)
    df["target_no2_log1p"] = np.log1p(clipped)
    df["target_no2_sqrt"] = np.sqrt(clipped)
    return df


def drop_constant_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    protected = set(TARGET_COLUMNS + [SAMPLE_WEIGHT_COLUMN])
    candidates = df.select_dtypes(include=[np.number, "bool"]).columns
    dropped = [
        column
        for column in candidates
        if column not in protected and df[column].nunique(dropna=False) <= 1
    ]
    return df.drop(columns=dropped), dropped


def validate_output(df: pd.DataFrame, grain: str) -> None:
    keys = ["station_id", "date"]
    if grain == "hourly":
        keys.append("hour_ending")
    if df.duplicated(keys).any():
        raise ValueError(f"{grain} output contains duplicate time keys")

    numeric = df.select_dtypes(include=[np.number, "bool"])
    if numeric.isna().any().any():
        columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"{grain} output contains numeric NaNs: {columns}")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError(f"{grain} output contains infinite values")


def build_dataset(input_path: Path, output_path: Path, grain: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, dtype={"station_id": str}, low_memory=False)
    df = validate_source(df, grain)
    df = add_calendar_features(df, grain)
    df = add_station_metadata_features(df)
    df = add_directional_features(df, grain)
    df = add_traffic_features(df, grain)
    df = add_weather_features(df)
    df = add_exact_lag_features(df, grain)
    df = add_rolling_features(df, grain)
    df = add_target_transforms(df)
    df["timestamp"] = df["_timestamp"]
    df = df.drop(columns="_timestamp")
    df, dropped = drop_constant_numeric_columns(df)
    validate_output(df, grain)
    df.to_csv(output_path, index=False)
    print(
        f"{grain}: {len(df):,} rows, {len(df.columns):,} columns, "
        f"dropped {len(dropped)} constant columns -> {output_path.name}"
    )
    return df


def feature_lists(df: pd.DataFrame) -> dict[str, list[str]]:
    metadata = [column for column in BASE_METADATA_COLUMNS if column in df.columns]
    excluded = set(metadata + TARGET_COLUMNS + [SAMPLE_WEIGHT_COLUMN])
    all_features = [
        column
        for column in df.select_dtypes(include=[np.number, "bool"]).columns
        if column not in excluded
    ]
    autoregressive = [
        column
        for column in all_features
        if column.startswith("no2_pphm_lag_")
        or column.startswith("no2_pphm_rolling_")
        or column.startswith("no2_prior_change_")
    ]
    exogenous = [column for column in all_features if column not in autoregressive]
    return {
        "target_columns": [column for column in TARGET_COLUMNS if column in df.columns],
        "sample_weight_column": [SAMPLE_WEIGHT_COLUMN],
        "metadata_columns": metadata,
        "exogenous_features": exogenous,
        "autoregressive_features": autoregressive,
        "all_candidate_features": all_features,
    }


def write_manifest(datasets: dict[str, pd.DataFrame]) -> None:
    manifest = {
        grain: {
            "rows": len(df),
            "columns": len(df.columns),
            **feature_lists(df),
        }
        for grain, df in datasets.items()
    }
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest -> {MANIFEST_OUTPUT.name}")


def run(grain: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, pd.DataFrame] = {}

    if grain in {"daily", "all"}:
        datasets["daily"] = build_dataset(DAILY_INPUT, DAILY_OUTPUT, "daily")
    if grain in {"hourly", "all"}:
        datasets["hourly"] = build_dataset(HOURLY_INPUT, HOURLY_OUTPUT, "hourly")

    write_manifest(datasets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grain", choices=["daily", "hourly", "all"], default="all")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().grain)
