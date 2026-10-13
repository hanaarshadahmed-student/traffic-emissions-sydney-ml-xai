"""
CO2/NO2 Traffic-Emissions Capstone -- Public-holiday natural experiment

Traffic and time of day move together (rush hour), so the models can't
fully tell them apart (see experiments/03_ablation.py). Weekday public holidays break
that link: the clock and the calendar look like a normal weekday, but much
less traffic is on the road. If NO2 also drops on those days -- at the same
station, hour and time of year, in similar weather -- that is evidence that
traffic itself drives NO2, not just the time of day.

Design
  holiday days   weekday (Mon-Fri) public holidays
  control days   normal working weekdays within +-14 days of a holiday,
                 same station (so same season, same monitor)
  1. raw comparison per station: average traffic and NO2, holiday vs control
  2. adjusted estimate (linear regression on holiday + control rows):
        NO2 = holiday effect + wind speed + temperature + humidity + rain
              + station + hour of day (hourly) + month
     and the same for log traffic, so the holiday effect on traffic (%) and
     on NO2 (pphm and %) are measured the same way. 95% intervals come from
     resampling whole holiday windows (a holiday and its control days
     together), 500 times.
  3. hourly profile (hourly data): holiday minus control, by hour of day, for
     traffic and NO2 -- does the morning peak shrink when traffic does?

Caveats (state them in the report): holidays are few (about 20 weekday
holidays in 2024-2025) and cluster around Christmas / Easter, when other
things also change (people away, industry closed), so this supports rather
than proves a traffic effect.

Reads data/processed/features_<grain>.csv (unscaled). Nothing is trained.

Outputs (results/experiments/holiday/):
  holiday_<grain>_by_station.csv   raw averages per station, holiday vs control
  holiday_<grain>_effect.csv       adjusted holiday effect on traffic and NO2, with 95% CI
  holiday_hourly_profile.png/.csv  holiday minus control by hour of day
  holiday_<grain>_dose_response.png/.csv
                                   per station: traffic change vs NO2 change,
                                   with the correlation (bigger traffic drop ->
                                   bigger NO2 drop = dose-response)

Usage:
    python experiments/04_holiday_check.py                 # both grains
    python experiments/04_holiday_check.py --grain hourly
Run AFTER the pipeline's data stages (needs data/processed/).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paths  # noqa: E402
from scripts.stations import add_station_name, station_label  # noqa: E402

OUT_DIR = paths.HOLIDAY_DIR
WINDOW_DAYS = 14
WEATHER = ["wind_speed_ms", "temp_c", "humidity_pct", "rain_mm"]
NEAR_MONITOR_KM = 2.0   # dose-response: stations this close to their NO2 monitor drawn solid


def load(grain: str) -> pd.DataFrame:
    path = paths.PROCESSED_DIR / f"features_{grain}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run the pipeline's data stages first.")
    cols = ["station_id", "date", "public_holiday", "calendar_day_of_week", "traffic_volume_total",
            "no2_pphm", "has_wind_data", *WEATHER] + (["hour_of_day"] if grain == "hourly" else [])
    df = pd.read_csv(path, usecols=lambda c: c in cols, dtype={"station_id": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["calendar_day_of_week"] < 5].copy()          # weekdays only
    holidays = np.sort(df.loc[df["public_holiday"] == 1, "date"].unique())
    if len(holidays) == 0:
        return df.iloc[0:0]
    # window id = the nearest weekday holiday; keep holidays + control days within +-14 days
    dates = df["date"].to_numpy()
    nearest = np.abs(dates[:, None] - holidays[None, :]).argmin(axis=1)
    gap = np.abs(dates - holidays[nearest]) / np.timedelta64(1, "D")
    df["window"] = nearest
    df = df[gap <= WINDOW_DAYS].copy()
    df["holiday"] = (df["public_holiday"] == 1).astype(float)
    df["month"] = df["date"].dt.month
    return df


def design(df: pd.DataFrame, grain: str) -> tuple[np.ndarray, list[str]]:
    """Columns: holiday, weather, station / hour / month dummies, intercept."""
    parts = [df[["holiday"] + WEATHER + ["has_wind_data"]].astype(float)]
    for col in ["station_id", "month"] + (["hour_of_day"] if grain == "hourly" else []):
        parts.append(pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.std() > 0]
    X.insert(0, "intercept", 1.0)
    return X.to_numpy(), list(X.columns)


def holiday_coef(X: np.ndarray, y: np.ndarray, holiday_col: int) -> float:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[holiday_col])


def effects(df: pd.DataFrame, grain: str, n_boot: int, seed: int) -> pd.DataFrame:
    X, names = design(df, grain)
    h = names.index("holiday")
    targets = {"traffic (log change)": np.log1p(df["traffic_volume_total"].to_numpy(float)),
               "NO2 (pphm)": df["no2_pphm"].to_numpy(float)}
    windows = df["window"].to_numpy()
    unique = np.unique(windows)
    members = {w: np.flatnonzero(windows == w) for w in unique}
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in targets}
    for _ in range(n_boot):
        idx = np.concatenate([members[w] for w in rng.choice(unique, len(unique))])
        for k, y in targets.items():
            boot[k].append(holiday_coef(X[idx], y[idx], h))
    control_no2 = df.loc[df["holiday"] == 0, "no2_pphm"].mean()
    rows = []
    for k, y in targets.items():
        est = holiday_coef(X, y, h)
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        row = {"grain": grain, "outcome": k, "holiday_effect": est, "ci_lo": lo, "ci_hi": hi}
        if k.startswith("traffic"):  # log change -> % change
            row.update(percent=100 * np.expm1(est), percent_lo=100 * np.expm1(lo),
                       percent_hi=100 * np.expm1(hi))
        else:                        # pphm -> % of the control-day average
            row.update(percent=100 * est / control_no2, percent_lo=100 * lo / control_no2,
                       percent_hi=100 * hi / control_no2)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["holiday_days"] = df.loc[df["holiday"] == 1, "date"].nunique()
    out["control_days"] = df.loc[df["holiday"] == 0, "date"].nunique()
    out["stations"] = df["station_id"].nunique()
    out["rows"] = len(df)
    return out


def by_station(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    g = (df.groupby(["station_id", "holiday"])[["traffic_volume_total", "no2_pphm", "wind_speed_ms", "temp_c"]]
         .mean().unstack("holiday"))
    out = pd.DataFrame({
        "traffic_control": g[("traffic_volume_total", 0.0)], "traffic_holiday": g[("traffic_volume_total", 1.0)],
        "no2_control": g[("no2_pphm", 0.0)], "no2_holiday": g[("no2_pphm", 1.0)],
        "wind_control": g[("wind_speed_ms", 0.0)], "wind_holiday": g[("wind_speed_ms", 1.0)],
    }).dropna()
    out["traffic_change_pct"] = 100 * (out["traffic_holiday"] / out["traffic_control"] - 1)
    out["no2_change_pct"] = 100 * (out["no2_holiday"] / out["no2_control"] - 1)
    out.insert(0, "grain", grain)
    return add_station_name(out.reset_index())


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Holiday minus control, by hour of day: traffic in % (per station, then
    averaged) and NO2 in pphm. `thin_hour` marks hours with far fewer rows
    than usual -- hour ending 02:00 is missing NO2 on ~80% of days at every
    monitor (see 02_cleaning.py), so its average rests on a few days and can
    jump around; the plot shows those points hollow."""
    m = df.groupby(["station_id", "hour_of_day", "holiday"])[["traffic_volume_total", "no2_pphm"]].mean()
    m = m.unstack("holiday").dropna()
    prof = pd.DataFrame({
        "traffic_change_pct": 100 * (m[("traffic_volume_total", 1.0)] / m[("traffic_volume_total", 0.0)] - 1),
        "no2_control": m[("no2_pphm", 0.0)], "no2_holiday": m[("no2_pphm", 1.0)],
    }).groupby("hour_of_day").mean()
    prof["no2_change_pphm"] = prof["no2_holiday"] - prof["no2_control"]
    prof["n_rows"] = df.groupby("hour_of_day").size().reindex(prof.index)
    prof["thin_hour"] = prof["n_rows"] < 0.6 * prof["n_rows"].median()
    return prof.reset_index()


def plot_profile(prof: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    axes[0].bar(prof["hour_of_day"], prof["traffic_change_pct"], color="#7f7f7f")
    axes[0].set_ylabel("traffic, holiday vs\nnormal weekday (%)")
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    thin = prof["thin_hour"]
    for col, color, label in (("no2_control", "#c0392b", "normal weekday"),
                              ("no2_holiday", "#2f6db5", "public holiday")):
        axes[1].plot(prof["hour_of_day"], prof[col], color=color, label=label)
        axes[1].scatter(prof.loc[~thin, "hour_of_day"], prof.loc[~thin, col], s=12, color=color)
        axes[1].scatter(prof.loc[thin, "hour_of_day"], prof.loc[thin, col], s=30,
                        facecolors="white", edgecolors=color, zorder=3)
    axes[1].set_ylabel("NO$_2$ (pphm)")
    axes[1].set_xlabel("hour of day")
    axes[1].legend()
    if thin.any():
        axes[1].annotate("hollow = few readings at that hour (NO$_2$ missing on most days)",
                         xy=(0.01, 0.02), xycoords="axes fraction", fontsize=7, color="#555555")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_title("Weekday public holidays vs normal weekdays (same stations, +-14 days)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dose_response(stations: pd.DataFrame, grain: str) -> pd.DataFrame:
    """One dot per station: how much traffic fell on holidays (x) against how
    much NO2 fell (y). If traffic drives NO2, stations whose traffic dropped
    more should see bigger NO2 drops (a dose-response pattern). Stations far
    from their monitor are drawn hollow -- their NO2 reflects a wider area, so
    they're expected to follow the pattern less closely. Raw averages, not
    weather-adjusted (the adjusted overall effect is in holiday_<grain>_effect.csv)."""
    from scipy import stats

    from scripts.stations import STATIONS

    distance = {key.split(" - ")[0]: cfg["aq_distance_km"] for key, cfg in STATIONS.items()}
    s = stations.copy()
    s["aq_distance_km"] = s["station_id"].map(distance)
    near = s["aq_distance_km"] <= NEAR_MONITOR_KM
    rows = []
    for name, sub in (("all stations", s), (f"monitor <= {NEAR_MONITOR_KM:g} km", s[near])):
        if len(sub) >= 3:
            r, p = stats.pearsonr(sub["traffic_change_pct"], sub["no2_change_pct"])
            rho, p_rho = stats.spearmanr(sub["traffic_change_pct"], sub["no2_change_pct"])
            slope = np.polyfit(sub["traffic_change_pct"], sub["no2_change_pct"], 1)[0]
            rows.append({"grain": grain, "stations_used": name, "n_stations": len(sub),
                         "pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": p_rho,
                         "slope_no2_pct_per_traffic_pct": slope})
    corr = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(s.loc[near, "traffic_change_pct"], s.loc[near, "no2_change_pct"], s=45,
               color="#2f6db5", label=f"monitor within {NEAR_MONITOR_KM:g} km")
    ax.scatter(s.loc[~near, "traffic_change_pct"], s.loc[~near, "no2_change_pct"], s=45,
               facecolors="white", edgecolors="#2f6db5", label=f"monitor over {NEAR_MONITOR_KM:g} km away")
    for r in s.itertuples():
        ax.annotate(station_label(r.station_id), (r.traffic_change_pct, r.no2_change_pct),
                    fontsize=6.5, xytext=(4, 3), textcoords="offset points", color="#444444")
    if len(s) >= 3:
        fit = np.polyfit(s["traffic_change_pct"], s["no2_change_pct"], 1)
        x = np.linspace(s["traffic_change_pct"].min(), s["traffic_change_pct"].max(), 50)
        ax.plot(x, np.polyval(fit, x), color="#c0392b", linewidth=1, label="straight-line fit (all)")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("traffic change on weekday holidays (%)")
    ax.set_ylabel("NO$_2$ change on weekday holidays (%)")
    text = "; ".join(f"{c.stations_used}: r = {c.pearson_r:.2f} (n = {c.n_stations})"
                     for c in corr.itertuples())
    ax.set_title(f"Holiday dose-response -- {grain}\n{text}", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"holiday_{grain}_dose_response.png", dpi=150)
    plt.close(fig)
    corr.to_csv(OUT_DIR / f"holiday_{grain}_dose_response.csv", index=False)
    return corr


def run(grain: str, n_boot: int, seed: int) -> None:
    df = load(grain)
    if df.empty or df["holiday"].sum() == 0:
        print(f"[{grain}] no weekday public holidays in the data -- skipped")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stations = by_station(df, grain)
    stations.to_csv(OUT_DIR / f"holiday_{grain}_by_station.csv", index=False)
    eff = effects(df, grain, n_boot, seed)
    eff.to_csv(OUT_DIR / f"holiday_{grain}_effect.csv", index=False)
    print(f"\n{grain}: {eff['holiday_days'].iloc[0]} weekday holidays vs {eff['control_days'].iloc[0]} "
          f"control weekdays, {eff['stations'].iloc[0]} stations ({eff['rows'].iloc[0]:,} rows)")
    for r in eff.itertuples():
        print(f"  holiday effect on {r.outcome:22s} {r.percent:+6.1f}%  "
              f"(95% CI {r.percent_lo:+.1f}% to {r.percent_hi:+.1f}%)  "
              f"[same station, month{', hour' if grain == 'hourly' else ''}, weather]")
    print("  by station (raw averages):\n    " + "\n    ".join(
        f"{station_label(r.station_id):38s} traffic {r.traffic_change_pct:+.0f}%, NO2 {r.no2_change_pct:+.0f}%"
        for r in stations.itertuples()))
    corr = dose_response(stations, grain)
    print("  dose-response (station traffic change vs NO2 change): " + "; ".join(
        f"{c.stations_used} r = {c.pearson_r:.2f} (p = {c.pearson_p:.3f}, n = {c.n_stations})"
        for c in corr.itertuples()))
    if grain == "hourly":
        prof = hourly_profile(df)
        prof.to_csv(OUT_DIR / "holiday_hourly_profile.csv", index=False)
        plot_profile(prof, OUT_DIR / "holiday_hourly_profile.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Public-holiday natural experiment: less traffic, same clock.")
    parser.add_argument("--grain", nargs="+", choices=["daily", "hourly"], default=["daily", "hourly"])
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=paths.CONFIG_PATH,
                        help="accepted for run_pipeline.py; not used")
    args = parser.parse_args()
    for grain in args.grain:
        run(grain, args.n_boot, args.seed)
    print(f"\nSaved to {OUT_DIR.relative_to(paths.ROOT_DIR)}/")


if __name__ == "__main__":
    main()