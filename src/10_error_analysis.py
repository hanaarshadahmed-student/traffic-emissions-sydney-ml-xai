"""
CO2/NO2 Traffic-Emissions Capstone -- Error analysis

WHEN do the models get NO2 wrong? Breaks the saved row-by-row predictions
(results/predictions/) down by time of day, day type, month, station, road
type and pollution level. Nothing is retrained.

Error measures, all in pphm:
  RMSE   typical size of an error (big misses count extra)
  MAE    average size of an error
  bias   average of (predicted - actual): negative = the model UNDER-predicts
         (e.g. misses pollution peaks), positive = it over-predicts

Breakdowns (one table per breakdown, every model side by side):
  hour         hour of day, 0-23                (hourly data only)
  day_type     weekday / weekend
  school_hol   school holiday or not
  month        calendar month
  station      each station
  road_type    Highway / Major Road / Local Street. The TEST rows are all
               Local Street, so this one is also computed on the VALIDATION
               rows (road_type_val) -- the only split with all three types.
  no2_level    actual NO2 in five equal-sized groups (quintiles), lowest to
               highest -- shows whether peaks are under-predicted

Which models: every model with saved predictions for --feature-set (default
"all"), tuned version where available, plus the B1 persistence baseline for
reference.

Outputs (results/error_analysis/):
  errors_<grain>_<split>_<feature_set>_<breakdown>.csv
  errors_<grain>_<split>_<feature_set>_<breakdown>.png   RMSE (and bias) per group

Usage:
    python src/10_error_analysis.py                   # both grains, test split, all features
    python src/10_error_analysis.py --grain hourly --feature-set exogenous
Run AFTER the pipeline (needs results/predictions/ and data/processed/splits/).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paths  # noqa: E402

OUT_DIR = paths.RESULTS_DIR / "error_analysis"
DISPLAY = {
    "naive_persistence": "B1 Persistence",
    "ridge": "Ridge", "decision_tree": "Decision tree", "random_forest": "Random forest",
    "xgboost": "XGBoost", "svr": "SVR", "lstm": "LSTM", "gru": "GRU",
}
FILE_RE = re.compile(r"^(?P<model>.+)_(?P<stage>default|tuned)_(?P<grain>daily|hourly)_"
                     r"(?P<feature_set>exogenous|autoregressive|all)_(?P<split>val|test)\.csv\.gz$")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load(grain: str, split: str, feature_set: str) -> pd.DataFrame | None:
    """One long table: split metadata + y_true + one y_pred column per model."""
    files = []
    for path in paths.PREDICTIONS_DIR.glob(f"*_{grain}_*_{split}.csv.gz"):
        m = FILE_RE.match(path.name)
        if not m or m["model"] not in DISPLAY:
            continue
        # persistence is saved under one feature set only (it uses no features)
        if m["model"] != "naive_persistence" and m["feature_set"] != feature_set:
            continue
        files.append({**m.groupdict(), "path": path, "_rank": int(m["stage"] != "tuned")})
    if not files:
        print(f"[{grain}/{split}/{feature_set}] no prediction files -- skipped")
        return None
    chosen = (pd.DataFrame(files).sort_values("_rank")
              .drop_duplicates("model").set_index("model"))

    meta = pd.read_csv(paths.SPLITS_DIR / f"{grain}_{split}.csv",
                       usecols=["station_id", "timestamp", "school_holiday", "road_type_bucket", "no2_pphm"],
                       dtype={"station_id": str})
    ts = pd.to_datetime(meta["timestamp"])
    frame = pd.DataFrame({
        "station": meta["station_id"],
        "road_type": meta["road_type_bucket"],
        "hour": ts.dt.hour,
        "day_type": np.where(ts.dt.dayofweek >= 5, "weekend", "weekday"),
        "school_hol": np.where(meta["school_holiday"].fillna(0).astype(int) == 1,
                               "school holiday", "term time"),
        "month": pd.Categorical(ts.dt.month.map(lambda m: MONTHS[m - 1]), MONTHS, ordered=True),
        "y_true": meta["no2_pphm"].to_numpy(float),
    })
    frame["no2_level"] = pd.qcut(frame["y_true"].rank(method="first"), 5,
                                 labels=["1 lowest", "2", "3", "4", "5 highest"])
    order = [m for m in DISPLAY if m in chosen.index]
    for model in order:
        pred = pd.read_csv(chosen.loc[model, "path"], dtype={"station_id": str}).sort_values("row")
        if len(pred) != len(frame) or not (pred["station_id"].to_numpy() == frame["station"].to_numpy()).all():
            raise SystemExit(f"{chosen.loc[model, 'path'].name} doesn't line up with the current "
                             f"{grain}_{split}.csv -- it comes from a different run. Rerun the pipeline.")
        frame[DISPLAY[model]] = pred["y_pred"].to_numpy(float)
    frame.attrs["models"] = [DISPLAY[m] for m in order]
    frame.attrs["stages"] = {DISPLAY[m]: chosen.loc[m, "stage"] for m in order}
    return frame


def breakdown(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for group, part in frame.groupby(by, observed=True, sort=True):
        row = {by: group, "n_rows": len(part), "n_stations": part["station"].nunique(),
               "mean_no2": part["y_true"].mean()}
        for model in frame.attrs["models"]:
            err = part[model] - part["y_true"]
            row[f"{model} | rmse"] = float(np.sqrt(np.mean(err ** 2)))
            row[f"{model} | mae"] = float(np.mean(np.abs(err)))
            row[f"{model} | bias"] = float(np.mean(err))
        rows.append(row)
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, by: str, models: list[str], title: str, path: Path) -> None:
    labels = table[by].astype(str).tolist()
    x = np.arange(len(labels))
    line = by in ("hour", "month", "no2_level")
    fig, axes = plt.subplots(2, 1, figsize=(max(6.5, 0.45 * len(labels) + 3), 7), sharex=True)
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        style = dict(color="#8a8a8a", linestyle="--") if model.startswith("B1") else {}
        for ax, metric in zip(axes, ("rmse", "bias")):
            values = table[f"{model} | {metric}"]
            if line:
                ax.plot(x, values, marker="o", markersize=3, label=model, **style)
            else:
                ax.bar(x - 0.4 + width * (i + 0.5), values, width, label=model,
                       color=style.get("color"))
    axes[0].set_ylabel("RMSE (pphm)")
    axes[1].set_ylabel("bias (pphm)\n< 0 = under-predicts")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(x, labels, rotation=0 if len(labels) <= 12 else 90)
    axes[1].set_xlabel({"no2_level": "actual NO$_2$ level (fifths, lowest to highest)",
                        "school_hol": "school holiday"}.get(by, by.replace("_", " ")))
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(grain: str, split: str, feature_set: str) -> None:
    frame = load(grain, split, feature_set)
    if frame is None:
        return
    models = frame.attrs["models"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{grain} | {split} | {feature_set} features: {len(frame):,} rows, "
          f"{frame['station'].nunique()} stations; models: "
          + ", ".join(f"{m} ({frame.attrs['stages'][m]})" for m in models))

    breakdowns = ["day_type", "school_hol", "month", "station", "road_type", "no2_level"]
    if grain == "hourly":
        breakdowns.insert(0, "hour")
    for by in breakdowns:
        table = breakdown(frame, by)
        if len(table) < 2:
            print(f"  {by}: only one group in {split} rows ({table[by].iloc[0]}) -- skipped")
            continue
        tag = f"{grain}_{split}_{feature_set}_{by}"
        table.to_csv(OUT_DIR / f"errors_{tag}.csv", index=False)
        plot(table, by, models, f"{grain.capitalize()} NO$_2$ errors by {by.replace('_', ' ')} "
             f"({split}, {feature_set} features)", OUT_DIR / f"errors_{tag}.png")
        print_table(table, by, models)

    # road type needs the validation rows (test is a single road type)
    if split == "test" and frame["road_type"].nunique() < 2:
        val = load(grain, "val", feature_set)
        if val is not None and val["road_type"].nunique() > 1:
            table = breakdown(val, "road_type")
            tag = f"{grain}_val_{feature_set}_road_type"
            table.to_csv(OUT_DIR / f"errors_{tag}.csv", index=False)
            plot(table, "road_type", val.attrs["models"],
                 f"{grain.capitalize()} NO$_2$ errors by road type (val, {feature_set} features)",
                 OUT_DIR / f"errors_{tag}.png")
            print("  road_type -- from VALIDATION rows (test is all one road type):")
            print_table(table, "road_type", val.attrs["models"])


def print_table(table: pd.DataFrame, by: str, models: list[str]) -> None:
    cols = [by, "n_rows", "n_stations", "mean_no2"] + [f"{m} | rmse" for m in models]
    shown = table[cols].rename(columns=lambda c: c.replace(" | rmse", ""))
    print(f"  RMSE by {by}:")
    print("    " + shown.to_string(index=False, float_format=lambda v: f"{v:.3f}").replace("\n", "\n    "))
    bias = table.set_index(by)[[f"{m} | bias" for m in models if not m.startswith("B1")]].mean(axis=1)
    print(f"    mean bias of the ML models: " + ", ".join(f"{k}: {v:+.3f}" for k, v in bias.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Break prediction errors down by time, station and NO2 level.")
    parser.add_argument("--grain", nargs="+", choices=["daily", "hourly"], default=["daily", "hourly"])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--feature-set", choices=["exogenous", "autoregressive", "all"], default="all")
    parser.add_argument("--config", type=Path, default=paths.CONFIG_PATH,
                        help="accepted for run_pipeline.py; not used (scores saved predictions only)")
    args = parser.parse_args()
    for grain in args.grain:
        run(grain, args.split, args.feature_set)
    print(f"\nSaved to {OUT_DIR.relative_to(paths.ROOT_DIR)}/")


if __name__ == "__main__":
    main()