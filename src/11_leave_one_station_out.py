"""
CO2/NO2 Traffic-Emissions Capstone -- Leave-one-station-out (LOSO) testing

Would the model work at a station it has NEVER seen? The main test split only
checks LATER DATES at the SAME stations. Here, for each station in turn:
  1. train on every OTHER station (all dates: train + val + test periods)
  2. predict the held-out station (all its dates) and score it
so each station gets an "unseen location" score.

Default feature set is `exogenous` (traffic, weather, calendar, road/station):
at a genuinely new location there is no NO2 monitor, so NO2-history features
wouldn't exist. Use --feature-set all to see the forecasting version.

Reference line: "network mean" predicts every row as the average NO2 of the
training stations -- what you'd guess with no model at all. A model is only
useful at a new site if it beats this. Expect wide variation between
stations: with ~9 stations, each held-out station is a large part of the
data and may be unlike all the others (that variation IS the finding).

Models use their config.yaml settings (training.models.<name>.params) and the
same sample weights as the main pipeline. Nothing here touches results.json.

Outputs (results/loso/):
  loso_<grain>_<feature_set>.csv           every station x model: rows, R2, RMSE, MAE
  loso_<grain>_<feature_set>_summary.csv   per model: mean / median / worst / best
                                           station R2, and R2 pooled over all stations
  loso_<grain>_<feature_set>.png           R2 per held-out station

Usage:
    python src/11_leave_one_station_out.py                     # daily+hourly, ridge+xgboost
    python src/11_leave_one_station_out.py --grain daily --models ridge xgboost random_forest
    python src/11_leave_one_station_out.py --feature-set all
Run AFTER the pipeline (needs data/processed/splits/). Hourly with
random_forest takes several minutes per station.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paths  # noqa: E402
from scripts.experiment_utils import (  # noqa: E402
    TARGET, TABULAR_MODELS, build, fit_score, load_config, read_split, weight_column,
)
from scripts.model_utils import evaluate, get_feature_columns  # noqa: E402
from scripts.models import display_name  # noqa: E402

OUT_DIR = paths.RESULTS_DIR / "loso"
MIN_ROWS = 30  # a station with fewer rows can't be scored reliably


def run(grain: str, feature_set: str, models: list[str], config: dict) -> None:
    data = pd.concat([read_split(grain, s) for s in ("train", "val", "test")], ignore_index=True)
    features = get_feature_columns(grain, feature_set)
    weights = weight_column(grain)
    counts = data["station_id"].value_counts()
    stations = sorted(counts[counts >= MIN_ROWS].index)
    skipped = sorted(counts[counts < MIN_ROWS].index)
    road = data.groupby("station_id")["road_type_bucket"].first()
    print(f"\n{grain} | {feature_set}: {len(data):,} rows, {len(stations)} stations held out in turn"
          + (f" (skipped, < {MIN_ROWS} rows: {skipped})" if skipped else ""), flush=True)

    rows, pooled = [], {m: [] for m in ["network_mean", *models]}
    for station in stations:
        held = data[data["station_id"] == station]
        rest = data[data["station_id"] != station]
        mean_pred = np.full(len(held), np.average(rest[TARGET], weights=rest[weights]))
        base = evaluate(held[TARGET], mean_pred)
        rows.append({"station_id": station, "road_type": road[station], "n_rows": len(held),
                     "model": "network_mean", **base})
        pooled["network_mean"].append((held[TARGET].to_numpy(), mean_pred))
        line = [f"network mean R2 {base['r2']:6.3f}"]
        for name in models:
            started = time.perf_counter()
            model = build(name, config)
            scores = fit_score(model, rest, {"held": held}, features, weights)["held"]
            pooled[name].append((held[TARGET].to_numpy(), model.predict(held[features])))
            rows.append({"station_id": station, "road_type": road[station], "n_rows": len(held),
                         "model": name, **scores, "seconds": round(time.perf_counter() - started, 1)})
            line.append(f"{display_name(name)} R2 {scores['r2']:6.3f}")
        print(f"  {station:8s} ({road[station]:12s}, {len(held):6,} rows): " + " | ".join(line), flush=True)

    table = pd.DataFrame(rows)
    summary = []
    for name, parts in pooled.items():
        per_station = table[table["model"] == name]["r2"]
        y = np.concatenate([p[0] for p in parts])
        p = np.concatenate([p[1] for p in parts])
        summary.append({"model": name, "stations": len(per_station),
                        "mean_station_r2": per_station.mean(), "median_station_r2": per_station.median(),
                        "worst_station_r2": per_station.min(), "best_station_r2": per_station.max(),
                        "stations_beating_network_mean": int(
                            (per_station.to_numpy() > table[table["model"] == "network_mean"]["r2"].to_numpy()).sum())
                        if name != "network_mean" else np.nan,
                        **{f"pooled_{k}": v for k, v in evaluate(y, p).items()}})
    summary = pd.DataFrame(summary)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{grain}_{feature_set}"
    table.to_csv(OUT_DIR / f"loso_{tag}.csv", index=False)
    summary.to_csv(OUT_DIR / f"loso_{tag}_summary.csv", index=False)
    plot(table, models, grain, feature_set, OUT_DIR / f"loso_{tag}.png")
    print("  summary (R2 across held-out stations):")
    print("    " + summary.round(3).to_string(index=False).replace("\n", "\n    "))


def plot(table: pd.DataFrame, models: list[str], grain: str, feature_set: str, path: Path) -> None:
    stations = table.drop_duplicates("station_id")
    labels = [f"{s}\n{r}" for s, r in zip(stations["station_id"], stations["road_type"])]
    names = ["network_mean", *models]
    x = np.arange(len(labels))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(labels) + 2), 4.8))
    for i, name in enumerate(names):
        values = table[table["model"] == name].set_index("station_id")["r2"].reindex(stations["station_id"])
        ax.bar(x - 0.4 + width * (i + 0.5), values.clip(lower=-1.0), width,
               label="network mean (no model)" if name == "network_mean" else display_name(name),
               color="#b0b0b0" if name == "network_mean" else None)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("R$^2$ on the held-out station (clipped at -1)")
    ax.set_title(f"Leave-one-station-out -- {grain}, {feature_set} features")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-station-out testing.")
    parser.add_argument("--grain", nargs="+", choices=["daily", "hourly"], default=["daily", "hourly"])
    parser.add_argument("--feature-set", nargs="+", dest="feature_sets",
                        choices=["exogenous", "autoregressive", "all"], default=["exogenous"])
    parser.add_argument("--models", nargs="+", choices=TABULAR_MODELS, default=["ridge", "xgboost"])
    parser.add_argument("--config", type=Path, default=paths.CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    for grain in args.grain:
        for feature_set in args.feature_sets:
            run(grain, feature_set, args.models, config)
    print(f"\nSaved to {OUT_DIR.relative_to(paths.ROOT_DIR)}/")


if __name__ == "__main__":
    main()