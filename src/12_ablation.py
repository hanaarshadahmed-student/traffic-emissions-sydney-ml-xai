"""
CO2/NO2 Traffic-Emissions Capstone -- Ablation study

Which parts of the pipeline actually matter? Retrains a model with ONE thing
removed at a time and measures how much the validation score drops:

  - <family>             every feature in one family removed (Traffic,
                         Weather, Calendar, Road / station, NO2 history,
                         Traffic x weather -- see scripts/feature_families.py)
  - ALL traffic / ALL weather
                         the family removed TOGETHER with Traffic x weather
                         (traffic / wind etc.), so no traffic (or weather)
                         information is left in at all -- the fair test of
                         "does the model need traffic?"
  - ALL traffic + Calendar
                         traffic AND calendar removed together. Traffic follows
                         a regular daily / weekly rhythm, so hour-of-day and
                         weekday features can stand in for it; if removing
                         traffic alone barely matters but removing both does,
                         the two are substitutes (the model can't separate them)
  - no sample weights    REVIEW-flagged stations count fully instead of half
                         (the aq_quality_weight from 03_data_preprocessing.py)
  - no feature pruning   the near-duplicate features 05_train_test_split.py
                         drops (|r| > 0.95 with a kept feature) put back in

A big drop = that part carries information the rest can't replace. A drop
near 0 = the other features cover for it (or it never helped). A RISE = the
model is better without it (a sign of overfitting -- relevant for daily data,
~150 features on ~2,500 rows).

Removing a family is a stronger test than SHAP: SHAP says how much the model
USES a feature; ablation says whether the model NEEDS it. Correlated families
(e.g. traffic vs calendar -- rush hour) can each look unimportant here while
SHAP shares the credit between them.

Scored on validation (the split for decisions); test is shown for reference.
Models use their config.yaml settings. Nothing here touches results.json.

Outputs (results/ablation/):
  ablation_<grain>.csv   every feature set x model x variant: features, val/test R2,
                         change in val R2 vs the full model
  ablation_<grain>.png   change in val R2 per variant

Usage:
    python src/12_ablation.py                          # both grains, ridge + xgboost
    python src/12_ablation.py --grain daily --models xgboost random_forest
    python src/12_ablation.py --feature-set exogenous
Run AFTER the pipeline (needs data/processed/splits/).
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
    TABULAR_MODELS, build, fit_score, load_config, read_split, weight_column,
)
from scripts.feature_families import group_features  # noqa: E402
from scripts.model_utils import get_feature_columns, load_excluded_features, load_manifest  # noqa: E402
from scripts.models import display_name  # noqa: E402

OUT_DIR = paths.RESULTS_DIR / "ablation"
MANIFEST_KEY = {"exogenous": "exogenous_features", "autoregressive": "autoregressive_features",
                "all": "all_candidate_features"}


def variants(grain: str, feature_set: str) -> list[tuple[str, list[str], bool]]:
    """(name, features, use_weights) for the full model and every ablation."""
    features = get_feature_columns(grain, feature_set)
    out = [("full model", features, True)]
    groups = group_features(features)
    removals = dict(groups)
    # "Traffic" alone isn't a clean test while traffic / wind (Traffic x weather)
    # stays in -- so also remove each together with the mixed family
    for main in ("Traffic", "Weather"):
        if main in groups and "Traffic x weather" in groups:
            removals[f"ALL {main.lower()}"] = groups[main] + groups["Traffic x weather"]
    # traffic follows a regular daily/weekly rhythm, so calendar features can
    # stand in for it: removing both together shows whether they are substitutes
    if "ALL traffic" in removals and "Calendar" in groups:
        removals["ALL traffic + Calendar"] = removals["ALL traffic"] + groups["Calendar"]
    for family, cols in removals.items():
        if len(cols) < len(features):  # removing the only family would leave nothing
            out.append((f"- {family} ({len(cols)})", [f for f in features if f not in cols], True))
    out.append(("- sample weights", features, False))
    excluded = load_excluded_features(grain)
    unpruned = load_manifest()[grain][MANIFEST_KEY[feature_set]]
    added = [f for f in unpruned if f in excluded]
    if added:
        out.append((f"- feature pruning (+{len(added)})", unpruned, True))
    return out


def run(grain: str, feature_sets: list[str], models: list[str], config: dict) -> pd.DataFrame:
    frames = {s: read_split(grain, s) for s in ("train", "val", "test")}
    weights = weight_column(grain)
    rows = []
    for feature_set in feature_sets:
        for name in models:
            print(f"\n{grain} | {feature_set} | {display_name(name)}", flush=True)
            full = None
            for variant, features, use_weights in variants(grain, feature_set):
                started = time.perf_counter()
                scores = fit_score(build(name, config), frames["train"],
                                   {"val": frames["val"], "test": frames["test"]},
                                   features, weights if use_weights else None)
                if full is None:
                    full = scores
                rows.append({"grain": grain, "feature_set": feature_set, "model": name,
                             "variant": variant, "n_features": len(features),
                             "val_r2": scores["val"]["r2"], "test_r2": scores["test"]["r2"],
                             "val_rmse": scores["val"]["rmse"], "test_rmse": scores["test"]["rmse"],
                             "val_r2_change": scores["val"]["r2"] - full["val"]["r2"],
                             "test_r2_change": scores["test"]["r2"] - full["test"]["r2"],
                             "seconds": round(time.perf_counter() - started, 1)})
                r = rows[-1]
                print(f"  {variant:40s} {r['n_features']:4d} features   val R2 {r['val_r2']:6.3f} "
                      f"({r['val_r2_change']:+.3f})   test R2 {r['test_r2']:6.3f} ({r['test_r2_change']:+.3f})",
                      flush=True)
    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / f"ablation_{grain}.csv", index=False)
    plot(table, grain, OUT_DIR / f"ablation_{grain}.png")
    return table


def plot(table: pd.DataFrame, grain: str, path: Path) -> None:
    panels = list(table[["feature_set", "model"]].drop_duplicates().itertuples(index=False))
    n_var = table.groupby(["feature_set", "model"]).size().max() - 1
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 0.42 * n_var + 1.8),
                             squeeze=False)
    for ax, (feature_set, model) in zip(axes[0], panels):
        part = table[(table["feature_set"] == feature_set) & (table["model"] == model)
                     & (table["variant"] != "full model")]
        y = np.arange(len(part))[::-1]
        colors = np.where(part["val_r2_change"] < 0, "#c0392b", "#2f6db5")
        ax.barh(y, part["val_r2_change"], color=colors)
        ax.set_yticks(y, part["variant"], fontsize=8)
        ax.axvline(0, color="#555555", linewidth=0.8)
        full = table[(table["feature_set"] == feature_set) & (table["model"] == model)
                     & (table["variant"] == "full model")]["val_r2"].iloc[0]
        ax.set_title(f"{display_name(model)}, {feature_set}\n(full model val R$^2$ = {full:.3f})", fontsize=10)
        ax.set_xlabel("change in val R$^2$ when removed")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Ablation -- {grain} (red = score drops, i.e. the part is needed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: retrain with one feature family or step removed.")
    parser.add_argument("--grain", nargs="+", choices=["daily", "hourly"], default=["daily", "hourly"])
    parser.add_argument("--feature-set", nargs="+", dest="feature_sets",
                        choices=["exogenous", "autoregressive", "all"], default=["exogenous", "all"])
    parser.add_argument("--models", nargs="+", choices=TABULAR_MODELS, default=["ridge", "xgboost"])
    parser.add_argument("--config", type=Path, default=paths.CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    for grain in args.grain:
        run(grain, args.feature_sets, args.models, config)
    print(f"\nSaved to {OUT_DIR.relative_to(paths.ROOT_DIR)}/")


if __name__ == "__main__":
    main()