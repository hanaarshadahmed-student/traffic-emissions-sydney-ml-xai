"""
Ablation study: retrains the model with one part removed at a time (a feature family, sample weights, or feature pruning) and measures how much the validation score drops, to show which parts of the pipeline actually matter.

Usage:
    python experiments/03_ablation.py
    python experiments/03_ablation.py --grain daily --models xgboost random_forest
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
from scripts.feature_families import group_features, group_traffic_parts  # noqa: E402
from scripts.model_utils import get_feature_columns, load_excluded_features, load_manifest  # noqa: E402
from models.models import display_name  # noqa: E402

OUT_DIR = paths.ABLATION_DIR
MANIFEST_KEY = {"exogenous": "exogenous_features", "autoregressive": "autoregressive_features",
                "all": "all_candidate_features"}


def variants(grain: str, feature_set: str) -> list[tuple[str, list[str], bool]]:
    """(name, features, use_weights) for the full model and every ablation."""
    features = get_feature_columns(grain, feature_set)
    out = [("full model", features, True)]
    groups = group_features(features)
    removals = dict(groups)
    for main in ("Traffic", "Weather"):
        if main in groups and "Traffic x weather" in groups:
            removals[f"ALL {main.lower()}"] = groups[main] + groups["Traffic x weather"]
    if "ALL traffic" in removals and "Calendar" in groups:
        removals["ALL traffic + Calendar"] = removals["ALL traffic"] + groups["Calendar"]
    parts = group_traffic_parts(features)
    if "ALL traffic" in removals and "Unexpected traffic" in parts:
        keep = set(parts["Unexpected traffic"])
        removals["ALL traffic except unexpected"] = [f for f in removals["ALL traffic"] if f not in keep]
    for part, cols in parts.items():
        removals[f"traffic part: {part}"] = cols
    for family, cols in removals.items():
        if len(cols) < len(features):  
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
                print(f"  {variant:44s} {r['n_features']:4d} features   val R2 {r['val_r2']:6.3f} "
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