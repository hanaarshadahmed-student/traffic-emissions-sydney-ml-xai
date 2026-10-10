"""
CO2/NO2 Traffic-Emissions Capstone -- Permutation importance

A second, independent check of what drives the model, to set beside SHAP.

How it works: take the trained model and the test rows; SHUFFLE one feature
(or one whole feature family) so its values no longer match their rows, and
re-score. The drop in R2 is that feature's importance -- if the model
genuinely relies on it, scrambling it hurts. Repeated --repeats times with
different shuffles; the spread gives an error bar.

Families are shuffled as a WHOLE (every feature in the family moved by the
same row shuffle). This matters: shuffling one traffic feature at a time
looks harmless when 40 other traffic features still carry the same
information.

How it differs from SHAP: SHAP shares out each prediction among the
features (how much the model USES each); permutation measures how much
ACCURACY depends on each. If both rank the families in the same order, the
explanation doesn't depend on the method -- reported as a Spearman rank
correlation.

Model: the same XGBoost the SHAP stage explains (the tuned one, if it matches
the current data). Rows: the test split, at most --max-rows sampled the same
way as SHAP.

Outputs (results/permutation/):
  permutation_families_<grain>_<feature_set>.csv   R2 drop per family (+ SHAP share, ranks)
  permutation_features_<grain>_<feature_set>.csv   R2 drop per single feature
  permutation_<grain>_<feature_set>.png            permutation vs SHAP, per family

Usage:
    python src/13_permutation_importance.py
    python src/13_permutation_importance.py --grain hourly --feature-set exogenous --repeats 20
Run AFTER the pipeline (it compares with results/shap/, made by the explain stage).
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
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paths  # noqa: E402
from scripts.experiment_utils import TARGET, explained_xgboost, load_config, r2, read_split  # noqa: E402
from scripts.feature_families import family_of, group_features  # noqa: E402
from scripts.model_utils import get_feature_columns  # noqa: E402

OUT_DIR = paths.RESULTS_DIR / "permutation"


def permutation_drop(model, X: pd.DataFrame, y: np.ndarray, columns: list[str],
                     repeats: int, rng: np.random.Generator, base: float) -> np.ndarray:
    """R2 drop for each repeat when `columns` are shuffled together."""
    drops = np.empty(repeats)
    shuffled = X.copy()
    for i in range(repeats):
        order = rng.permutation(len(X))
        shuffled[columns] = X[columns].to_numpy()[order]
        drops[i] = base - r2(y, model.predict(shuffled))
    return drops


def run(grain: str, feature_set: str, args, config: dict) -> None:
    features = get_feature_columns(grain, feature_set)
    test = read_split(grain, "test")
    X, y = test[features], test[TARGET].to_numpy(float)
    if len(X) > args.max_rows:  # same sample as SHAP
        X = X.sample(args.max_rows, random_state=42).sort_index()
        y = test.loc[X.index, TARGET].to_numpy(float)
    model, source = explained_xgboost(grain, feature_set, config, features)
    base = r2(y, model.predict(X))
    rng = np.random.default_rng(args.seed)
    print(f"\n{grain} | {feature_set}: {len(X):,} test rows, {source}, R2 {base:.3f}, "
          f"{args.repeats} shuffles each", flush=True)

    family_rows = []
    for family, cols in group_features(features).items():
        drops = permutation_drop(model, X, y, cols, args.repeats, rng, base)
        family_rows.append({"family": family, "n_features": len(cols),
                            "r2_drop": drops.mean(), "r2_drop_sd": drops.std(ddof=1)})
    families = pd.DataFrame(family_rows).sort_values("r2_drop", ascending=False)
    families["permutation_rank"] = range(1, len(families) + 1)

    feature_rows = []
    for feature in features:
        drops = permutation_drop(model, X, y, [feature], args.repeats, rng, base)
        feature_rows.append({"feature": feature, "family": family_of(feature),
                             "r2_drop": drops.mean(), "r2_drop_sd": drops.std(ddof=1)})
    singles = pd.DataFrame(feature_rows).sort_values("r2_drop", ascending=False)
    singles.insert(0, "rank", range(1, len(singles) + 1))

    # side by side with SHAP (re-grouped here so both use the same families)
    shap_path = paths.SHAP_DIR / f"importance_{grain}_{feature_set}.csv"
    rho = np.nan
    if shap_path.exists():
        shap_table = pd.read_csv(shap_path)
        shap_table["family"] = shap_table["feature"].map(family_of)
        share = shap_table.groupby("family")["share_of_total"].sum()
        families["shap_share"] = families["family"].map(share)
        families["shap_rank"] = families["shap_share"].rank(ascending=False, method="min")
        both = families.dropna(subset=["shap_share"])
        if len(both) >= 3:
            rho = spearmanr(both["r2_drop"], both["shap_share"]).statistic
    else:
        print(f"  (no {shap_path.name} -- run the explain stage to compare with SHAP)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{grain}_{feature_set}"
    families.to_csv(OUT_DIR / f"permutation_families_{tag}.csv", index=False)
    singles.to_csv(OUT_DIR / f"permutation_features_{tag}.csv", index=False)
    plot(families, singles, grain, feature_set, rho, base, OUT_DIR / f"permutation_{tag}.png")

    print("  by family (R2 drop when shuffled together):")
    cols = ["family", "n_features", "r2_drop", "r2_drop_sd", "permutation_rank"] + \
           (["shap_share", "shap_rank"] if "shap_share" in families else [])
    print("    " + families[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
          .replace("\n", "\n    "))
    if not np.isnan(rho):
        print(f"  agreement with SHAP family ranking: Spearman rho = {rho:.2f}")
    print("  top 10 single features:")
    print("    " + singles.head(10)[["rank", "feature", "family", "r2_drop"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}").replace("\n", "\n    "))


def plot(families: pd.DataFrame, singles: pd.DataFrame, grain: str, feature_set: str,
         rho: float, base: float, path: Path) -> None:
    has_shap = "shap_share" in families
    fig, axes = plt.subplots(1, 3 if has_shap else 2, figsize=(15 if has_shap else 10, 4.6))
    order = families["family"].tolist()
    y = np.arange(len(order))[::-1]
    axes[0].barh(y, families["r2_drop"], xerr=families["r2_drop_sd"], color="#2f6db5", capsize=3)
    axes[0].set_yticks(y, order)
    axes[0].set_xlabel("drop in test R$^2$ when the family is shuffled")
    axes[0].set_title("Permutation importance (family)")
    if has_shap:
        axes[1].barh(y, families["shap_share"] * 100, color="#e69f00")
        axes[1].set_yticks(y, order)
        axes[1].set_xlabel("share of SHAP importance (%)")
        title = "SHAP (same families)"
        if not np.isnan(rho):
            title += f"\nrank agreement: Spearman $\\rho$ = {rho:.2f}"
        axes[1].set_title(title)
    top = singles.head(12)
    y2 = np.arange(len(top))[::-1]
    axes[-1].barh(y2, top["r2_drop"], xerr=top["r2_drop_sd"], color="#7f7f7f", capsize=2)
    axes[-1].set_yticks(y2, top["feature"], fontsize=7)
    axes[-1].set_xlabel("drop in test R$^2$")
    axes[-1].set_title("Top single features")
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Permutation importance -- {grain}, {feature_set} features "
                 f"(XGBoost, test R$^2$ = {base:.3f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation importance, compared with SHAP.")
    parser.add_argument("--grain", nargs="+", choices=["daily", "hourly"], default=["daily", "hourly"])
    parser.add_argument("--feature-set", nargs="+", dest="feature_sets",
                        choices=["exogenous", "autoregressive", "all"], default=["all", "exogenous"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=paths.CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    for grain in args.grain:
        for feature_set in args.feature_sets:
            run(grain, feature_set, args, config)
    print(f"\nSaved to {OUT_DIR.relative_to(paths.ROOT_DIR)}/")


if __name__ == "__main__":
    main()