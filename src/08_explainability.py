"""
CO2/NO2 Traffic-Emissions Capstone -- Stage 08: Explainability (SHAP)

Explains the tuned XGBoost model: which inputs push the predicted NO2 up or
down, and by how much. Nothing is tuned here; the model is the one the tune
stage saved (results/saved_models/xgboost_<grain>_<feature_set>.joblib). If
none is saved yet, XGBoost is refitted on the train split with the settings
in config.yaml, so this stage also works after a --no-tune run.

Two views per grain (set under `explainability:` in config.yaml):
  all         exogenous + NO2 history: explains the best FORECASTING model.
              Expect the NO2 lags to dominate -- that is the "what does the
              model use" answer.
  exogenous   traffic / weather / calendar / road / station only: explains
              what EXTERNAL factors drive NO2 -- the "traffic vs weather"
              answer the research question is really after.

How to read the outputs:
  SHAP value  how many NO2 units one feature moved ONE prediction away from
              the average prediction. Positive = pushed NO2 up.
  importance  mean |SHAP| over the explained rows = the feature's typical
              effect size (same units as the target, pphm).
  direction   correlation between a feature's value and its SHAP value:
              positive = higher values raise NO2, negative = lower it.
              (Rough guide only -- see the dependence plots for the shape.)
  Features are scaled (05_train_test_split.py), so colours/axes in the plots
  show scaled values; compare high vs low, not absolute numbers.

Outputs (results/shap/):
  importance_<grain>_<feature_set>.csv      every feature: importance, share, direction
  summary_<grain>_<feature_set>.png         beeswarm: effect and value, top features
  bar_<grain>_<feature_set>.png             mean |SHAP| bar chart, top features
  dependence_<grain>_<feature_set>_<feature>.png   effect against value, top features
  groups_<grain>_<feature_set>.csv          importance summed by feature family
  shap_values_<grain>_<feature_set>.csv.gz  SHAP value of every explained row

Usage:
    python src/08_explainability.py
    python src/08_explainability.py --grain hourly --feature-set exogenous
    python src/08_explainability.py --max-rows 20000 --top 20
Or as part of:   python run_pipeline.py --stage explain
Needs the `shap` package (pip install shap).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paths  # noqa: E402
from scripts.model_utils import (  # noqa: E402
    fit_with_optional_weight,
    load_split,
)
from scripts.models import build_model  # noqa: E402

OUT_DIR = paths.SHAP_DIR
TARGET_UNIT = "pphm"

# Feature families, matched on the feature name (first match wins). Used only
# to sum importance into a "traffic vs weather vs NO2 history" table; the
# per-feature table is the primary result.
FAMILIES = [
    ("NO2 history", ("no2_pphm",)),
    ("Traffic", ("traffic", "vehicle")),
    ("Weather", ("temp", "rain", "precip", "wind", "humid", "pressure", "solar",
                 "radiation", "cloud", "dew", "weather")),
    ("Calendar", ("hour", "dow", "day_of_week", "weekday", "weekend", "month",
                  "season", "holiday", "day_of_year", "doy", "working_day", "is_")),
    ("Road / station", ("road", "lane", "station", "distance", "dist_", "zone", "speed",
                        "intersection", "aq_", "lat", "lon", "elevation", "class", "type")),
]


def family_of(feature: str) -> str:
    name = feature.lower()
    for family, keys in FAMILIES:
        if any(key in name for key in keys):
            return family
    return "Other"


def load_main_config(path: Path) -> dict:
    with open(path) as file:
        return yaml.safe_load(file) or {}


def get_model(grain: str, feature_set: str, main_config: dict):
    """The tuned XGBoost if the tune stage saved one, else a fresh fit on train."""
    model_path = paths.SAVED_MODELS_DIR / f"xgboost_{grain}_{feature_set}.joblib"
    if model_path.exists():
        return joblib.load(model_path), f"tuned model ({model_path.name})"
    params = ((main_config.get("training", {}).get("models", {}).get("xgboost") or {})
              .get("params") or {})
    model = build_model("xgboost", **params)
    X, y, weights = load_split(grain, "train", feature_set, with_weight=True)
    fit_with_optional_weight(model, X, y, weights)
    return model, "default XGBoost refitted on train (no tuned model saved yet)"


def importance_table(shap_values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mean_abs = np.abs(shap_values).mean(axis=0)
    total = mean_abs.sum() or 1.0
    for index, feature in enumerate(X.columns):
        values, effects = X[feature].to_numpy(dtype=float), shap_values[:, index]
        if np.std(values) > 0 and np.std(effects) > 0:
            direction = float(np.corrcoef(values, effects)[0, 1])
        else:
            direction = 0.0
        rows.append({"feature": feature, "family": family_of(feature),
                     f"mean_abs_shap_{TARGET_UNIT}": float(mean_abs[index]),
                     "share_of_total": float(mean_abs[index] / total),
                     "direction_corr": direction})
    table = pd.DataFrame(rows).sort_values(f"mean_abs_shap_{TARGET_UNIT}", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def explain(grain: str, feature_set: str, args, main_config: dict) -> pd.DataFrame | None:
    import shap

    try:
        X, _ = load_split(grain, args.split, feature_set)
    except FileNotFoundError as error:
        print(f"[{grain}/{feature_set}] skipped: {error}")
        return None
    model, source = get_model(grain, feature_set, main_config)

    if len(X) > args.max_rows:
        X = X.sample(args.max_rows, random_state=42).sort_index()
    print(f"\n{grain} | {feature_set}: explaining {len(X):,} {args.split} rows "
          f"with the {source}", flush=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # additivity: base value + sum of SHAP values must equal the prediction
    gap = float(np.max(np.abs(explainer.expected_value + shap_values.sum(axis=1)
                              - model.predict(X))))
    print(f"  base value {float(explainer.expected_value):.3f} {TARGET_UNIT}; "
          f"max additivity error {gap:.2e}", flush=True)

    tag = f"{grain}_{feature_set}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table = importance_table(shap_values, X)
    table.to_csv(OUT_DIR / f"importance_{tag}.csv", index=False)
    groups = (table.groupby("family")[[f"mean_abs_shap_{TARGET_UNIT}", "share_of_total"]]
              .sum().sort_values("share_of_total", ascending=False).reset_index())
    groups.to_csv(OUT_DIR / f"groups_{tag}.csv", index=False)
    pd.DataFrame(shap_values, columns=X.columns, index=X.index).to_csv(
        OUT_DIR / f"shap_values_{tag}.csv.gz", index_label="row")

    top = min(args.top, X.shape[1])
    shap.summary_plot(shap_values, X, max_display=top, show=False)
    plt.title(f"SHAP summary -- {grain}, {feature_set} features ({args.split} rows)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"summary_{tag}.png", dpi=150)
    plt.close("all")

    shap.summary_plot(shap_values, X, plot_type="bar", max_display=top, show=False)
    plt.xlabel(f"mean |SHAP value| ({TARGET_UNIT})")
    plt.title(f"Feature importance -- {grain}, {feature_set} features")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"bar_{tag}.png", dpi=150)
    plt.close("all")

    for feature in table["feature"].head(args.dependence):
        shap.dependence_plot(feature, shap_values, X, interaction_index=None, show=False)
        plt.title(f"Effect of {feature} -- {grain}, {feature_set}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"dependence_{tag}_{feature}.png", dpi=150)
        plt.close("all")

    unit_col = f"mean_abs_shap_{TARGET_UNIT}"
    shown = table.head(10)[["rank", "feature", "family", unit_col, "share_of_total", "direction_corr"]]
    print(f"  top 10 features ({grain}, {feature_set}):")
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("  by feature family:")
    print(groups.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    table.insert(0, "feature_set", feature_set)
    table.insert(0, "grain", grain)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=paths.CONFIG_PATH,
                        help="config.yaml of the pipeline run (grains, XGBoost settings, explainability:).")
    parser.add_argument("--grain", nargs="+", choices=["daily", "hourly"], default=None)
    parser.add_argument("--feature-set", nargs="+", dest="feature_sets",
                        choices=["all", "exogenous", "autoregressive"], default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None,
                        help="which rows to explain (default: test)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="explain at most this many randomly chosen rows (default 5000)")
    parser.add_argument("--top", type=int, default=None, help="features shown in plots (default 15)")
    parser.add_argument("--dependence", type=int, default=None,
                        help="dependence plots for the top N features (default 4)")
    args = parser.parse_args()

    main_config = load_main_config(args.config)
    settings = main_config.get("explainability") or {}
    if not settings.get("enabled", True):
        print("Explainability skipped -- explainability.enabled is false in config.yaml.")
        return
    args.split = args.split or settings.get("split", "test")
    args.max_rows = args.max_rows or int(settings.get("max_rows", 5000))
    args.top = args.top or int(settings.get("top_features", 15))
    args.dependence = args.dependence if args.dependence is not None else int(settings.get("dependence_plots", 4))
    grains = args.grain or main_config.get("training", {}).get("grain", ["daily"])
    grains = [grains] if isinstance(grains, str) else list(grains)
    feature_sets = args.feature_sets or settings.get("feature_sets", ["all", "exogenous"])

    try:
        import shap  # noqa: F401
    except ImportError:
        raise SystemExit("SHAP needs the 'shap' package:  pip install shap")

    tables = []
    for grain in grains:
        for feature_set in feature_sets:
            table = explain(grain, feature_set, args, main_config)
            if table is not None:
                tables.append(table)
    if tables:
        pd.concat(tables).to_csv(OUT_DIR / "importance_all.csv", index=False)
        print(f"\nSHAP outputs saved to {OUT_DIR.relative_to(paths.ROOT_DIR)}/")


if __name__ == "__main__":
    main()
