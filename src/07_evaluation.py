"""
CO2/NO2 Traffic-Emissions Capstone -- Stage 07: Evaluation / method comparison

Reads results/results.json (nothing is retrained, so it's instant) and
compares every method, grouped by type:

  Baseline methods (not ML -- the bar to beat)
      B1 Persistence, B2 Seasonal climatology
  ML models
      1 Ridge, 2 Decision tree, 3 Random forest, 4 XGBoost, 5 SVR, 6 LSTM,
      7 GRU
      (each as "default" = config.yaml settings, and "tuned" once tuned)

What it prints (per grain):
  SUMMARY   one row per method: R2 on validation and test for each feature
            set. Choose between methods on VALIDATION; report TEST once.
  --detail  the full table, one block per method: RMSE / MAE / R2 on
            train, val and test for every feature set, plus the gaps
            (train >> val = overfitting; val ~= test = val was a fair
            stand-in for unseen data).

Outputs (results/evaluation/):
  summary_<grain>.csv / .png        the summary table + chart (start here)
  train_val_test_<grain>.csv / .png the full detail table + chart
  stage_delta_<grain>.csv           tuned minus default, when tuning has run
  all_results.csv                   every results.json row, flattened

Usage:
    python src/07_evaluation.py
    python src/07_evaluation.py --detail                 # + full tables
    python src/07_evaluation.py --grain daily --model ridge lstm
    python src/07_evaluation.py --trials                 # + every tuning trial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sys

# Repo root on the import path, so the shared code in scripts/ is importable
# whichever folder you run this from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paths  # noqa: E402
from scripts.models import BASELINE_METHODS, REGISTRY, display_name, model_type  # noqa: E402

ROOT_DIR = paths.ROOT_DIR
RESULTS_PATH = paths.RESULTS_PATH
OUT_DIR = paths.EVALUATION_DIR

STAGES = ["default", "tuned"]
TYPES = ["baseline", "ML"]                       # baselines always listed first
MODEL_ORDER = [*BASELINE_METHODS, *REGISTRY]
FEATURE_SETS = ["exogenous", "autoregressive", "all"]
SPLITS = ["train", "val", "test"]
METRICS = [("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R2")]
KEY = ["model", "stage", "grain", "feature_set", "split"]
SPLIT_COLOURS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
SET_COLOURS = {"exogenous": "#4C72B0", "autoregressive": "#DD8452", "all": "#55A868"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise SystemExit(
            f"{RESULTS_PATH} not found -- run the train stage "
            "(and/or the tune stage) of run_pipeline.py first."
        )
    df = pd.DataFrame(json.loads(RESULTS_PATH.read_text()))

    # Rows from before `stage` existed: tuning scripts always set tuned=True.
    tuned = df["tuned"].eq(True) if "tuned" in df else pd.Series(False, index=df.index)
    inferred = pd.Series(np.where(tuned, "tuned", "default"), index=df.index)
    df["stage"] = df["stage"].fillna(inferred) if "stage" in df else inferred
    # "baseline" was the old name for the untuned stage -- now "default", so it
    # can't be confused with the baseline METHODS
    df["stage"] = df["stage"].replace({"baseline": "default"})

    # Older rows kept train scores as train_* columns on the val row rather
    # than as their own split="train" row -- turn those into train rows.
    if "train_rmse" in df:
        legacy = df[df["train_rmse"].notna() & (df["split"] != "train")]
        have_train = set(map(tuple, df.loc[df["split"] == "train", KEY[:4]].values))
        extra = []
        for _, row in legacy.iterrows():
            if tuple(row[KEY[:4]]) not in have_train:
                extra.append({**{k: row[k] for k in KEY[:4]}, "split": "train",
                              "rmse": row["train_rmse"], "mae": row.get("train_mae"),
                              "r2": row["train_r2"]})
        if extra:
            df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    df = df.drop_duplicates(subset=KEY, keep="last")
    df = df[KEY + [m for m, _ in METRICS]].copy()
    df.insert(0, "type", df["model"].map(model_type))
    # one label per method + stage, e.g. "3 Random forest" / "4 XGBoost (tuned)"
    df.insert(1, "method", [
        display_name(m) + (" (tuned)" if st == "tuned" else "")
        for m, st in zip(df["model"], df["stage"])
    ])
    return df


def _method_order(df: pd.DataFrame) -> list[str]:
    pairs = df[["type", "model", "stage", "method"]].drop_duplicates()
    pairs = pairs.assign(
        _t=pairs["type"].map(TYPES.index),
        _m=pairs["model"].map(lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99),
        _s=pairs["stage"].map(lambda s: STAGES.index(s) if s in STAGES else 99),
    ).sort_values(["_t", "_m", "_s"])
    return list(pairs["method"])


def _feature_sets(df: pd.DataFrame) -> list[str]:
    present = set(df["feature_set"])
    return [f for f in FEATURE_SETS if f in present] + sorted(present - set(FEATURE_SETS))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per method; columns (feature set, val R2 / test R2)."""
    sub = df[df["split"].isin(["val", "test"])]
    table = sub.pivot_table(index=["type", "method"], columns=["feature_set", "split"], values="r2")
    sets = _feature_sets(df)
    columns = [(fs, sp) for fs in sets for sp in ("val", "test") if (fs, sp) in table.columns]
    table = table.reindex(columns=pd.MultiIndex.from_tuples(columns))
    table.columns = pd.MultiIndex.from_tuples([(fs, f"{sp} R2") for fs, sp in table.columns])
    order = _method_order(df)
    table = table.reindex(sorted(table.index, key=lambda ix: order.index(ix[1])))
    table.index.names = ["type", "method"]
    return table


def detail_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows (type, method, feature set); columns (metric, split) + gaps."""
    table = df.pivot_table(index=["type", "method", "feature_set"], columns="split",
                           values=[m for m, _ in METRICS])
    splits = [s for s in SPLITS if s in table.columns.get_level_values(1)]
    table = table.reindex(columns=pd.MultiIndex.from_product([[m for m, _ in METRICS], splits]))
    table.columns = pd.MultiIndex.from_tuples([(label, s) for m, s in table.columns
                                               for mm, label in METRICS if mm == m])
    if ("R2", "train") in table and ("R2", "val") in table:
        table[("gap", "train-val R2")] = table[("R2", "train")] - table[("R2", "val")]
    if ("R2", "val") in table and ("R2", "test") in table:
        table[("gap", "val-test R2")] = table[("R2", "val")] - table[("R2", "test")]
    order, sets = _method_order(df), _feature_sets(df)
    return table.loc[sorted(table.index, key=lambda ix: (order.index(ix[1]), sets.index(ix[2])))]


def stage_delta(df: pd.DataFrame) -> pd.DataFrame:
    ml = df[df["type"] == "ML"]
    wide = ml.pivot_table(index=["model", "feature_set", "split"], columns="stage", values=["rmse", "r2"])
    if not {"default", "tuned"} <= set(wide.columns.get_level_values(1)):
        return pd.DataFrame()
    out = pd.DataFrame({
        "default_rmse": wide[("rmse", "default")], "tuned_rmse": wide[("rmse", "tuned")],
        "default_r2": wide[("r2", "default")], "tuned_r2": wide[("r2", "tuned")],
    })
    out["delta_rmse"] = out["tuned_rmse"] - out["default_rmse"]
    out["delta_r2"] = out["tuned_r2"] - out["default_r2"]
    out = out.dropna(subset=["default_rmse", "tuned_rmse"])
    out.index = out.index.set_levels(
        [display_name(m) for m in out.index.levels[0]], level=0
    )
    return out


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _shade_baselines(ax, n_baselines: int) -> None:
    if n_baselines:
        ax.axvspan(-0.5, n_baselines - 0.5, color="grey", alpha=0.12, lw=0)
        ax.text(-0.45, 0.98, "baselines", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color="dimgrey")


def plot_summary(df: pd.DataFrame, path: Path, title: str) -> None:
    """Validation and test R2 per method, one bar per feature set."""
    order, sets = _method_order(df), _feature_sets(df)
    n_baselines = sum(df.drop_duplicates("method").set_index("method").loc[order, "type"] == "baseline")
    pos = np.arange(len(order))
    width = 0.8 / max(len(sets), 1)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.3 * len(order)) * 1.6, 5), sharey=True)
    for ax, split in zip(axes, ["val", "test"]):
        _shade_baselines(ax, n_baselines)
        for i, fs in enumerate(sets):
            sub = df[(df["split"] == split) & (df["feature_set"] == fs)].set_index("method")["r2"]
            ax.bar(pos + (i - (len(sets) - 1) / 2) * width, [sub.get(m, np.nan) for m in order],
                   width, label=fs, color=SET_COLOURS.get(fs))
        ax.set_xticks(pos, order, rotation=35, ha="right", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{'Validation' if split == 'val' else 'Test'} R²")
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="feature set", loc="upper right", ncol=len(labels),
               fontsize=9, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_detail(df: pd.DataFrame, path: Path, title: str) -> None:
    """Train / val / test R2 and RMSE per method, one panel per feature set."""
    sets, order = _feature_sets(df), _method_order(df)
    fig, axes = plt.subplots(2, len(sets), figsize=(5.5 * len(sets), 8), squeeze=False)
    for col, fs in enumerate(sets):
        sub = df[df["feature_set"] == fs]
        methods = [m for m in order if m in set(sub["method"])]
        n_baselines = sum(model_type(sub.loc[sub["method"] == m, "model"].iloc[0]) == "baseline"
                          for m in methods)
        pos = np.arange(len(methods))
        splits = [s for s in SPLITS if s in set(sub["split"])]
        w = 0.8 / max(len(splits), 1)
        for row, metric in enumerate(["r2", "rmse"]):
            ax = axes[row][col]
            _shade_baselines(ax, n_baselines)
            for i, sp in enumerate(splits):
                vals = sub[sub["split"] == sp].set_index("method")[metric]
                ax.bar(pos + (i - (len(splits) - 1) / 2) * w, [vals.get(m, np.nan) for m in methods],
                       w, label=sp, color=SPLIT_COLOURS[sp])
            ax.set_xticks(pos, methods, rotation=45, ha="right", fontsize=8)
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            ax.set_title(f"{fs} -- {'R²' if metric == 'r2' else 'RMSE'}")
            ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), fontsize=9, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_trials(models: list[str] | None) -> None:
    for model in ["xgboost", "svr"]:
        path = paths.TUNING_DIR / f"{model}_tuning_trials.csv"
        if (models and model not in models) or not path.exists():
            continue
        trials = pd.read_csv(path)
        cols = [c for c in ["grain", "feature_set", "trial", "train_r2", "r2", "rmse", "r2_gap",
                            "overfitting_risk"] if c in trials]
        print(f"\n--- {display_name(model)} tuning trials (train_r2 vs validation r2, per candidate) ---")
        print(trials[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grain", help="Only show this grain (daily/hourly)")
    parser.add_argument("--model", nargs="+", help="Only show these models (config names, e.g. ridge lstm)")
    parser.add_argument("--detail", action="store_true", help="Also print the full train/val/test tables")
    parser.add_argument("--trials", action="store_true", help="Also print every tuning trial")
    args = parser.parse_args()

    df = load_results()
    if args.grain:
        df = df[df["grain"] == args.grain]
    if args.model:
        df = df[df["model"].isin([*args.model, *BASELINE_METHODS])]
    if df.empty:
        raise SystemExit("No results match those filters.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "all_results.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 500)
    fmt = lambda x: f"{x:.3f}"  # noqa: E731

    for grain, group in df.groupby("grain"):
        summary = summary_table(group)
        summary.to_csv(OUT_DIR / f"summary_{grain}.csv")
        detail = detail_table(group)
        detail.to_csv(OUT_DIR / f"train_val_test_{grain}.csv")

        print(f"\n{'=' * 90}\n{grain.upper()} -- R² by method   "
              f"(choose on val, report test once; higher is better)\n{'=' * 90}")
        print(summary.to_string(float_format=fmt, na_rep="-"))
        print("\n  baseline = not ML (the bar to beat)    ML = models 1-7    '-' = not run yet")

        if args.detail:
            print(f"\n--- {grain} detail: train / val / test, one block per method ---")
            for method in detail.index.get_level_values("method").unique():
                block = detail.xs(method, level="method").droplevel("type")
                print(f"\n[{method}]")
                print(block.to_string(float_format=lambda x: f"{x:.4f}", na_rep="-"))

        delta = stage_delta(group)
        if not delta.empty:
            print("\n--- Default -> tuned (negative delta_rmse / positive delta_r2 = tuning helped) ---")
            print(delta.to_string(float_format=lambda x: f"{x:.4f}"))
            delta.to_csv(OUT_DIR / f"stage_delta_{grain}.csv")

        plot_summary(group, OUT_DIR / f"summary_{grain}.png", f"{grain} -- R² by method")
        plot_detail(group, OUT_DIR / f"train_val_test_{grain}.png", f"{grain} -- train / val / test")

    if args.trials:
        print_trials(args.model)
    print(f"\nSaved tables and charts to {OUT_DIR.relative_to(ROOT_DIR)}/ "
          "(summary_<grain>.png is the quickest overview; --detail prints the full tables)")


if __name__ == "__main__":
    main()
