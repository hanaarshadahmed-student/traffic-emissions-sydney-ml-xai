"""
CO2/NO2 Traffic-Emissions Capstone -- Evaluation / model comparison

Shows how every model does on TRAIN, VALIDATION and TEST, side by side,
for both its baseline run (06_run_models.py) and its tuned run
(models/fine_tuning/*), per grain and feature set.

It only reads data/processed/model_results/results.json -- nothing is
retrained -- so it's instant and safe to run as often as you like.

Reading the table:
  train >> val          the model is overfitting (memorising train)
  val ~= test           validation was a fair stand-in for unseen data
  val >> test           the model was (indirectly) tuned to val, or the
                        test period genuinely behaves differently

Outputs (data/processed/model_results/evaluation/):
  all_results.csv                 every results.json row, flattened
  train_val_test_<grain>.csv      model x stage rows, metric x split columns
  stage_delta_<grain>.csv         tuned minus baseline, per split
  train_val_test_<grain>.png      R2 and RMSE bars, train/val/test per model

Usage:
    python src/07_evaluation.py
    python src/07_evaluation.py --grain daily --model random_forest xgboost
    python src/07_evaluation.py --trials      # + every xgboost/svr tuning trial
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

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "data" / "processed" / "model_results"
RESULTS_PATH = RESULTS_DIR / "results.json"
OUT_DIR = RESULTS_DIR / "evaluation"

STAGES = ["baseline", "tuned"]
SPLITS = ["train", "val", "test"]
METRICS = [("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R2")]
KEY = ["model", "stage", "grain", "feature_set", "split"]


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise SystemExit(
            f"{RESULTS_PATH} not found -- run src/06_run_models.py "
            "(and/or models/train_xgboost.py, models/train_svr.py) first."
        )
    df = pd.DataFrame(json.loads(RESULTS_PATH.read_text()))

    # Rows from before `stage` existed: tuning scripts always set tuned=True.
    tuned = df["tuned"].eq(True) if "tuned" in df else pd.Series(False, index=df.index)
    inferred = pd.Series(np.where(tuned, "tuned", "baseline"), index=df.index)
    df["stage"] = df["stage"].fillna(inferred) if "stage" in df else inferred

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
    return df[KEY + [m for m, _ in METRICS]].copy()


def wide_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows (feature_set, model, stage); columns (metric, split)."""
    table = df.pivot_table(index=["feature_set", "model", "stage"], columns="split",
                           values=[m for m, _ in METRICS])
    splits = [s for s in SPLITS if s in table.columns.get_level_values(1)]
    table = table.reindex(columns=pd.MultiIndex.from_product([[m for m, _ in METRICS], splits]))
    table.columns = pd.MultiIndex.from_tuples([(label, s) for m, s in table.columns
                                               for mm, label in METRICS if mm == m])
    if ("R2", "train") in table and ("R2", "val") in table:
        table[("gap", "train-val R2")] = table[("R2", "train")] - table[("R2", "val")]
    if ("R2", "val") in table and ("R2", "test") in table:
        table[("gap", "val-test R2")] = table[("R2", "val")] - table[("R2", "test")]
    stage_rank = {s: i for i, s in enumerate(STAGES)}
    order = sorted(table.index, key=lambda ix: (ix[0], ix[1], stage_rank.get(ix[2], 99)))
    return table.loc[order]


def stage_delta(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index=["feature_set", "model", "split"], columns="stage",
                          values=["rmse", "r2"])
    if not {"baseline", "tuned"} <= set(wide.columns.get_level_values(1)):
        return pd.DataFrame()
    out = pd.DataFrame({
        "baseline_rmse": wide[("rmse", "baseline")], "tuned_rmse": wide[("rmse", "tuned")],
        "baseline_r2": wide[("r2", "baseline")], "tuned_r2": wide[("r2", "tuned")],
    })
    out["delta_rmse"] = out["tuned_rmse"] - out["baseline_rmse"]
    out["delta_r2"] = out["tuned_r2"] - out["baseline_r2"]
    return out.dropna(subset=["baseline_rmse", "tuned_rmse"])


def plot(df: pd.DataFrame, path: Path, title: str) -> None:
    feature_sets = sorted(df["feature_set"].unique())
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    fig, axes = plt.subplots(2, len(feature_sets), figsize=(5.5 * len(feature_sets), 8), squeeze=False)
    for col, fs in enumerate(feature_sets):
        sub = df[df["feature_set"] == fs]
        groups = sorted(sub.groupby(["model", "stage"]).groups,
                        key=lambda g: (g[0], STAGES.index(g[1]) if g[1] in STAGES else 99))
        pos = np.arange(len(groups))
        splits = [s for s in SPLITS if s in set(sub["split"])]
        w = 0.8 / max(len(splits), 1)
        for row, metric in enumerate(["r2", "rmse"]):
            ax = axes[row][col]
            for i, sp in enumerate(splits):
                vals = [sub[(sub.model == m) & (sub.stage == st) & (sub.split == sp)][metric].mean()
                        for m, st in groups]
                ax.bar(pos + (i - (len(splits) - 1) / 2) * w, vals, w, label=sp, color=colors[sp])
            ax.set_xticks(pos, [f"{m}\n{st}" for m, st in groups], rotation=45, ha="right", fontsize=8)
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            ax.set_title(f"{fs} -- {'R²' if metric == 'r2' else 'RMSE'}")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_trials(models: list[str] | None) -> None:
    for model in ["xgboost", "svr"]:
        path = RESULTS_DIR / f"{model}_tuning_trials.csv"
        if (models and model not in models) or not path.exists():
            continue
        trials = pd.read_csv(path)
        cols = [c for c in ["grain", "feature_set", "trial", "train_r2", "r2", "rmse", "r2_gap",
                            "overfitting_risk"] if c in trials]
        print(f"\n--- {model} tuning trials (train_r2 vs validation r2, per candidate) ---")
        print(trials[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grain", help="Only show this grain (daily/hourly)")
    parser.add_argument("--model", nargs="+", help="Only show these models")
    parser.add_argument("--trials", action="store_true", help="Also print every tuning trial")
    args = parser.parse_args()

    df = load_results()
    if args.grain:
        df = df[df["grain"] == args.grain]
    if args.model:
        df = df[df["model"].isin(args.model)]
    if df.empty:
        raise SystemExit("No results match those filters.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "all_results.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    fmt = lambda x: f"{x:.4f}"  # noqa: E731

    for grain, group in df.groupby("grain"):
        table = wide_table(group)
        table.to_csv(OUT_DIR / f"train_val_test_{grain}.csv")
        print(f"\n{'=' * 100}\n{grain.upper()} grain -- train / val / test\n{'=' * 100}")
        for fs in table.index.get_level_values(0).unique():
            print(f"\n[{fs}]")
            print(table.loc[fs].to_string(float_format=fmt, na_rep="-"))

        missing = table[table.isna().any(axis=1)]
        if len(missing):
            print("\n  '-' = that split hasn't been scored yet for that model/stage; "
                  "re-run it (06_run_models.py or the tuning script) to fill it in.")

        delta = stage_delta(group)
        if not delta.empty:
            print("\n--- Baseline -> tuned (negative delta_rmse / positive delta_r2 = tuning helped) ---")
            print(delta.to_string(float_format=fmt))
            delta.to_csv(OUT_DIR / f"stage_delta_{grain}.csv")

        plot(group, OUT_DIR / f"train_val_test_{grain}.png", f"{grain} -- train / val / test")

    if args.trials:
        print_trials(args.model)
    print(f"\nSaved tables and plots to {OUT_DIR.relative_to(ROOT_DIR)}/")


if __name__ == "__main__":
    main()