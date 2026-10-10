"""
CO2/NO2 Traffic-Emissions Capstone -- Bootstrap confidence intervals

How sure can we be about each model's test score, and are the top models
really different or just tied? This script answers both from the row-by-row
predictions the train/tune stages save in results/predictions/. Nothing is
retrained -- it only re-scores saved predictions.

Method (block bootstrap):
  1. Split the test rows into blocks of one station x one calendar week.
     Whole weeks are resampled together because NO2 on neighbouring days /
     hours is correlated; resampling single rows would pretend there is more
     independent data than there is and give intervals that are too narrow.
  2. Draw that many blocks WITH replacement (some weeks twice, some not at
     all) and recompute R2 / RMSE / MAE on the resampled rows. Repeat
     --n-boot times (default 1000).
  3. The 95% interval is the middle 95% of those scores (2.5th-97.5th
     percentile).
  4. Every model is scored on the SAME resampled rows in each round
     (a paired bootstrap), so the difference between two models gets its own
     interval. If the interval for "best model minus this model" contains 0,
     the two are statistically indistinguishable (a tie).

Which version of each model: the tuned one if its predictions are saved,
otherwise the default one (--stage default / tuned forces one). Baselines
(B1/B2) don't use features, so their predictions are shared across feature
sets.

Outputs (results/evaluation/):
  bootstrap_<grain>_<split>.csv           every model: score + 95% interval
  bootstrap_<grain>_<split>_vs_best.csv   each model vs the best one in its
                                          feature set (paired difference)
  bootstrap_<grain>_<split>.png           R2 with 95% intervals, per feature set

Usage:
    python src/09_bootstrap_ci.py                       # both grains, test split
    python src/09_bootstrap_ci.py --grain hourly
    python src/09_bootstrap_ci.py --split val --n-boot 2000
Run AFTER the pipeline (it needs results/predictions/ and data/processed/splits/).
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

FEATURE_SETS = ["exogenous", "autoregressive", "all"]
BASELINES = ["naive_persistence", "naive_seasonal_climatology"]
DISPLAY = {
    "naive_persistence": "B1 Persistence",
    "naive_seasonal_climatology": "B2 Seasonal climatology",
    "ridge": "Ridge", "decision_tree": "Decision tree", "random_forest": "Random forest",
    "xgboost": "XGBoost", "svr": "SVR", "lstm": "LSTM", "gru": "GRU",
}
ORDER = list(DISPLAY)
FILE_RE = re.compile(r"^(?P<model>.+)_(?P<stage>default|tuned)_(?P<grain>daily|hourly)_"
                     r"(?P<feature_set>exogenous|autoregressive|all)_(?P<split>val|test)\.csv\.gz$")


# --- loading -------------------------------------------------------------------
def find_prediction_files(grain: str, split: str) -> pd.DataFrame:
    rows = []
    for path in sorted(paths.PREDICTIONS_DIR.glob(f"*_{grain}_*_{split}.csv.gz")):
        match = FILE_RE.match(path.name)
        if match:
            rows.append({**match.groupdict(), "path": path})
    return pd.DataFrame(rows)


def choose_versions(files: pd.DataFrame, stage: str) -> pd.DataFrame:
    """One prediction file per (model, feature_set): tuned if available
    (stage='best'), or exactly the stage asked for."""
    if stage != "best":
        return files[files["stage"] == stage]
    files = files.assign(_rank=(files["stage"] != "tuned").astype(int))
    return files.sort_values("_rank").drop_duplicates(["model", "feature_set"]).drop(columns="_rank")


def load_blocks(grain: str, split: str, n_rows: int) -> tuple[np.ndarray, pd.Series]:
    """Block id (station x ISO week) for every row of the split, in row order."""
    path = paths.SPLITS_DIR / f"{grain}_{split}.csv"
    meta = pd.read_csv(path, usecols=["station_id", "timestamp"], dtype={"station_id": str})
    if len(meta) != n_rows:
        raise SystemExit(f"{path.name} has {len(meta)} rows but the predictions have {n_rows} -- "
                         "the predictions come from a different run. Rerun the pipeline.")
    week = pd.to_datetime(meta["timestamp"]).dt.isocalendar()
    block_key = meta["station_id"] + "_" + week["year"].astype(str) + "W" + week["week"].astype(str)
    return pd.factorize(block_key)[0], meta["station_id"]


def load_predictions(chosen: pd.DataFrame, station_ids: pd.Series) -> tuple[np.ndarray, dict]:
    """y_true (shared) and {(model, stage, feature_set): y_pred}."""
    y_true, preds = None, {}
    for item in chosen.itertuples():
        frame = pd.read_csv(item.path, dtype={"station_id": str}).sort_values("row")
        if len(frame) != len(station_ids) or not (frame["station_id"].to_numpy()
                                                 == station_ids.to_numpy()).all():
            raise SystemExit(f"{item.path.name} doesn't line up with the current split file -- "
                             "it comes from a different run. Rerun the pipeline.")
        if y_true is None:
            y_true = frame["y_true"].to_numpy(float)
        elif not np.allclose(y_true, frame["y_true"].to_numpy(float)):
            raise SystemExit(f"{item.path.name}: y_true differs from the other files.")
        preds[(item.model, item.stage, item.feature_set)] = frame["y_pred"].to_numpy(float)
    return y_true, preds


# --- scoring -------------------------------------------------------------------
def scores(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    """(r2, rmse, mae) -- works on a single sample (1-D) or many at once (2-D,
    one resample per row)."""
    err = p - y
    ss_res = np.sum(err ** 2, axis=-1)
    ss_tot = np.sum((y - y.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
    n = y.shape[-1]
    return 1 - ss_res / ss_tot, np.sqrt(ss_res / n), np.sum(np.abs(err), axis=-1) / n


def bootstrap_indices(blocks: np.ndarray, n_boot: int, seed: int):
    """Yield row indices for each bootstrap round (whole blocks resampled)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(blocks, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(blocks[order])) + 1]
    members = np.split(order, starts[1:])
    n_blocks = len(members)
    for _ in range(n_boot):
        picked = rng.integers(0, n_blocks, n_blocks)
        yield np.concatenate([members[i] for i in picked])


def run(grain: str, split: str, n_boot: int, stage: str, seed: int) -> pd.DataFrame | None:
    files = find_prediction_files(grain, split)
    if files.empty:
        print(f"[{grain}/{split}] no prediction files in {paths.PREDICTIONS_DIR} -- skipped")
        return None
    chosen = choose_versions(files, stage)
    n_rows = len(pd.read_csv(chosen["path"].iloc[0], usecols=["row"]))
    blocks, station_ids = load_blocks(grain, split, n_rows)
    y, preds = load_predictions(chosen, station_ids)
    keys = sorted(preds, key=lambda k: (FEATURE_SETS.index(k[2]), ORDER.index(k[0])
                                       if k[0] in ORDER else 99))
    print(f"\n{grain} | {split}: {len(y):,} rows in {blocks.max() + 1} station-week blocks, "
          f"{len(keys)} models, {n_boot} bootstrap rounds", flush=True)

    # paired bootstrap: every model scored on the same resampled rows each round
    boot = {k: np.empty((n_boot, 3)) for k in keys}
    for b, idx in enumerate(bootstrap_indices(blocks, n_boot, seed)):
        yb = y[idx]
        for k in keys:
            boot[k][b] = scores(yb, preds[k][idx])

    rows = []
    for k in keys:
        point = scores(y, preds[k])
        lo, hi = np.percentile(boot[k], [2.5, 97.5], axis=0)
        model, model_stage, feature_set = k
        rows.append({
            "grain": grain, "split": split, "feature_set": feature_set,
            "model": model, "method": DISPLAY.get(model, model), "stage": model_stage,
            "r2": point[0], "r2_lo": lo[0], "r2_hi": hi[0],
            "rmse": point[1], "rmse_lo": lo[1], "rmse_hi": hi[1],
            "mae": point[2], "mae_lo": lo[2], "mae_hi": hi[2],
            "n_rows": len(y), "n_blocks": int(blocks.max() + 1),
        })
    table = pd.DataFrame(rows)

    # baselines apply to every feature set (they ignore features)
    base = table[table["model"].isin(BASELINES)]
    extra = [base.assign(feature_set=fs) for fs in FEATURE_SETS
             if fs in table["feature_set"].values and fs not in base["feature_set"].values]
    table = pd.concat([table, *extra], ignore_index=True)
    for (model, model_stage, fs) in [(r.model, r.stage, r.feature_set) for r in base.itertuples()]:
        for other in FEATURE_SETS:
            boot.setdefault((model, model_stage, other), boot[(model, model_stage, fs)])

    vs_best = compare_to_best(table, boot)
    out = paths.EVALUATION_DIR
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"bootstrap_{grain}_{split}.csv", index=False)
    vs_best.to_csv(out / f"bootstrap_{grain}_{split}_vs_best.csv", index=False)
    plot(table, grain, split, out / f"bootstrap_{grain}_{split}.png")
    print_summary(table, vs_best)
    return table


def compare_to_best(table: pd.DataFrame, boot: dict) -> pd.DataFrame:
    """Per feature set: the best ML model (highest R2) minus every other
    model, with a paired 95% interval for that difference."""
    rows = []
    for fs, group in table.groupby("feature_set", sort=False):
        ml = group[~group["model"].isin(BASELINES)]
        if ml.empty:
            continue
        best = ml.loc[ml["r2"].idxmax()]
        best_boot = boot[(best["model"], best["stage"], fs)][:, 0]
        for r in group.itertuples():
            if r.model == best["model"]:
                continue
            diff = best_boot - boot[(r.model, r.stage, fs)][:, 0]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            rows.append({
                "feature_set": fs, "best": best["method"], "model": r.method, "stage": r.stage,
                "r2_difference": best["r2"] - r.r2, "diff_lo": lo, "diff_hi": hi,
                "verdict": "not distinguishable from best" if lo <= 0 else "worse than best",
            })
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, grain: str, split: str, path: Path) -> None:
    sets = [fs for fs in FEATURE_SETS if fs in table["feature_set"].values]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.2 * len(sets), 0.45 * table["model"].nunique() + 1.6),
                             sharey=True, squeeze=False)
    methods = [DISPLAY[m] for m in ORDER if m in table["model"].values]
    for ax, fs in zip(axes[0], sets):
        group = table[table["feature_set"] == fs].set_index("method").reindex(methods)
        ypos = np.arange(len(methods))[::-1]
        is_base = group["model"].isin(BASELINES).to_numpy()
        for color, mask in (("#8a8a8a", is_base), ("#2f6db5", ~is_base)):
            ax.errorbar(group["r2"][mask], ypos[mask],
                        xerr=[(group["r2"] - group["r2_lo"])[mask], (group["r2_hi"] - group["r2"])[mask]],
                        fmt="o", color=color, ecolor=color, capsize=3, markersize=5, linewidth=1.4)
        ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=0)
        ax.set_yticks(np.arange(len(methods))[::-1], methods)
        ax.set_title(fs, fontsize=11)
        ax.set_xlabel(f"{split} $R^2$ (95% CI)")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"{grain.capitalize()} NO$_2$: {split} $R^2$ with 95% block-bootstrap intervals", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_summary(table: pd.DataFrame, vs_best: pd.DataFrame) -> None:
    for fs, group in table.groupby("feature_set", sort=False):
        print(f"  {fs}:")
        for r in group.sort_values("r2", ascending=False).itertuples():
            verdict = vs_best[(vs_best["feature_set"] == fs) & (vs_best["model"] == r.method)]
            note = verdict["verdict"].iloc[0] if len(verdict) else "BEST"
            print(f"    {r.method:24s} {r.stage:7s} R2 {r.r2:6.3f} [{r.r2_lo:6.3f}, {r.r2_hi:6.3f}]"
                  f"   RMSE {r.rmse:.3f} [{r.rmse_lo:.3f}, {r.rmse_hi:.3f}]   {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals from saved predictions.")
    parser.add_argument("--grain", nargs="+", choices=["daily", "hourly"], default=["daily", "hourly"])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--stage", choices=["best", "tuned", "default"], default="best",
                        help="best = tuned where available, else default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=paths.CONFIG_PATH,
                        help="accepted for run_pipeline.py; not used (scores saved predictions only)")
    args = parser.parse_args()
    for grain in args.grain:
        run(grain, args.split, args.n_boot, args.stage, args.seed)
    print(f"\nSaved to {paths.EVALUATION_DIR.relative_to(paths.ROOT_DIR)}/bootstrap_*")


if __name__ == "__main__":
    main()