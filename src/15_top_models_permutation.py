"""
CO2/NO2 Traffic-Emissions Capstone -- Do the top 3 models rely on the same drivers?

SHAP (stage 08) explains XGBoost in detail, but XGBoost is not the most
accurate model everywhere: on hourly data the GRU and LSTM score higher. This
script takes the THREE most accurate ML models for each grain and feature set
(by test R2, from the bootstrap table made by 09_bootstrap_ci.py) and runs the
same model-agnostic test on each: permutation importance.

Permutation importance: shuffle one feature family (all weather features,
all traffic features, ...) across the test rows, so its values no longer
match their rows, and re-score. The drop in R2 is how much the model depends
on that family. It needs only the model's predictions, so it works the same
way for XGBoost, SVR, LSTM and GRU. For LSTM / GRU the shuffle is applied to
every hour (or day) of the model's input window at once.

Outputs (results/permutation/):
  top3_families_<grain>_<feature_set>.csv   R2 drop per family for each top model,
                                            each model's rank, and XGBoost's SHAP share
  top3_agreement_<grain>_<feature_set>.csv  Spearman rank agreement between every pair
                                            of models (and with XGBoost's SHAP ranking)
  top3_traffic_parts_<grain>_<feature_set>.csv   the same for the parts of traffic
  top3_<grain>_<feature_set>.png            family importance, models side by side
  top3_window_<grain>_<feature_set>_<model>.csv/.png   LSTM / GRU only: which
                                            position of the input window matters
                                            (how far back the model looks)

Models: the tuned versions saved by the tune stage (results/saved_models/).
A tree / linear / SVR model without a matching saved file is refitted with
its default settings (and labelled so); an LSTM / GRU without one is skipped.

Usage:
    python src/15_top_models_permutation.py
    python src/15_top_models_permutation.py --grain hourly --feature-set exogenous --top 3
    python src/15_top_models_permutation.py --grain hourly --models gru xgboost
Run AFTER 09_bootstrap_ci.py (top models) and 08 / 13 (SHAP, for the comparison).
"""

from __future__ import annotations

import argparse
import itertools
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
from scripts.experiment_utils import (  # noqa: E402
    TABULAR_MODELS, TARGET, build, load_config, r2, read_split, weight_column,
)
from scripts.feature_families import group_features, group_traffic_parts  # noqa: E402
from scripts.model_utils import fit_with_optional_weight, get_feature_columns  # noqa: E402
from scripts.models import display_name  # noqa: E402

OUT_DIR = paths.RESULTS_DIR / "permutation"
SEQUENCE_MODELS = ["lstm", "gru"]
BASELINES = ["naive_persistence", "naive_seasonal_climatology"]
UNIT = {"daily": "days", "hourly": "hours"}


def top_models(grain: str, feature_set: str, n: int) -> list[str]:
    """The n ML models with the highest test R2 (bootstrap table from stage 09)."""
    path = paths.EVALUATION_DIR / f"bootstrap_{grain}_test.csv"
    if not path.exists():
        return []
    table = pd.read_csv(path)
    table = table[(table["feature_set"] == feature_set) & ~table["model"].isin(BASELINES)]
    return table.sort_values("r2", ascending=False)["model"].head(n).tolist()


# --- each model wrapped so it can be scored after shuffling ----------------------

class TabularScorer:
    """XGBoost / random forest / ridge / decision tree / SVR on single rows."""

    def __init__(self, name, grain, feature_set, features, config, test, rows):
        import joblib

        self.label = "tuned"
        path = paths.SAVED_MODELS_DIR / f"{name}_{grain}_{feature_set}.joblib"
        model = joblib.load(path) if path.exists() else None
        saved_names = list(getattr(model, "feature_names_in_", []))
        if model is None or (saved_names and saved_names != features) or \
                getattr(model, "n_features_in_", len(features)) != len(features):
            train = read_split(grain, "train")
            model = build(name, config)
            fit_with_optional_weight(model, train[features], train[TARGET], train[weight_column(grain)])
            self.label = "default, refitted (no matching tuned model saved)"
        self.model = model
        self.X = test.iloc[rows][features].reset_index(drop=True)
        self.features = features

    def score(self, y, columns=None, rng=None) -> float:
        X = self.X
        if columns:
            X = X.copy()
            X[columns] = self.X[columns].to_numpy()[rng.permutation(len(X))]
        return r2(y, self.model.predict(X))


class SequenceScorer:
    """LSTM / GRU: each prediction looks at a window of past rows."""

    def __init__(self, name, grain, feature_set, frames, rows):
        import torch

        from scripts.lstm import build_sequences
        if name == "gru":
            from scripts.gru import _Network
        else:
            from scripts.lstm import _Network

        path = paths.SAVED_MODELS_DIR / f"{name}_{grain}_{feature_set}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found -- run the tune stage")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or "model_params" not in saved:
            raise FileNotFoundError(f"{path.name} was saved without its settings "
                                    f"(older version of tune_{name}.py) -- rerun the tune stage")
        p = saved["model_params"]
        self.network = _Network(p["n_inputs"], p["hidden_size"], p["num_layers"], p["dropout"])
        self.network.load_state_dict(saved["state_dict"])
        self.network.eval()
        self.seq_len, self.mean, self.std = int(p["seq_len"]), saved["target_mean"], saved["target_std"]
        self.features = list(saved["features"])
        missing = [f for f in self.features if f not in frames["test"].columns]
        if missing:
            raise FileNotFoundError(f"{path.name} was trained on different data "
                                    f"({len(missing)} features missing) -- rerun the tune stage")
        self.data = build_sequences(frames, self.features, grain, self.seq_len)
        self.rows = self.data.split_rows["test"][rows]
        window = self.data.windows[torch.as_tensor(self.rows)].numpy()
        self.history = np.unique(window[window >= 0])  # every row the windows look at
        self.original = self.data.features.clone()
        self.label = "tuned"

    def _predict(self, x) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                out.append(self.network(x[start:start + 4096]).numpy())
        return np.concatenate(out) * self.std + self.mean

    def score(self, y, columns=None, rng=None) -> float:
        from scripts.lstm import _gather
        if not columns:
            return r2(y, self._predict(_gather(self.data, self.rows)))
        idx = [self.features.index(c) for c in columns]
        block = np.ix_(self.history, idx)
        self.data.features[block] = self.original[np.ix_(rng.permutation(self.history), idx)]
        try:
            return r2(y, self._predict(_gather(self.data, self.rows)))
        finally:
            self.data.features[block] = self.original[block]

    def window_importance(self, y, base, repeats, rng) -> pd.DataFrame:
        """R2 drop when ONE position of the input window is shuffled across rows."""
        import torch

        from scripts.lstm import _gather
        x = _gather(self.data, self.rows)
        out = []
        for pos in range(self.seq_len):
            drops = []
            for _ in range(repeats):
                xp = x.clone()
                xp[:, pos, :] = x[torch.as_tensor(rng.permutation(len(x))), pos, :]
                drops.append(base - r2(y, self._predict(xp)))
            out.append({"steps_back": self.seq_len - 1 - pos,
                        "r2_drop": float(np.mean(drops)), "r2_drop_sd": float(np.std(drops))})
        return pd.DataFrame(out).sort_values("steps_back")


# --- the experiment -------------------------------------------------------------

def drops_for(scorer, groups: dict[str, list[str]], y, base, repeats, rng) -> dict[str, tuple]:
    out = {}
    for name, cols in groups.items():
        cols = [c for c in cols if c in scorer.features]
        if not cols:
            continue
        d = [base - scorer.score(y, cols, rng) for _ in range(repeats)]
        out[name] = (float(np.mean(d)), float(np.std(d)), len(cols))
    return out


def run(grain: str, feature_set: str, args, config: dict) -> None:
    models = args.models or top_models(grain, feature_set, args.top)
    if not models:
        print(f"[{grain}/{feature_set}] no bootstrap_{grain}_test.csv -- run 09_bootstrap_ci.py first. Skipped.")
        return
    frames = {s: read_split(grain, s) for s in ("train", "val", "test")}
    test = frames["test"]
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(len(test), min(args.max_rows, len(test)), replace=False))
    y = test[TARGET].to_numpy(float)[rows]
    features = get_feature_columns(grain, feature_set)
    families_of = group_features(features)
    parts_of = group_traffic_parts(features)
    print(f"\n{grain} | {feature_set}: top {len(models)} models by test R2: "
          + ", ".join(display_name(m) for m in models) + f"  ({len(rows):,} test rows, "
          f"{args.repeats} shuffles each)", flush=True)

    fam_rows, part_rows, tag = [], [], f"{grain}_{feature_set}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in models:
        try:
            scorer = (SequenceScorer(name, grain, feature_set, frames, rows) if name in SEQUENCE_MODELS
                      else TabularScorer(name, grain, feature_set, features, config, test, rows))
        except FileNotFoundError as error:
            print(f"  {display_name(name)}: skipped -- {error}")
            continue
        base = scorer.score(y)
        print(f"  {display_name(name)} ({scorer.label}): R2 {base:.3f}", flush=True)
        for family, (drop, sd, n) in drops_for(scorer, families_of, y, base, args.repeats, rng).items():
            fam_rows.append({"model": name, "method": display_name(name), "family": family,
                             "n_features": n, "r2_drop": drop, "r2_drop_sd": sd, "model_r2": base})
            print(f"      {family:20s} R2 drop {drop:6.3f} +- {sd:.3f}", flush=True)
        for part, (drop, sd, n) in drops_for(scorer, parts_of, y, base, args.repeats, rng).items():
            part_rows.append({"model": name, "method": display_name(name), "part": part,
                              "n_features": n, "r2_drop": drop, "r2_drop_sd": sd})
        if name in SEQUENCE_MODELS:
            window = scorer.window_importance(y, base, args.repeats, rng)
            window.to_csv(OUT_DIR / f"top3_window_{tag}_{name}.csv", index=False)
            plot_window(window, name, grain, feature_set, OUT_DIR / f"top3_window_{tag}_{name}.png")
            top = window.nlargest(3, "r2_drop")["steps_back"].tolist()
            print(f"      input window: most important = {top} {UNIT[grain]} back (0 = current)")

    if not fam_rows:
        return
    families = pd.DataFrame(fam_rows)
    families["rank"] = families.groupby("model")["r2_drop"].rank(ascending=False, method="min").astype(int)
    shap = load_shap_shares(grain, feature_set)
    if shap is not None:
        families = families.merge(shap, on="family", how="left")
    families.to_csv(OUT_DIR / f"top3_families_{tag}.csv", index=False)
    if part_rows:
        parts = pd.DataFrame(part_rows)
        parts["rank"] = parts.groupby("model")["r2_drop"].rank(ascending=False, method="min").astype(int)
        parts.to_csv(OUT_DIR / f"top3_traffic_parts_{tag}.csv", index=False)

    agreement = rank_agreement(families)
    agreement.to_csv(OUT_DIR / f"top3_agreement_{tag}.csv", index=False)
    print("  rank agreement (Spearman, 1 = same order of families):")
    for r in agreement.itertuples():
        print(f"      {r.a:22s} vs {r.b:22s} {r.spearman:5.2f}")
    plot_families(families, grain, feature_set, OUT_DIR / f"top3_{tag}.png")


def load_shap_shares(grain: str, feature_set: str) -> pd.DataFrame | None:
    path = paths.RESULTS_DIR / "shap" / f"groups_{grain}_{feature_set}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)[["family", "share_of_total"]].rename(
        columns={"share_of_total": "xgboost_shap_share"})


def rank_agreement(families: pd.DataFrame) -> pd.DataFrame:
    wide = families.pivot(index="family", columns="method", values="r2_drop")
    if "xgboost_shap_share" in families:
        wide["XGBoost SHAP"] = families.groupby("family")["xgboost_shap_share"].first()
    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        both = wide[[a, b]].dropna()
        if len(both) >= 3:
            rows.append({"a": a, "b": b, "families": len(both),
                         "spearman": spearmanr(both[a], both[b]).statistic})
    return pd.DataFrame(rows)


def plot_families(families: pd.DataFrame, grain: str, feature_set: str, path: Path) -> None:
    wide = families.pivot(index="family", columns="method", values="r2_drop")
    sd = families.pivot(index="family", columns="method", values="r2_drop_sd")
    order = wide.mean(axis=1).sort_values().index
    wide, sd = wide.loc[order], sd.loc[order]
    y = np.arange(len(wide))
    h = 0.8 / len(wide.columns)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(wide) * max(1, len(wide.columns) / 2) + 1.8))
    for i, method in enumerate(wide.columns):
        ax.barh(y - 0.4 + h * (i + 0.5), wide[method], h, xerr=sd[method], label=method)
    ax.set_yticks(y, wide.index)
    ax.set_xlabel("drop in test R$^2$ when the family is shuffled (higher = relied on more)")
    ax.set_title(f"What the top models rely on -- {grain}, {feature_set} features", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_window(window: pd.DataFrame, name: str, grain: str, feature_set: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(window["steps_back"], window["r2_drop"], yerr=window["r2_drop_sd"], color="#2f6db5")
    ax.set_xlabel(f"{UNIT[grain]} before the predicted {UNIT[grain][:-1]} (0 = the predicted one)")
    ax.set_ylabel("drop in test R$^2$")
    ax.set_title(f"How far back the {display_name(name).split(' ', 1)[-1]} looks -- "
                 f"{grain}, {feature_set} features", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation importance for the top models.")
    parser.add_argument("--grain", nargs="+", choices=["daily", "hourly"], default=["daily", "hourly"])
    parser.add_argument("--feature-set", nargs="+", dest="feature_sets",
                        choices=["exogenous", "autoregressive", "all"], default=["exogenous", "all"])
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=None,
                        help="explain these models instead of the top ones, e.g. --models gru lstm xgboost")
    parser.add_argument("--repeats", type=int, default=5)
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