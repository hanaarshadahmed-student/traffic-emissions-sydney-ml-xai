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

Outputs (results/latest/shap/):
  importance_<grain>_<feature_set>.csv      every feature: importance, share, direction
  summary_<grain>_<feature_set>.png         beeswarm: effect and value, top features
  bar_<grain>_<feature_set>.png             mean |SHAP| bar chart, top features
  dependence_<grain>_<feature_set>_<feature>.png   effect against value, top features
  groups_<grain>_<feature_set>.csv          importance summed by feature family
  shap_values_<grain>_<feature_set>.csv.gz  SHAP value of every explained row
  interaction_traffic_wind_<grain>_<feature_set>.png/.csv
                                            traffic effect vs traffic volume, one
                                            line per wind-speed third
  groups_by_road_type_<grain>_<feature_set>.png/.csv
  top_features_by_road_type_<grain>_<feature_set>.csv
                                            importance per road type, on TRAIN rows
                                            (the only split with several stations
                                            per road type); --no-road-type skips it

Families: "Traffic x weather" holds features that mix both (mainly
traffic_dispersion_proxy = log traffic / (1 + wind speed)), so the "Traffic"
share is pure traffic.

Usage:
    python src/09_shap_explainability.py
    python src/09_shap_explainability.py --grain hourly --feature-set exogenous
    python src/09_shap_explainability.py --max-rows 20000 --top 20
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
from scripts.feature_families import FAMILIES, family_of, traffic_part  # noqa: E402,F401
from models.models import build_model  # noqa: E402
from scripts.stations import station_label  # noqa: E402

OUT_DIR = paths.SHAP_DIR
TARGET_UNIT = "pphm"

def load_main_config(path: Path) -> dict:
    with open(path) as file:
        return yaml.safe_load(file) or {}


def get_model(grain: str, feature_set: str, main_config: dict, features: list[str]):
    """The tuned XGBoost if the tune stage saved one AND it was trained on
    exactly these features, else a fresh fit on train. The check matters
    because results/saved_models/ survives between runs: after a run with
    different data (e.g. a station excluded for a sensitivity check) the
    saved model can expect different columns than the current splits."""
    model_path = paths.SAVED_MODELS_DIR / f"xgboost_{grain}_{feature_set}.joblib"
    reason = "no tuned model saved yet"
    if model_path.exists():
        model = joblib.load(model_path)
        saved_features = list(getattr(model, "feature_names_in_", []))
        if saved_features == list(features):
            return model, f"tuned model ({model_path.name})"
        reason = (f"saved {model_path.name} was trained on {len(saved_features)} features, "
                  f"current data has {len(features)} -- from a different run")
    params = ((main_config.get("training", {}).get("models", {}).get("xgboost") or {})
              .get("params") or {})
    model = build_model("xgboost", **params)
    X, y, weights = load_split(grain, "train", feature_set, with_weight=True)
    fit_with_optional_weight(model, X, y, weights)
    return model, f"default XGBoost refitted on train ({reason})"


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
    model, source = get_model(grain, feature_set, main_config, list(X.columns))

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

    traffic_wind_interaction(shap_values, X, grain, feature_set, args.split)
    traffic_parts(table, grain, feature_set)
    if args.waterfalls:
        single_predictions(explainer, shap_values, model, X, grain, feature_set, args.split)
    if args.interaction_rows > 0:
        interaction_strengths(explainer, X, grain, feature_set, args.interaction_rows)
    if args.by_road_type:
        shap_by_road_type(model, grain, feature_set, args)

    unit_col = f"mean_abs_shap_{TARGET_UNIT}"
    shown = table.head(10)[["rank", "feature", "family", unit_col, "share_of_total", "direction_corr"]]
    print(f"  top 10 features ({grain}, {feature_set}):")
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("  by feature family:")
    print(groups.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    table.insert(0, "feature_set", feature_set)
    table.insert(0, "grain", grain)
    return table


# --- C2: which PARTS of traffic matter -----------------------------------------
def traffic_parts(table: pd.DataFrame, grain: str, feature_set: str) -> None:
    """SHAP importance of the Traffic family split into its parts (total volume,
    heavy vehicles, direction of flow, traffic x time, unexpected traffic --
    scripts/feature_families.py), plus Traffic x weather for comparison."""
    unit = f"mean_abs_shap_{TARGET_UNIT}"
    t = table.assign(part=table["feature"].map(traffic_part))
    t.loc[t["family"] == "Traffic x weather", "part"] = "Traffic x weather (e.g. traffic / wind)"
    t = t.dropna(subset=["part"])
    if t.empty:
        return
    parts = (t.groupby("part").agg(**{unit: (unit, "sum"), "share_of_total": ("share_of_total", "sum"),
                                       "n_features": ("feature", "size"),
                                       "top_feature": ("feature", "first")})
             .sort_values(unit, ascending=False).reset_index())
    parts["share_of_traffic"] = parts[unit] / parts[unit].sum()
    tag = f"{grain}_{feature_set}"
    parts.to_csv(OUT_DIR / f"traffic_parts_{tag}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(parts) + 1.4))
    ax.barh(parts["part"][::-1], parts["share_of_total"][::-1] * 100, color="#2f6db5")
    ax.set_xlabel("share of total SHAP importance (%)")
    ax.set_title(f"Which parts of traffic matter -- {grain}, {feature_set}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"traffic_parts_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"  traffic parts ({tag}): " + ", ".join(
        f"{r.part} {r.share_of_total:.1%}" for r in parts.itertuples()))


# --- C4: explaining SINGLE predictions --------------------------------------------
def _real_values(grain: str, X: pd.DataFrame) -> pd.DataFrame:
    """Undo 05's scaling so plots show real units (e.g. wind 0.8 m/s, not -1.2)."""
    raw = X.copy()
    path = paths.SPLITS_DIR / f"{grain}_scaler.joblib"
    if path.exists():
        scaler = joblib.load(path)
        names = list(getattr(scaler, "feature_names_in_", []))
        cols = [c for c in names if c in raw.columns]
        idx = [names.index(c) for c in cols]
        raw[cols] = raw[cols].to_numpy() * scaler.scale_[idx] + scaler.mean_[idx]
    return raw


def _pick_cases(meta: pd.DataFrame, raw: pd.DataFrame, grain: str) -> dict[str, int]:
    """Rows worth explaining one by one (index labels of X)."""
    y = meta["no2_pphm"]
    cases = {"highest_NO2": y.idxmax(),
             "typical": (y - y.median()).abs().idxmin()}
    wind = raw["wind_speed_ms"] if "wind_speed_ms" in raw else None
    if wind is not None:
        if grain == "hourly":
            when = pd.to_datetime(meta["timestamp"])
            rush = when.dt.hour.between(7, 9) & (when.dt.dayofweek < 5)  # weekday morning peak
            label = "rush_hour"
        else:
            rush = pd.to_datetime(meta["timestamp"]).dt.dayofweek < 5
            label = "weekday"
        calm, windy = wind <= wind.quantile(1 / 3), wind >= wind.quantile(2 / 3)
        for name, mask in ((f"calm_{label}", rush & calm), (f"windy_{label}", rush & windy)):
            rows = y[mask]
            if len(rows):
                cases[name] = (rows - rows.median()).abs().idxmin()
    return cases


def single_predictions(explainer, shap_values: np.ndarray, model, X: pd.DataFrame,
                       grain: str, feature_set: str, split: str) -> None:
    """SHAP waterfall plots for a few individual rows: the highest-NO2 row, a
    typical row, and a calm vs a windy rush hour (weekday, for daily data).
    Each shows how the model got from the average prediction to this one,
    feature by feature -- 'interpreting individual predictions' (RQ4)."""
    import shap

    meta = pd.read_csv(paths.SPLITS_DIR / f"{grain}_{split}.csv",
                       usecols=["station_id", "timestamp", "no2_pphm"],
                       dtype={"station_id": str}).loc[X.index]
    raw = _real_values(grain, X)
    position = {label: i for i, label in enumerate(X.index)}
    base = float(np.ravel(explainer.expected_value)[0])
    tag = f"{grain}_{feature_set}"
    rows = []
    for case, label in _pick_cases(meta, raw, grain).items():
        i = position[label]
        prediction = float(model.predict(X.iloc[[i]])[0])
        explanation = shap.Explanation(values=shap_values[i], base_values=base,
                                       data=raw.iloc[i].to_numpy(), feature_names=list(X.columns))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        m = meta.loc[label]
        plt.title(f"{case.replace('_', ' ')} -- {station_label(m.station_id)}, {m.timestamp}\n"
                  f"actual NO$_2$ {m.no2_pphm:.2f}, predicted {prediction:.2f} {TARGET_UNIT}",
                  fontsize=10)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"waterfall_{tag}_{case}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        order = np.argsort(-np.abs(shap_values[i]))[:10]
        for rank, j in enumerate(order, 1):
            rows.append({"case": case, "station_id": m.station_id, "station": station_label(m.station_id),
                         "timestamp": m.timestamp,
                         "actual_no2": m.no2_pphm, "predicted_no2": prediction, "base_value": base,
                         "rank": rank, "feature": X.columns[j], "family": family_of(X.columns[j]),
                         "value": raw.iloc[i, j], f"shap_{TARGET_UNIT}": shap_values[i, j]})
    pd.DataFrame(rows).to_csv(OUT_DIR / f"waterfall_{tag}.csv", index=False)
    print(f"  single predictions ({tag}): {', '.join(dict.fromkeys(r['case'] for r in rows))}")


# --- C5: how strongly features INTERACT (SHAP interaction values) --------------------
def interaction_strengths(explainer, X: pd.DataFrame, grain: str, feature_set: str,
                          n_rows: int) -> None:
    """SHAP interaction values split each feature's effect into its own part
    and the part that depends on another feature. Mean |interaction| per pair
    = how much the two features' effects depend on each other. Reports the top
    pairs, totals per family pair, and how much interaction runs between
    traffic and wind (the traffic x wind finding as a number). Costly
    (features^2 per row), so it uses a sample of --interaction-rows rows."""
    sample = X.sample(min(n_rows, len(X)), random_state=0)
    values = explainer.shap_interaction_values(sample)
    strength = np.abs(values).mean(axis=0)  # features x features
    np.fill_diagonal(strength, 0.0)
    names = list(X.columns)
    iu = np.triu_indices(len(names), 1)
    pairs = pd.DataFrame({"feature_a": np.array(names)[iu[0]], "feature_b": np.array(names)[iu[1]],
                          # both halves of the symmetric matrix belong to the pair
                          f"mean_abs_interaction_{TARGET_UNIT}": 2 * strength[iu]})
    col = f"mean_abs_interaction_{TARGET_UNIT}"
    pairs["family_a"] = pairs["feature_a"].map(family_of)
    pairs["family_b"] = pairs["feature_b"].map(family_of)
    total = pairs[col].sum() or 1.0
    pairs["share_of_interaction"] = pairs[col] / total
    pairs = pairs.sort_values(col, ascending=False)
    tag = f"{grain}_{feature_set}"
    pairs.head(30).to_csv(OUT_DIR / f"interactions_top_pairs_{tag}.csv", index=False)

    fam = pairs.assign(pair=[" x ".join(sorted(p)) for p in zip(pairs["family_a"], pairs["family_b"])])
    fam = (fam.groupby("pair")[[col, "share_of_interaction"]].sum()
           .sort_values(col, ascending=False).reset_index())
    fam.to_csv(OUT_DIR / f"interactions_families_{tag}.csv", index=False)

    is_traffic = lambda f: family_of(f) in ("Traffic", "Traffic x weather")  # noqa: E731
    is_wind = lambda f: "wind" in f.lower()  # noqa: E731
    tw = pairs[[(is_traffic(a) and is_wind(b)) or (is_traffic(b) and is_wind(a))
                for a, b in zip(pairs["feature_a"], pairs["feature_b"])]]
    print(f"  interactions ({tag}, {len(sample):,} rows): traffic x wind = "
          f"{tw['share_of_interaction'].sum():.1%} of all pairwise interaction; top pairs: "
          + "; ".join(f"{r.feature_a} x {r.feature_b} {r.share_of_interaction:.1%}"
                      for r in pairs.head(3).itertuples()))

    top = pairs.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([f"{a} x {b}" for a, b in zip(top["feature_a"], top["feature_b"])], top[col],
            color=["#c0392b" if (is_traffic(a) and is_wind(b)) or (is_traffic(b) and is_wind(a))
                   else "#2f6db5" for a, b in zip(top["feature_a"], top["feature_b"])])
    ax.set_xlabel(f"mean |SHAP interaction| ({TARGET_UNIT}); red = traffic x wind")
    ax.set_title(f"Strongest feature interactions -- {grain}, {feature_set}", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"interactions_top_pairs_{tag}.png", dpi=150)
    plt.close(fig)


TRAFFIC_FAMILIES = ("Traffic", "Traffic x weather")


def traffic_wind_interaction(shap_values: np.ndarray, X: pd.DataFrame, grain: str,
                             feature_set: str, split: str) -> None:
    """How much traffic pushes NO2 up, at low vs medium vs high wind.

    y = the combined SHAP effect of every traffic feature (Traffic and
    Traffic x weather families) for each row; x = traffic volume, in ten
    equal-sized bins (deciles, 1 = quietest); one line per wind-speed third.
    If wind disperses traffic NO2, the low-wind line should climb more
    steeply than the high-wind line. Features are scaled, so bins are used
    rather than raw vehicle counts."""
    tag = f"{grain}_{feature_set}"
    if not {"traffic_volume_total", "wind_speed_ms"} <= set(X.columns):
        print(f"  [traffic x wind] skipped for {tag}: needs traffic_volume_total and wind_speed_ms")
        return
    traffic_cols = [i for i, f in enumerate(X.columns) if family_of(f) in TRAFFIC_FAMILIES]
    frame = pd.DataFrame({
        "traffic_effect": shap_values[:, traffic_cols].sum(axis=1),
        "traffic_bin": pd.qcut(X["traffic_volume_total"].rank(method="first"), 10, labels=range(1, 11)),
        "wind": pd.qcut(X["wind_speed_ms"].rank(method="first"), 3,
                        labels=["low wind", "medium wind", "high wind"]),
    })
    table = (frame.groupby(["wind", "traffic_bin"], observed=True)["traffic_effect"]
             .agg(mean="mean", sem="sem", n="count").reset_index())
    table.to_csv(OUT_DIR / f"interaction_traffic_wind_{tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"low wind": "#c0392b", "medium wind": "#e69f00", "high wind": "#2f6db5"}
    print(f"  traffic x wind ({tag}): traffic effect, quietest -> busiest decile")
    for wind, group in table.groupby("wind", observed=True):
        x = group["traffic_bin"].astype(int)
        ax.plot(x, group["mean"], marker="o", color=colors[wind], label=wind)
        ax.fill_between(x, group["mean"] - 1.96 * group["sem"], group["mean"] + 1.96 * group["sem"],
                        color=colors[wind], alpha=0.15, linewidth=0)
        print(f"    {wind:12s} {group['mean'].iloc[0]:+.3f} -> {group['mean'].iloc[-1]:+.3f} {TARGET_UNIT} "
              f"(rise {group['mean'].iloc[-1] - group['mean'].iloc[0]:.3f})")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("traffic volume decile (1 = quietest, 10 = busiest)")
    ax.set_ylabel(f"combined traffic SHAP effect ({TARGET_UNIT})")
    ax.set_title(f"Traffic effect on NO$_2$ by wind speed -- {grain}, {feature_set} ({split} rows)")
    ax.legend(title="wind speed (thirds)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"interaction_traffic_wind_{tag}.png", dpi=150)
    plt.close(fig)


def shap_by_road_type(model, grain: str, feature_set: str, args) -> None:
    """Feature-family importance separately for each road type.

    Uses the TRAIN split by default: it is the only split with more than one
    station per road type (test rows are all Local Street; in validation,
    Highway is a single station). Explaining training rows shows what the
    model LEARNED for each road type -- fine for interpretation, but not a
    measure of accuracy. Each road type still covers only 2-5 stations, so
    differences can reflect individual stations as much as road type."""
    import shap

    split = args.road_type_split
    X, _ = load_split(grain, split, feature_set)
    meta = pd.read_csv(paths.SPLITS_DIR / f"{grain}_{split}.csv",
                       usecols=["station_id", "road_type_bucket"], dtype={"station_id": str})
    if len(X) > args.max_rows:
        X = X.sample(args.max_rows, random_state=42).sort_index()
    meta = meta.loc[X.index]
    shap_values = shap.TreeExplainer(model).shap_values(X)
    abs_shap = pd.DataFrame(np.abs(shap_values), columns=X.columns, index=X.index)

    family_rows, feature_rows = [], []
    for road_type, idx in meta.groupby("road_type_bucket").groups.items():
        mean_abs = abs_shap.loc[idx].mean()
        total = mean_abs.sum() or 1.0
        stations = sorted(meta.loc[idx, "station_id"].unique())
        info = {"road_type": road_type, "n_rows": len(idx), "n_stations": len(stations),
                "stations": "; ".join(station_label(s) for s in stations)}
        by_family = mean_abs.groupby([family_of(f) for f in mean_abs.index]).sum()
        for family, value in by_family.items():
            family_rows.append({**info, "family": family,
                                f"mean_abs_shap_{TARGET_UNIT}": value, "share_of_total": value / total})
        for rank, (feature, value) in enumerate(mean_abs.sort_values(ascending=False).head(10).items(), 1):
            feature_rows.append({**info, "rank": rank, "feature": feature, "family": family_of(feature),
                                 f"mean_abs_shap_{TARGET_UNIT}": value, "share_of_total": value / total})

    tag = f"{grain}_{feature_set}"
    families = pd.DataFrame(family_rows)
    families.to_csv(OUT_DIR / f"groups_by_road_type_{tag}.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(OUT_DIR / f"top_features_by_road_type_{tag}.csv", index=False)

    shares = families.pivot(index="family", columns="road_type", values="share_of_total").fillna(0)
    shares = shares.loc[shares.sum(axis=1).sort_values(ascending=False).index]
    labels = {r: f"{r}\n({n} stations)" for r, n in
              families.groupby("road_type")["n_stations"].first().items()}
    ax = (shares * 100).rename(columns=labels).plot.barh(figsize=(8, 0.6 * len(shares) + 1.5), width=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("share of total SHAP importance (%)")
    ax.set_ylabel("")
    ax.set_title(f"What drives NO$_2$ by road type -- {grain}, {feature_set} ({split} rows)")
    ax.legend(title="road type", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"groups_by_road_type_{tag}.png", dpi=150)
    plt.close("all")

    print(f"  by road type ({tag}, {split} rows -- share of importance):")
    print((shares * 100).round(1).to_string())


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
    parser.add_argument("--no-road-type", dest="by_road_type", action="store_false", default=None,
                        help="skip the SHAP-by-road-type breakdown")
    parser.add_argument("--road-type-split", choices=["train", "val", "test"], default=None,
                        help="rows used for the road-type breakdown (default: train)")
    parser.add_argument("--no-waterfalls", dest="waterfalls", action="store_false", default=None,
                        help="skip the single-prediction waterfall plots")
    parser.add_argument("--interaction-rows", type=int, default=None,
                        help="rows used for SHAP interaction values (default 1000; 0 = skip)")
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
    if args.by_road_type is None:
        args.by_road_type = bool(settings.get("by_road_type", True))
    args.road_type_split = args.road_type_split or settings.get("road_type_split", "train")
    if args.waterfalls is None:
        args.waterfalls = bool(settings.get("waterfalls", True))
    if args.interaction_rows is None:
        args.interaction_rows = int(settings.get("interaction_rows", 1000))
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