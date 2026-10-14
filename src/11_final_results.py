"""
Final results: copies the ~50 files the report uses from a saved run into results/, grouped by report section with a README explaining each file. Nothing is recomputed. Runs automatically after a complete pipeline run.

Usage:
    python src/11_final_results.py                          # newest saved full run
    python src/11_final_results.py --run working/runs/<run_id>
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paths  # noqa: E402
from scripts.feature_families import family_of  # noqa: E402
from scripts.stations import STATIONS, station_label  # noqa: E402

GRAINS = ["daily", "hourly"]


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """pd.read_csv, but an empty file (e.g. no stations excluded hourly) gives an empty table."""
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
FEATURE_SETS = ["exogenous", "all"]   # traffic + weather only, and with NO2 history
TOP_MODELS = ["gru", "lstm", "xgboost"]
# everything this script writes at the top of results/ (anything else there = not ours)
OWN_ENTRIES = {"README.md", "run", "1_data", "2_models", "3_drivers", "4_explainability", "5_errors", ".gitkeep"}


@functools.lru_cache(maxsize=1)
def _feature_engineering():
    spec = importlib.util.spec_from_file_location("fe", paths.ROOT_DIR / "src" / "04_feature_engineering.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _classify_road_type(name: str) -> str:
    """Same Highway / Major Road / Local Street rule as 04_feature_engineering.py."""
    return _feature_engineering().classify_named_road_type(name)


class Builder:
    def __init__(self, run_dir: Path, out_dir: Path):
        self.run = run_dir
        self.out = out_dir
        self.latest = run_dir / "outputs" / "latest"
        self.experiments = run_dir / "outputs" / "experiments"
        self.manifests = run_dir / "manifests"
        self.readme: list[tuple[str, str, str, str, str]] = []   # folder, file, what, rq, source
        self.missing: list[str] = []

    # ----- helpers ---------------------------------------------------------
    def _note(self, folder, name, what, rq, source):
        self.readme.append((folder, name, what, rq, source))

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self.run)).replace("\\", "/")

    def copy(self, folder: str, name: str, source: Path, what: str, rq: str) -> None:
        if not source.exists():
            self.missing.append(f"{folder}/{name}  (no {self._rel(source)})")
            return
        target = self.out / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        self._note(folder, name, what, rq, self._rel(source))

    def copy_first(self, folder: str, name: str, sources: list[Path], what: str, rq: str) -> None:
        """First source that exists (e.g. a dependence plot whose file name
        depends on which wind feature ranked highest)."""
        found = next((s for s in sources if s.exists()), None)
        if found is None:
            self.missing.append(f"{folder}/{name}  (none of {[self._rel(s) for s in sources]})")
            return
        self.copy(folder, name, found, what, rq)

    def stack(self, folder: str, name: str, parts: list[tuple[dict, Path]], what: str, rq: str,
              transform=None) -> None:
        """Stack several CSVs into one, adding the given columns (grain,
        feature_set, ...) in front. Columns that already exist are kept."""
        frames, sources = [], []
        for labels, path in parts:
            if not path.exists():
                continue
            frame = read_csv(path)
            if frame.empty:
                continue
            if transform:
                frame = transform(frame, labels)
            for position, (column, value) in enumerate(labels.items()):
                if column not in frame.columns:
                    frame.insert(position, column, value)
            frames.append(frame)
            sources.append(self._rel(path))
        if not frames:
            self.missing.append(f"{folder}/{name}  (no source files)")
            return
        self.write(folder, name, pd.concat(frames, ignore_index=True), what, rq, "; ".join(sources))

    def write(self, folder: str, name: str, table: pd.DataFrame, what: str, rq: str, source: str) -> None:
        target = self.out / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target, index=False)
        self._note(folder, name, what, rq, source)

    def split_manifest(self) -> dict:
        path = self.manifests / "split_manifest.json"
        return json.loads(path.read_text()) if path.exists() else {}

    # ----- folders ---------------------------------------------------------
    def build_run(self) -> None:
        f, rq = "run", "replication"
        self.copy(f, "config.yaml", self.run / "config.yaml", "the exact settings the run used", rq)
        self.copy(f, "tuning.yaml", self.run / "tuning.yaml", "hyperparameter search spaces", rq)
        self.copy(f, "run_info.json", self.run / "run_info.json",
                  "command, git commit, Python + package versions, raw-data fingerprint, stage timings", rq)
        self.copy(f, "split_manifest.json", self.manifests / "split_manifest.json",
                  "split dates, rows, features kept / pruned", rq)
        self.copy(f, "pipeline.log", self.run / "pipeline.log", "full console output of the run", rq)

    def build_data(self) -> None:
        f = "1_data"
        manifest = self.split_manifest()
        excluded = {}
        for grain in GRAINS:
            path = self.manifests / f"excluded_stations_log_{grain}.csv"
            if path.exists():
                for row in read_csv(path, dtype={"station_id": str}).itertuples():
                    excluded[row.station_id] = row.reason
        rows = []
        for key, cfg in STATIONS.items():
            sid, road = key.split(" - ", 1)
            row = {"station_id": sid, "station": station_label(sid), "road": road, "suburb": cfg["suburb"],
                   "road_type": _classify_road_type(key), "no2_monitor": cfg["aq_site"],
                   "monitor_distance_km": cfg["aq_distance_km"],
                   "weather_source": f"{cfg['weather'][0]}: {cfg['weather'][1]}",
                   "rain_from": cfg.get("rain_from", ""), "excluded_reason": excluded.get(sid, "")}
            for grain in GRAINS:
                for split in ("train", "val", "test"):
                    counts = manifest.get(grain, {}).get(split, {}).get("rows_per_station", {})
                    row[f"{grain}_{split}_rows"] = counts.get(sid, 0)
            row["role"] = ("excluded" if row["excluded_reason"] else
                           "train + test" if row["daily_test_rows"] or row["hourly_test_rows"] else
                           "train only (record ends before the test period)")
            rows.append(row)
        self.write(f, "stations.csv", pd.DataFrame(rows),
                   "every candidate station: road, road type, NO2 monitor + distance, weather source, "
                   "rows per split, train-only / tested / excluded (report table T1)",
                   "methodology", "scripts/stations.py + manifests/split_manifest.json + excluded_stations_log")
        self.stack(f, "cleaning_steps.csv",
                   [({"grain": g}, self.manifests / f"data_quality_{g}.csv") for g in GRAINS],
                   "rows and stations removed at each cleaning step (report table T2)", "methodology")
        self.stack(f, "missing_data.csv",
                   [({"grain": g}, self.manifests / f"missing_data_{g}.csv") for g in GRAINS],
                   "% missing per variable before / after cleaning, and how each gap is filled", "methodology")
        feature_rows = []
        for grain, info in manifest.items():
            pruned = info.get("excluded_features", {})
            history = set(info.get("autoregressive_features", []))
            frame = pd.DataFrame({"feature": info.get("all_candidate_features", [])})
            frame["family"] = [("NO2 history" if f in history else family_of(f)) for f in frame["feature"]]
            frame["kept"] = ~frame["feature"].isin(pruned)
            for family, group in frame.groupby("family"):
                feature_rows.append({"grain": grain, "family": family, "candidates": len(group),
                                     "pruned_near_duplicates": int((~group["kept"]).sum()),
                                     "kept": int(group["kept"].sum()),
                                     "kept_features": "; ".join(group.loc[group["kept"], "feature"])})
        self.write(f, "features_by_family.csv", pd.DataFrame(feature_rows),
                   "engineered features per family, kept vs pruned as near-duplicates, with names",
                   "methodology, RQ2", "manifests/split_manifest.json")
        split_rows = []
        for grain, info in manifest.items():
            for split in ("train", "val", "test"):
                part = info.get(split, {})
                split_rows.append({"grain": grain, "split": split, "date_from": part.get("date_min"),
                                   "date_to": part.get("date_max"), "rows": part.get("rows"),
                                   "stations": part.get("stations"), "no2_mean_pphm": part.get("no2_mean"),
                                   "no2_sd_pphm": part.get("no2_sd"),
                                   "no2_sd_within_station": part.get("no2_sd_within_station")})
        self.write(f, "split_summary.csv", pd.DataFrame(split_rows),
                   "chronological split: dates, rows, stations, NO2 level + spread per split "
                   "(lower test spread = lower R2 for the same error)", "methodology",
                   "manifests/split_manifest.json")
        self.stack(f, "excluded_stations.csv",
                   [({"grain": g}, self.manifests / f"excluded_stations_log_{g}.csv") for g in GRAINS],
                   "stations dropped and why", "methodology")
        self.copy(f, "aq_site_verification.csv", self.manifests / "aq_site_verification.csv",
                  "each station's NO2 monitor vs the truly nearest monitor (distance check)", "methodology")

    def build_models(self) -> None:
        f, rq = "2_models", "RQ3"
        ev = self.latest / "evaluation"
        self.copy(f, "scores_all.csv", ev / "all_results.csv",
                  "every model x feature set x split: RMSE, MAE, R2 (default and tuned)", rq)

        def flatten_summary(frame, labels):
            path = ev / f"summary_{labels['grain']}.csv"
            wide = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
            wide.columns = [f"{a} | {b}" for a, b in wide.columns]
            return wide.reset_index()

        self.stack(f, "summary.csv", [({"grain": g}, ev / f"summary_{g}.csv") for g in GRAINS],
                   "val / test R2 and per-station median test R2 for every method (report table T3)", rq,
                   transform=flatten_summary)
        self.stack(f, "bootstrap_test.csv", [({"grain": g}, ev / f"bootstrap_{g}_test.csv") for g in GRAINS],
                   "test R2 / RMSE / MAE with 95% confidence intervals (block bootstrap)", rq)
        self.stack(f, "bootstrap_vs_best.csv", [({"grain": g}, ev / f"bootstrap_{g}_test_vs_best.csv")
                                                for g in GRAINS],
                   "each model vs the best: tied, or worse (95% CI of the R2 difference)", rq)
        self.stack(f, "tuning_gains.csv", [({"grain": g}, ev / f"stage_delta_{g}.csv") for g in GRAINS],
                   "default -> tuned change in RMSE and R2 (report table T4)", rq)
        self.write(f, "model_times.csv", self.model_times(),
                   "seconds to train each model once, and minutes for its whole tuning search "
                   "(models shared the CPU -- compare orders of magnitude)", rq,
                   "scores.json + outputs/latest/tuning/*_tuned_metrics.csv")
        per_station = self.latest / "per_station"
        self.stack(f, "per_station_test.csv",
                   [({"model": m, "grain": g, "feature_set": fs},
                     per_station / f"{m}_tuned_{g}_{fs}_test.csv")
                    for m in TOP_MODELS for g in GRAINS for fs in FEATURE_SETS],
                   "test RMSE / R2 per station for the top models (GRU, LSTM, XGBoost; R2 blank under 30 rows)",
                   rq)
        curves = self.latest / "tuning" / "learning_curves"
        self.stack(f, "learning_curve_daily.csv",
                   [({}, curves / f"xgboost_daily_{fs}.csv") for fs in FEATURE_SETS],
                   "daily XGBoost: accuracy as training data grows (still rising = limited by data)",
                   "RQ2, RQ3")
        self.copy(f, "summary_daily.png", ev / "summary_daily.png", "val / test R2 by method, daily", rq)
        self.copy(f, "summary_hourly.png", ev / "summary_hourly.png", "val / test R2 by method, hourly", rq)
        self.copy(f, "bootstrap_hourly_test.png", ev / "bootstrap_hourly_test.png",
                  "hourly test R2 with 95% intervals (report figure F2)", rq)
        self.copy(f, "learning_curve_daily.png", curves / "xgboost_daily_all.png",
                  "daily XGBoost learning curve (appendix)", "RQ2, RQ3")

    def model_times(self) -> pd.DataFrame:
        scores_path = self.run / "scores.json"
        scores = pd.DataFrame(json.loads(scores_path.read_text())) if scores_path.exists() else pd.DataFrame()
        rows = []
        if not scores.empty and "train_seconds" in scores:
            fits = scores[scores["split"] == "val"].pivot_table(
                index=["model", "grain", "feature_set"], columns="stage", values="train_seconds")
            fits.columns = [f"train_seconds_{c}" for c in fits.columns]
            rows.append(fits)
        tuning = []
        for path in (self.latest / "tuning").glob("*_tuned_metrics.csv"):
            t = pd.read_csv(path)
            if {"n_trials", "tuning_seconds"} <= set(t.columns):
                tuning.append(t[["model", "grain", "feature_set", "n_trials", "tuning_seconds"]])
        if tuning:
            t = pd.concat(tuning).set_index(["model", "grain", "feature_set"])
            t["tuning_minutes"] = (t.pop("tuning_seconds") / 60).round(2)
            rows.append(t)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, axis=1).reset_index().round(2)

    def build_drivers(self) -> None:
        f = "3_drivers"
        shap, exp = self.latest / "shap", self.experiments
        self.stack(f, "shap_groups.csv",
                   [({"grain": g, "feature_set": fs}, shap / f"groups_{g}_{fs}.csv")
                    for g in GRAINS for fs in FEATURE_SETS],
                   "SHAP importance share per feature family -- weather, traffic, calendar, ... (report table T5)",
                   "RQ1")
        self.stack(f, "shap_importance.csv",
                   [({"grain": g, "feature_set": "exogenous"}, shap / f"importance_{g}_exogenous.csv")
                    for g in GRAINS],
                   "every feature ranked by SHAP, with direction (traffic + weather model)", "RQ1")
        self.stack(f, "traffic_parts.csv",
                   [({"grain": g}, shap / f"traffic_parts_{g}_exogenous.csv") for g in GRAINS],
                   "which parts of traffic matter: total, heavy vehicles, direction, traffic/wind, ...", "RQ1")
        self.stack(f, "traffic_wind.csv",
                   [({"grain": g}, shap / f"interaction_traffic_wind_{g}_exogenous.csv") for g in GRAINS],
                   "traffic's effect on NO2 at low / medium / high wind, by traffic decile", "RQ1")
        self.stack(f, "ablation.csv", [({"grain": g}, exp / "ablation" / f"ablation_{g}.csv") for g in GRAINS],
                   "accuracy when each feature family is removed; also sample weights / pruning on-off",
                   "RQ1, RQ2")
        for kind, what in (("effect", "weekday-holiday effect on traffic and NO2, weather-adjusted, 95% CI"),
                           ("by_station", "holiday vs normal weekday averages per station"),
                           ("dose_response", "correlation of traffic drop vs NO2 drop across stations")):
            self.stack(f, f"holiday_{kind}.csv",
                       [({"grain": g}, exp / "holiday" / f"holiday_{g}_{kind}.csv") for g in GRAINS],
                       what, "RQ1")
        self.copy(f, "shap_beeswarm_hourly.png", shap / "summary_hourly_exogenous.png",
                  "SHAP summary, hourly traffic + weather model (report figure F3)", "RQ1, RQ4")
        self.copy(f, "traffic_wind_hourly.png", shap / "interaction_traffic_wind_hourly_exogenous.png",
                  "traffic effect by wind speed -- calm air amplifies traffic NO2 (report figure F4)", "RQ1")
        self.copy_first(f, "wind_effect_hourly.png",
                        sorted(shap.glob("dependence_hourly_exogenous_wind_speed*.png")),
                        "SHAP effect of wind speed (real units, m/s)", "RQ1")
        self.copy(f, "holiday_profile_hourly.png", exp / "holiday" / "holiday_hourly_profile.png",
                  "traffic and NO2 by hour, holidays vs normal weekdays (report figure F5)", "RQ1")
        self.copy(f, "holiday_dose_response_hourly.png", exp / "holiday" / "holiday_hourly_dose_response.png",
                  "per-station traffic drop vs NO2 drop (appendix)", "RQ1")
        self.copy(f, "road_type_hourly.png", shap / "groups_by_road_type_hourly_exogenous.png",
                  "SHAP family shares by road type, train rows (appendix)", "RQ1, RQ3")
        self.copy(f, "ablation_hourly.png", exp / "ablation" / "ablation_hourly.png",
                  "feature-family ablation, hourly (appendix)", "RQ1, RQ2")

    def build_explainability(self) -> None:
        f, rq = "4_explainability", "RQ4"
        perm, shap = self.latest / "permutation", self.latest / "shap"
        self.stack(f, "model_agreement.csv",
                   [({"grain": g, "feature_set": fs}, perm / f"top3_agreement_{g}_{fs}.csv")
                    for g in GRAINS for fs in FEATURE_SETS],
                   "rank agreement (Spearman) of feature families between models and methods", rq)
        self.stack(f, "permutation_families.csv",
                   [({"grain": g, "feature_set": fs}, perm / f"top3_families_{g}_{fs}.csv")
                    for g in GRAINS for fs in FEATURE_SETS],
                   "permutation importance (R2 drop) per family for the top-3 models", rq)
        self.stack(f, "waterfall_cases.csv",
                   [({"grain": g}, shap / f"waterfall_{g}_exogenous.csv") for g in GRAINS],
                   "single predictions explained: highest NO2, typical, calm vs windy rush hour", rq)
        self.copy(f, "permutation_top3_hourly.png", perm / "top3_hourly_exogenous.png",
                  "top-3 models' reliance per feature family, hourly (report figure F6)", rq)
        self.copy_first(f, "history_window_hourly.png",
                        [perm / "top3_window_hourly_exogenous_gru.png", perm / "top3_window_hourly_exogenous_lstm.png"],
                        "how far back the sequence model looks (current + last 1-2 hours)", rq)
        self.copy(f, "waterfall_calm_rush_hour.png", shap / "waterfall_hourly_exogenous_calm_rush_hour.png",
                  "one calm rush hour explained feature by feature (report figure F7)", rq)
        self.copy(f, "waterfall_highest_no2.png", shap / "waterfall_hourly_exogenous_highest_NO2.png",
                  "the highest-NO2 hour explained -- shows the failure on extremes", "RQ3, RQ4")

    def build_errors(self) -> None:
        f, rq = "5_errors", "RQ3"
        err, loso = self.experiments / "error_analysis", self.experiments / "loso"
        self.stack(f, "errors_by_hour.csv", [({"grain": "hourly"}, err / "errors_hourly_test_all_hour.csv")],
                   "test RMSE / MAE / bias by hour of day for every model", rq)

        def as_condition(frame, labels):
            frame = frame.rename(columns={labels["breakdown"]: "group"})
            frame.insert(0, "breakdown", labels["breakdown"])
            return frame

        parts = [({"grain": g, "breakdown": b}, err / f"errors_{g}_test_all_{b}.csv")
                 for g in GRAINS for b in ("no2_level", "traffic_level", "road_type")]
        self.stack(f, "errors_by_condition.csv", parts,
                   "test errors by NO2 level, traffic level and road type, every model", rq,
                   transform=as_condition)
        self.stack(f, "unseen_stations.csv",
                   [({"grain": g}, loso / f"loso_{g}_exogenous.csv") for g in GRAINS],
                   "leave-one-station-out: each station predicted by a model that never saw it", rq)
        self.stack(f, "unseen_stations_summary.csv",
                   [({"grain": g}, loso / f"loso_{g}_exogenous_summary.csv") for g in GRAINS],
                   "leave-one-station-out summary: median / worst station R2, stations beating the average", rq)
        self.copy(f, "errors_by_hour_hourly.png", err / "errors_hourly_test_all_hour.png",
                  "error by hour of day (report figure F8, optional)", rq)
        self.copy(f, "errors_by_no2_level_hourly.png", err / "errors_hourly_test_all_no2_level.png",
                  "error by NO2 level -- extremes under-predicted (appendix)", rq)
        self.copy(f, "unseen_stations_hourly.png", loso / "loso_hourly_exogenous.png",
                  "unseen-station R2 per station (appendix)", rq)

    # ----- README ----------------------------------------------------------
    def write_readme(self) -> None:
        info_path = self.run / "run_info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        git = info.get("git") or {}
        titles = {"run": "Replicate the run", "1_data": "Data and preprocessing (Methodology)",
                  "2_models": "Model comparison (Results, RQ3)",
                  "3_drivers": "What drives NO2 (Results, RQ1 + RQ2)",
                  "4_explainability": "Do the explanations agree? (Results, RQ4)",
                  "5_errors": "When and where the models fail (Results / Discussion, RQ3)"}
        lines = [
            "# Final results",
            "",
            f"Built from run **{self.run.name}** by `src/11_final_results.py` -- "
            "rebuilt automatically after every complete run, so don't edit files here by hand.",
            "",
            f"- finished: {info.get('finished', '?')}  |  status: {info.get('status', '?')}",
            f"- command: `{info.get('command', '?')}`",
            f"- git commit: `{git.get('commit', '?')}` on `{git.get('branch', '?')}`"
            + ("  -- **uncommitted changes at run time**" if git.get("uncommitted_changes") else ""),
            f"- full backup of every output (~750 files): `working/runs/{self.run.name}/` "
            "(emptied at the start of the next full run -- zip it to keep it)",
            "",
            "To replicate: check out the commit above, then "
            "`python run_pipeline.py --config results/run/config.yaml --experiments`.",
            "",
            "Feature sets: `exogenous` = traffic, weather, calendar and station only (answers *what drives "
            "NO2*); `all` = the same plus NO2's own recent history (the best *forecast*).",
            "",
        ]
        for folder in titles:
            entries = [e for e in self.readme if e[0] == folder]
            if not entries:
                continue
            lines += [f"## `{folder}/` -- {titles[folder]}", "", "| file | what it shows | RQ | source in the run |",
                      "|---|---|---|---|"]
            lines += [f"| `{name}` | {what} | {rq} | `{source}` |" for _, name, what, rq, source in entries]
            lines.append("")
        if self.missing:
            lines += ["## Not built (source missing in this run)", ""] + [f"- {m}" for m in self.missing] + [""]
        (self.out / "README.md").write_text("\n".join(lines), encoding="utf-8")

    def build(self) -> None:
        # The output folder is wiped and rebuilt, so refuse if it holds anything
        # this script didn't make (e.g. an old results/runs/ backup).
        if self.out.exists():
            unexpected = [p.name for p in self.out.iterdir() if p.name not in OWN_ENTRIES]
            if unexpected:
                raise SystemExit(
                    f"{self.out} contains things this script didn't create: {unexpected}.\n"
                    "It is wiped and rebuilt, so move those somewhere else first "
                    "(e.g. an old run backup -> working/runs/).")
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)
        self.build_run()
        self.build_data()
        self.build_models()
        self.build_drivers()
        self.build_explainability()
        self.build_errors()
        self.write_readme()


def newest_full_run() -> Path:
    runs = []
    for info_path in paths.RUNS_DIR.glob("*/run_info.json"):
        info = json.loads(info_path.read_text())
        stages = {s["stage"] for s in info.get("stages", [])}
        if info.get("status") == "ok" and {"train", "tune", "evaluate", "explain", "experiments"} <= stages:
            runs.append((info.get("finished", ""), info_path.parent))
    if not runs:
        raise SystemExit(f"No successful complete run found in {paths.RUNS_DIR} -- pass --run <folder>.")
    return max(runs)[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build results/ (the curated report files) from a saved run.")
    parser.add_argument("--run", type=Path, default=None,
                        help="saved run folder (default: newest complete run in working/runs/)")
    parser.add_argument("--out", type=Path, default=paths.RESULTS_DIR,
                        help="where to write (default: results/)")
    args = parser.parse_args()
    run_dir = (args.run or newest_full_run()).resolve()
    if not (run_dir / "outputs").exists():
        raise SystemExit(f"{run_dir} doesn't look like a saved run (no outputs/ folder).")

    builder = Builder(run_dir, args.out.resolve())
    builder.build()
    count = sum(1 for p in builder.out.rglob("*") if p.is_file())
    print(f"Final results: {count} files from {run_dir.name} -> {builder.out}")
    for folder in ("run", "1_data", "2_models", "3_drivers", "4_explainability", "5_errors"):
        n = sum(1 for p in (builder.out / folder).glob("*") if p.is_file()) if (builder.out / folder).exists() else 0
        print(f"  {folder + '/':20s} {n} files")
    if builder.missing:
        print(f"  {len(builder.missing)} file(s) not built -- their source isn't in this run (listed in README.md):")
        for m in builder.missing:
            print(f"    {m}")


if __name__ == "__main__":
    main()