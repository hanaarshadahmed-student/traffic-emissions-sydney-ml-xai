"""
Pipeline runner: the one command that runs the whole project (build data, train, tune, evaluate, explain with SHAP), as set in config/config.yaml. Options only run less of it.

Usage:
    python run_pipeline.py                      # everything
    python run_pipeline.py --no-tune            # skip tuning
    python run_pipeline.py --experiments        # everything + the extra experiments
    python run_pipeline.py --stage evaluate     # just one stage
    python run_pipeline.py --grain daily --models ridge xgboost
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from datetime import datetime

import yaml

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from scripts import paths  # noqa: E402
from scripts import run_record  # noqa: E402

# stages whose outputs are worth keeping as a run record
RECORDED_STAGES = {"train", "tune", "evaluate", "explain", "experiments"}

# (stage name, scripts (relative to the repo root), label, required inputs relative
# to the repo root -- {grain} is filled in from training.grain,
# whether the scripts accept --config <config.yaml>)
STAGES = [
    ("ingest", ["src/01_data_ingestion.py"], "Data ingestion", [], False),
    ("clean", ["src/02_cleaning.py"], "Data cleaning", [
        "data/processed/final_combined_dataset_daily.csv",
        "data/processed/final_combined_dataset_hourly.csv",
    ], True),
    ("preprocess", ["src/03_data_preprocessing.py"], "Preprocessing (impute/encode)", [
        "data/processed/preprocessed_daily.csv",
        "data/processed/preprocessed_hourly.csv",
    ], False),
    ("features", ["src/04_feature_engineering.py"], "Feature engineering", [
        "data/processed/final_daily.csv",
        "data/processed/final_hourly.csv",
    ], False),
    ("split", ["src/05_train_test_split.py"], "Train/val/test split + scaling", [
        "data/processed/features_daily.csv",
        "data/processed/features_hourly.csv",
        "data/processed/feature_manifest.json",
    ], False),
    ("train", ["src/06_run_models.py"], "Model training + scoring", [
        "data/processed/splits/{grain}_train.csv",
        "data/processed/splits/{grain}_val.csv",
    ], True),
    ("tune", ["models/tuning/tune_ridge.py", "models/tuning/tune_decision_tree.py", "models/tuning/tune_random_forest.py", "models/tuning/tune_lstm.py", "models/tuning/tune_gru.py", "models/tuning/tune_xgboost.py", "models/tuning/tune_svr.py"], "Hyperparameter tuning", [
        "data/processed/splits/{grain}_train.csv",
        "data/processed/splits/{grain}_val.csv",
    ], False),
    ("evaluate", ["src/07_evaluation.py", "src/08_bootstrap_ci.py"],
     "Evaluation tables + confidence intervals", [
        "working/latest/results.json",
        "working/latest/predictions",
    ], False),
    ("explain", ["src/09_shap_explainability.py", "src/10_permutation_importance.py"],
     "Explainability (SHAP + permutation)", [
        "data/processed/splits/{grain}_train.csv",
        "data/processed/splits/{grain}_test.csv",
        "data/processed/feature_manifest.json",
        "working/latest/evaluation/all_results.csv",
    ], True),
    ("experiments", ["experiments/01_error_analysis.py", "experiments/02_leave_one_station_out.py",
                     "experiments/03_ablation.py", "experiments/04_holiday_check.py"],
     "Experiments (errors, unseen stations, ablation, holidays)", [
        "data/processed/splits/{grain}_test.csv",
        "working/latest/predictions",
    ], True),
]
STAGE_NAMES = [s[0] for s in STAGES]


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def resolve_stage(config: dict, cli_stage: str | None) -> str:
    stage = cli_stage or config.get("pipeline", {}).get("stage", "all")
    stage = str(stage).strip().lower()
    if stage not in ("all", *STAGE_NAMES):
        raise SystemExit(
            f"Unknown pipeline.stage {stage!r}. "
            f"Expected 'all' or one of {STAGE_NAMES}."
        )
    return stage


def wipe_processed_dir() -> None:
    """Empty data/processed/ before a full rebuild. Everything in there is
    an output of stages 01-05, so it can always be regenerated."""
    processed = paths.PROCESSED_DIR
    processed.mkdir(parents=True, exist_ok=True)
    removed = []
    for item in processed.iterdir():
        removed.append(item.name)
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    if removed:
        say(f"[wipe] emptied data/processed/ before full rebuild ({len(removed)} items)")
    else:
        say("[wipe] data/processed/ already empty")


def results_source_run() -> str | None:
    """Run id that results/ was built from (results/run/run_info.json)."""
    info = paths.RESULTS_DIR / "run" / "run_info.json"
    try:
        return json.loads(info.read_text()).get("run_id")
    except (OSError, ValueError):
        return None


def wipe_working(keep_run: str) -> None:
    removed = []
    for folder in (paths.LATEST_DIR, paths.EXPERIMENTS_RESULTS_DIR, paths.SAVED_MODELS_DIR):
        if folder.exists() and any(folder.iterdir()):
            shutil.rmtree(folder)
            removed.append(folder.relative_to(ROOT_DIR).as_posix())
    if not paths.SENSITIVITY and paths.RUNS_DIR.exists():
        keep = {keep_run, results_source_run()}  # never the backup behind results/
        old_runs = [p for p in paths.RUNS_DIR.iterdir() if p.name not in keep]
        for p in old_runs:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        if old_runs:
            removed.append(f"{paths.RUNS_DIR.relative_to(ROOT_DIR).as_posix()} ({len(old_runs)} old items)")
    say(f"[wipe] emptied {', '.join(removed)}" if removed else "[wipe] working/ already empty")


def build_final_results(run_dir: Path) -> None:
    script = ROOT_DIR / "src" / "11_final_results.py"
    result = subprocess.run([sys.executable, str(script), "--run", str(run_dir)],
                            cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(result.stdout.rstrip())
    if result.returncode:
        print(f"[final] WARNING: results/ was not rebuilt:\n{result.stderr.strip()[-2000:]}\n"
              f"  re-run with: python src/11_final_results.py --run {run_dir.relative_to(ROOT_DIR)}")


def resolve(rel: str) -> Path:
    for prefix, folder in (("data/processed/", paths.PROCESSED_DIR), ("working/latest/", paths.LATEST_DIR)):
        if rel.startswith(prefix):
            return folder / rel[len(prefix):]
    return ROOT_DIR / rel


def check_inputs(required: list[str], grains: list[str]) -> list[str]:
    needed = dict.fromkeys(rel.format(grain=g) for rel in required for g in grains)
    return [rel for rel in needed if not resolve(rel).exists()]


def deep_merge(base: dict, changes: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (changes or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_sensitivity(config: dict, name: str) -> tuple[dict, str]:
    runs = load_config(paths.SENSITIVITY_CONFIG_PATH)
    if name not in runs:
        raise SystemExit(f"--sensitivity: {name!r} isn't in {paths.SENSITIVITY_CONFIG_PATH.name}. "
                         f"Known: {sorted(runs)}")
    config = deep_merge(config, runs[name].get("changes") or {})
    config.setdefault("pipeline", {})["run_tuning"] = False
    return config, runs[name].get("description", "")


def apply_overrides(config: dict, args) -> tuple[dict, list[str]]:
    config = copy.deepcopy(config)
    training = config.setdefault("training", {})
    pipeline = config.setdefault("pipeline", {})
    notes = []
    if args.grain:
        training["grain"] = list(dict.fromkeys(args.grain))
        notes.append("--grain " + " ".join(args.grain))
    if args.models:
        models = training.get("models") or {}
        unknown = sorted(set(args.models) - set(models))
        if unknown:
            raise SystemExit(f"--models: unknown {unknown}. Known: {sorted(models)}")
        for name, model in models.items():
            (model or {})["enabled"] = name in args.models
        notes.append("--models " + " ".join(args.models))
    if args.no_tune:
        pipeline["run_tuning"] = False
        notes.append("--no-tune")
    if args.experiments:
        pipeline["run_experiments"] = True
        notes.append("--experiments")
    if args.jobs is not None:
        pipeline["jobs"] = args.jobs
        notes.append(f"--jobs {args.jobs}")
    grain = training.get("grain", "daily")
    training["grain"] = list(grain) if isinstance(grain, (list, tuple)) else [grain]
    return config, notes


def resolve_jobs(config: dict) -> int:
    """pipeline.jobs: a number, or "auto" = one parallel job per 4 CPU cores
    (max 4, min 1)."""
    jobs = (config.get("pipeline") or {}).get("jobs", "auto")
    if str(jobs).lower() == "auto":
        return max(1, min(4, (os.cpu_count() or 1) // 4))
    return max(1, int(jobs))


LOG: list[str] = []  # everything printed this run, saved as pipeline.log


def say(text: str = "") -> None:
    print(text, flush=True)
    LOG.append(text + "\n")


class StageFailed(Exception):
    def __init__(self, code: int):
        self.code = code


def _stop_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_script(name: str, script: str, label: str, extra_args: list[str], env: dict) -> float:
    cmd = [sys.executable, "-u", str(ROOT_DIR / script), *extra_args]
    say(f"\n{'='*70}\n[{name}] {label}  ->  {script} {' '.join(extra_args)}\n{'='*70}")
    start = time.perf_counter()

    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})
    proc = subprocess.Popen(cmd, cwd=ROOT_DIR, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", **group)

    def pump() -> None:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            LOG.append(line)
    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        while proc.poll() is None:
            time.sleep(0.2)
    except KeyboardInterrupt:
        say(f"\n[{name}] Ctrl+C -- stopping {script} ...")
        _stop_process_tree(proc)
        reader.join(timeout=2)
        say(f"[{name}] stopped.")
        raise
    reader.join(timeout=5)

    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        say(f"\n[{name}] FAILED after {elapsed:.1f}s (exit code {proc.returncode})")
        raise StageFailed(proc.returncode)
    say(f"\n[{name}] done in {elapsed:.1f}s")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pipeline. With no options it runs everything in config/config.yaml.")
    parser.add_argument("--config", type=Path, default=paths.CONFIG_PATH)
    parser.add_argument("--stage", choices=["all", *STAGE_NAMES], default=None,
                        help="Run only this stage (default: pipeline.stage in the config, normally all).")
    parser.add_argument("--grain", nargs="+", choices=["daily", "hourly"],
                        help="Only these grains (default: training.grain in the config).")
    parser.add_argument("--models", nargs="+", metavar="MODEL",
                        help="Only these ML models, e.g. --models ridge lstm (baselines always run).")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip the (slow) tune stage when running all stages.")
    parser.add_argument("--experiments", action="store_true",
                        help="Also run the experiments (error analysis, leave-one-station-out, "
                             "ablation, holiday check) after the main pipeline.")
    parser.add_argument("--sensitivity", metavar="NAME",
                        help="Sensitivity run from config/sensitivity.yaml, e.g. no2_clipped. "
                             "Uses its own data/results folders; tuning off.")
    parser.add_argument("--jobs", type=int, default=None,
                        help="How many models to train at the same time (default: pipeline.jobs in the "
                             "config, normally auto = one per 4 CPU cores, max 4). --jobs 1 = one at a time.")
    parser.add_argument("--name", help="Short label added to the saved run's folder name.")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save a run record to working/runs/.")
    args = parser.parse_args()
    started = datetime.now()

    base_config = load_config(args.config)
    if args.sensitivity:
        os.environ[paths.SENSITIVITY_ENV] = args.sensitivity
        importlib.reload(paths)
        base_config, description = apply_sensitivity(base_config, args.sensitivity)
        args.name = args.name or f"sens_{args.sensitivity}"
    config, overrides = apply_overrides(base_config, args)
    if args.sensitivity:
        overrides.insert(0, f"--sensitivity {args.sensitivity}")
    stage = resolve_stage(config, args.stage)
    grains = config["training"]["grain"]
    run_tuning = bool(config["pipeline"].get("run_tuning", True))
    run_experiments = bool(config["pipeline"].get("run_experiments", False))

    if stage == "all":
        run_list = [s for s in STAGES
                    if (s[0] != "tune" or run_tuning) and (s[0] != "experiments" or run_experiments)]
    else:
        run_list = [s for s in STAGES if s[0] == stage]
    save = (bool(config["pipeline"].get("save_run", True)) and not args.no_save
            and any(s[0] in RECORDED_STAGES for s in run_list))

    run_id = run_record.make_run_id(grains, args.name)
    effective_path = (run_record.RUNS_DIR / run_id / "config.yaml" if save
                      else Path(tempfile.gettempdir()) / f"pipeline_config_{run_id}.yaml")
    run_record.write_effective_config(config, effective_path, overrides)
    env = {**os.environ, run_record.RUN_ID_ENV: run_id, "PYTHONIOENCODING": "utf-8"}

    if stage == "all":
        say(f"Running the FULL pipeline: grains {grains}, "
            f"tuning {'ON' if run_tuning else 'OFF'}, experiments {'ON' if run_experiments else 'OFF'}"
            + (f", overrides: {' '.join(overrides)}" if overrides else ""))
        wipe_processed_dir()
        wipe_working(run_id)
    else:
        say(f"Running SINGLE stage: {stage!r}, grains {grains}"
            + (f", overrides: {' '.join(overrides)}" if overrides else ""))
    if args.sensitivity:
        say(f"SENSITIVITY run {args.sensitivity!r}: {description}\n"
            f"  data    -> {paths.PROCESSED_DIR.relative_to(ROOT_DIR)}\n"
            f"  results -> {paths.LATEST_DIR.relative_to(ROOT_DIR)}  (main results untouched)")
    say(f"Training {resolve_jobs(config)} model(s) at a time (pipeline.jobs / --jobs)")
    if save:
        say(f"Run id: {run_id}")

    config_arg = ["--config", str(effective_path)]
    stage_args = {
        "clean": config_arg,
        "train": [*config_arg, "--jobs", str(resolve_jobs(config))],
        "tune": ["--main-config", str(effective_path)],
        "explain": config_arg,
        "experiments": [*config_arg, "--grain", *grains],
    }
    script_args = {  # scripts that need different arguments from the rest of their stage
        "src/07_evaluation.py": ["--grain", grains[0]] if len(grains) == 1 else [],
        "src/08_bootstrap_ci.py": ["--grain", *grains],
        "src/10_permutation_importance.py": [*config_arg, "--grain", *grains],
    }

    timings: list[tuple[str, str, float]] = []
    status, exit_code = "ok", 0
    try:
        for name, scripts, label, required_inputs, _ in run_list:
            if stage != "all":  # "all" produces its own inputs stage-to-stage
                missing = check_inputs(required_inputs, grains)
                if missing:
                    idx = STAGE_NAMES.index(name)
                    prev = STAGE_NAMES[idx - 1] if idx > 0 else None
                    hint = f" Run stage {prev!r} first (or the full pipeline)." if prev else ""
                    if save:
                        shutil.rmtree(effective_path.parent, ignore_errors=True)
                    else:
                        effective_path.unlink(missing_ok=True)
                    raise SystemExit(f"[{name}] can't run -- missing {missing}.{hint}")
            try:
                elapsed = sum(run_script(name, script, label,
                                         script_args.get(script, stage_args.get(name, [])), env)
                              for script in scripts)
            except StageFailed as failure:
                timings.append((name, label, 0.0))
                status, exit_code = f"failed at {name}", failure.code
                break
            timings.append((name, label, elapsed))
    except KeyboardInterrupt:
        status, exit_code = "interrupted", 130

    total = sum(t for _, _, t in timings)
    say(f"\n{'='*70}\nPipeline summary\n{'='*70}")
    for name, label, elapsed in timings:
        say(f"  [{'OK' if elapsed or status == 'ok' else '!!'}] {name:12s} {label:35s} {elapsed:6.1f}s")
    say(f"  {'-'*58}")
    say(f"  Total: {total / 60:.1f} min   status: {status}")

    if save and timings:
        run_dir = run_record.save_run(run_id, effective_path, timings, status, started,
                                      [Path(sys.executable).name, *sys.argv], "".join(LOG))
        print(f"\nRun saved -> {run_dir.relative_to(ROOT_DIR)}")
        ran = {name for name, _, _ in timings}
        if stage == "all" and status == "ok" and not args.sensitivity:
            if {"tune", "experiments"} <= ran:
                build_final_results(run_dir)
            else:  # a quick/partial run must never replace the report's results
                print("[final] results/ left unchanged -- it's only rebuilt after a complete run "
                      "(with tuning and --experiments). To build it from this run anyway:\n"
                      f"  python src/11_final_results.py --run {run_dir.relative_to(ROOT_DIR)}")
    elif not save:
        effective_path.unlink(missing_ok=True)

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()