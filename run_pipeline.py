"""
CO2/NO2 Traffic-Emissions Capstone -- Pipeline runner

The one command for the whole project:

    python run_pipeline.py

runs EVERYTHING the report needs, as set in config/config.yaml: builds the
data, trains every enabled method on every grain (daily and hourly), tunes,
and makes the evaluation tables and charts. Options are only for running
LESS -- e.g. one stage, one grain, a few models, or no tuning.

Stages (pipeline order):
    ingest      src/01_data_ingestion.py
    clean       src/02_cleaning.py
    preprocess  src/03_data_preprocessing.py
    features    src/04_feature_engineering.py
    split       src/05_train_test_split.py
    train       src/06_run_models.py
    tune        scripts/tuning/tune_xgboost.py + scripts/tuning/tune_svr.py
    evaluate    src/07_evaluation.py

`--stage all` (the default) runs every stage and first empties
data/processed/ so a run never mixes with a stale one. results/ is never
wiped -- results.json is the cross-run log that train/tune upsert into.
Any other stage runs just that stage and reuses what's already on disk.

Every run that trains, tunes or evaluates is saved to results/runs/<run_id>/
with the exact (effective) config, data fingerprint, git commit and full
log (see scripts/run_record.py), and listed in results/runs/index.csv.

Usage:
    python run_pipeline.py                          # everything
    python run_pipeline.py --no-tune                # everything except tuning
    python run_pipeline.py --grain daily            # only daily data
    python run_pipeline.py --models ridge lstm      # only these ML models
    python run_pipeline.py --stage evaluate         # just one stage
    python run_pipeline.py --name my_label          # label the saved run
    python run_pipeline.py --no-save                # don't save a run record
    python run_pipeline.py --config config/other.yaml
Options combine, e.g.  --stage train --grain hourly --models lstm
"""

from __future__ import annotations

import argparse
import copy
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
RECORDED_STAGES = {"train", "tune", "evaluate"}

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
    ("tune", ["scripts/tuning/tune_xgboost.py", "scripts/tuning/tune_svr.py"], "Hyperparameter tuning", [
        "data/processed/splits/{grain}_train.csv",
        "data/processed/splits/{grain}_val.csv",
    ], False),
    ("evaluate", ["src/07_evaluation.py"], "Evaluation tables + plots", [
        "results/results.json",
    ], False),
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


def check_inputs(required: list[str], grains: list[str]) -> list[str]:
    needed = dict.fromkeys(rel.format(grain=g) for rel in required for g in grains)
    return [rel for rel in needed if not (ROOT_DIR / rel).exists()]


def apply_overrides(config: dict, args) -> tuple[dict, list[str]]:
    """config.yaml + command-line options -> the config this run uses."""
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
    grain = training.get("grain", "daily")
    training["grain"] = list(grain) if isinstance(grain, (list, tuple)) else [grain]
    return config, notes


LOG: list[str] = []  # everything printed this run, saved as pipeline.log


def say(text: str = "") -> None:
    print(text, flush=True)
    LOG.append(text + "\n")


class StageFailed(Exception):
    def __init__(self, code: int):
        self.code = code


def _stop_process_tree(proc: subprocess.Popen) -> None:
    """Kill a stage script AND anything it started (e.g. the worker
    processes random forest / XGBoost use with n_jobs=-1), so Ctrl+C
    never leaves the terminal hanging."""
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

    # The stage runs in its own process group, so Ctrl+C reaches only this
    # runner, which then stops the whole stage cleanly (see below).
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})
    proc = subprocess.Popen(cmd, cwd=ROOT_DIR, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", **group)

    # Stream the script's output live AND keep a copy for the run log. Done
    # in a background thread so the main thread is free to notice Ctrl+C
    # immediately, even while a model trains silently for minutes.
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
    parser.add_argument("--name", help="Short label added to the saved run's folder name.")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save a run record to results/runs/.")
    args = parser.parse_args()
    started = datetime.now()

    config, overrides = apply_overrides(load_config(args.config), args)
    stage = resolve_stage(config, args.stage)
    grains = config["training"]["grain"]
    run_tuning = bool(config["pipeline"].get("run_tuning", True))

    if stage == "all":
        run_list = [s for s in STAGES if s[0] != "tune" or run_tuning]
    else:
        run_list = [s for s in STAGES if s[0] == stage]
    save = (bool(config["pipeline"].get("save_run", True)) and not args.no_save
            and any(s[0] in RECORDED_STAGES for s in run_list))

    # Every stage reads the EFFECTIVE config (config.yaml + overrides), saved
    # with the run so it can be replayed exactly.
    run_id = run_record.make_run_id(grains, args.name)
    effective_path = (run_record.RUNS_DIR / run_id / "config.yaml" if save
                      else Path(tempfile.gettempdir()) / f"pipeline_config_{run_id}.yaml")
    run_record.write_effective_config(config, effective_path, overrides)
    env = {**os.environ, run_record.RUN_ID_ENV: run_id, "PYTHONIOENCODING": "utf-8"}

    if stage == "all":
        say(f"Running the FULL pipeline: grains {grains}, "
            f"tuning {'ON' if run_tuning else 'OFF'}"
            + (f", overrides: {' '.join(overrides)}" if overrides else ""))
        wipe_processed_dir()
    else:
        say(f"Running SINGLE stage: {stage!r}, grains {grains}"
            + (f", overrides: {' '.join(overrides)}" if overrides else ""))
    if save:
        say(f"Run id: {run_id}")

    stage_args = {
        "clean": ["--config", str(effective_path)],
        "train": ["--config", str(effective_path)],
        "tune": ["--main-config", str(effective_path)],
        "evaluate": ["--grain", grains[0]] if len(grains) == 1 else [],
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
                elapsed = sum(run_script(name, script, label, stage_args.get(name, []), env)
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
        print(f"\nRun saved -> {run_dir.relative_to(ROOT_DIR)}  (overview: results/runs/index.csv)")
    elif not save:
        effective_path.unlink(missing_ok=True)

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
