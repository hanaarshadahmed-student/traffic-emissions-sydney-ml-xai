"""
CO2/NO2 Traffic-Emissions Capstone -- Pipeline runner

The one command for the whole project. Runs the numbered scripts in
src/ in order, or just one stage, based on `pipeline.stage` in
config/config.yaml (or whatever --config points at). Streams each
script's own output live, times it, and prints a summary at the end.

Stages (pipeline order):
    ingest      src/01_data_ingestion.py
    clean       src/02_data_cleaning.py
    preprocess  src/03_data_preprocessing.py
    features    src/04_feature_engineering.py
    split       src/05_train_test_split.py
    train       src/06_train_models.py
    tune        src/tuning/tune_xgboost.py + src/tuning/tune_svr.py
    evaluate    src/07_evaluation.py

`stage: all` runs every stage (tune only if pipeline.run_tuning: true)
and first empties data/processed/ so a run never mixes this run's
outputs with a stale run's. results/ is never wiped -- results.json is
the cross-run log that train/tune upsert into.

Any other stage name runs just that one stage and reuses whatever is
already on disk; the runner checks the stage's inputs exist first and
tells you which earlier stage to run if something's missing.

Usage:
    python run_pipeline.py
    python run_pipeline.py --stage features        # override the yaml once
    python run_pipeline.py --config config/my_experiment.yaml
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import paths  # noqa: E402

# (stage name, scripts (relative to src/), label, required inputs relative
# to the repo root -- {grain} is filled in from training.grain,
# whether the scripts accept --config <config.yaml>)
STAGES = [
    ("ingest", ["01_data_ingestion.py"], "Data ingestion", [], False),
    ("clean", ["02_data_cleaning.py"], "Data cleaning", [
        "data/processed/final_combined_dataset_daily.csv",
        "data/processed/final_combined_dataset_hourly.csv",
    ], True),
    ("preprocess", ["03_data_preprocessing.py"], "Preprocessing (impute/encode)", [
        "data/processed/preprocessed_daily.csv",
        "data/processed/preprocessed_hourly.csv",
    ], False),
    ("features", ["04_feature_engineering.py"], "Feature engineering", [
        "data/processed/final_daily.csv",
        "data/processed/final_hourly.csv",
    ], False),
    ("split", ["05_train_test_split.py"], "Train/val/test split + scaling", [
        "data/processed/features_daily.csv",
        "data/processed/features_hourly.csv",
        "data/processed/feature_manifest.json",
    ], False),
    ("train", ["06_train_models.py"], "Model training + scoring", [
        "data/processed/splits/{grain}_train.csv",
        "data/processed/splits/{grain}_val.csv",
    ], True),
    ("tune", ["tuning/tune_xgboost.py", "tuning/tune_svr.py"], "Hyperparameter tuning", [
        "data/processed/splits/daily_train.csv",
        "data/processed/splits/hourly_train.csv",
    ], False),
    ("evaluate", ["07_evaluation.py"], "Evaluation tables + plots", [
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
        print(f"[wipe] emptied data/processed/ before full rebuild ({len(removed)} items)")
    else:
        print("[wipe] data/processed/ already empty")


def check_inputs(required: list[str], grain: str) -> list[str]:
    return [rel.format(grain=grain) for rel in required
            if not (ROOT_DIR / rel.format(grain=grain)).exists()]


def run_script(name: str, script: str, label: str, extra_args: list[str]) -> float:
    cmd = [sys.executable, str(SRC_DIR / script), *extra_args]
    print(f"\n{'='*70}\n[{name}] {label}  ->  src/{script} {' '.join(extra_args)}\n{'='*70}")
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        print(f"\n[{name}] FAILED after {elapsed:.1f}s (exit code {result.returncode})")
        raise SystemExit(result.returncode)
    print(f"\n[{name}] done in {elapsed:.1f}s")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=paths.CONFIG_PATH)
    parser.add_argument(
        "--stage", choices=["all", *STAGE_NAMES], default=None,
        help="Override pipeline.stage from the config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    stage = resolve_stage(config, args.stage)
    grain = config.get("training", {}).get("grain", "daily")
    run_tuning = bool(config.get("pipeline", {}).get("run_tuning", False))

    if stage == "all":
        run_list = [s for s in STAGES if s[0] != "tune" or run_tuning]
        print(f"Running the FULL pipeline, grain={grain!r}, "
              f"tuning {'ON' if run_tuning else 'OFF (pipeline.run_tuning: false)'}")
        wipe_processed_dir()
    else:
        run_list = [s for s in STAGES if s[0] == stage]
        print(f"Running SINGLE stage: {stage!r}")

    timings: list[tuple[str, str, float]] = []
    for name, scripts, label, required_inputs, takes_config in run_list:
        if stage != "all":  # "all" produces its own inputs stage-to-stage
            missing = check_inputs(required_inputs, grain)
            if missing:
                idx = STAGE_NAMES.index(name)
                prev = STAGE_NAMES[idx - 1] if idx > 0 else None
                hint = f" Run stage {prev!r} first (or stage: all)." if prev else ""
                raise SystemExit(f"[{name}] can't run -- missing {missing}.{hint}")

        extra_args = ["--config", str(args.config)] if takes_config else []
        elapsed = sum(run_script(name, script, label, extra_args) for script in scripts)
        timings.append((name, label, elapsed))

    total = sum(t for _, _, t in timings)
    print(f"\n{'='*70}\nPipeline summary\n{'='*70}")
    for name, label, elapsed in timings:
        print(f"  [OK] {name:12s} {label:35s} {elapsed:6.1f}s")
    print(f"  {'-'*58}")
    print(f"  Total: {total:.1f}s")


if __name__ == "__main__":
    main()
