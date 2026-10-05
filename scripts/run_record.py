"""
Run records -- a saved copy of every pipeline run's results AND the exact
setup that produced them, so any result can be traced back and re-run.

run_pipeline.py calls this automatically (turn off with
`pipeline.save_run: false` in config.yaml or `--no-save`). Each run gets
its own folder:

    results/runs/<run_id>/
        run_info.json     when, which stages, timings, status, git commit,
                          Python + package versions, raw-data fingerprint
        config.yaml       the exact config the run used (config.yaml plus
                          any --grain / --models / --no-tune overrides)
        tuning.yaml       the tuning config (only if the tune stage ran)
        scores.json       every results.json row this run wrote
        manifests/        feature / split manifests, AQ verification,
                          excluded-station logs (which features, stations,
                          cutoff dates the models actually saw)
        outputs/          every results/ file this run created or updated
                          (per-station scores, tuning, evaluation tables/plots)
        pipeline.log      the full console output

and one row in results/runs/index.csv, a one-line-per-run overview.

To reproduce a run: check out its git commit (run_info.json), then
    python run_pipeline.py --config results/runs/<run_id>/config.yaml
Fitted model files (results/saved_models/) are not copied -- they're
large and can be rebuilt from the config.

Every row a run writes to results/results.json is also tagged with its
run_id (via the PIPELINE_RUN_ID environment variable that run_pipeline.py
sets), so any score in results.json points back to its run folder.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import yaml

from scripts import paths

RUN_ID_ENV = "PIPELINE_RUN_ID"
RUNS_DIR = paths.RESULTS_DIR / "runs"
INDEX_PATH = RUNS_DIR / "index.csv"

PACKAGES = ["pandas", "numpy", "scikit-learn", "xgboost", "scipy", "holidays",
            "python-calamine", "pyshp", "pyyaml", "matplotlib"]

# results/ files that belong to a run's manifests (copied from processed/)
MANIFEST_FILES = [
    paths.FEATURE_MANIFEST_PATH,
    paths.SPLIT_MANIFEST_PATH,
    paths.PROCESSED_DIR / "aq_site_verification.csv",
    paths.PROCESSED_DIR / "excluded_stations_log_daily.csv",
    paths.PROCESSED_DIR / "excluded_stations_log_hourly.csv",
]


def make_run_id(grains: list[str], name: str | None) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = f"_{name}" if name else ""
    return f"{stamp}_{'-'.join(grains)}{suffix}"


def write_effective_config(config: dict, path: Path, overrides: list[str]) -> Path:
    """The config the run ACTUALLY used (config.yaml plus any command-line
    overrides such as --grain / --models / --no-tune). Every stage reads this
    file, and it's what gets saved with the run -- so re-running it with
    --config reproduces the run exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Effective config for this run = config/config.yaml"
              + (f" + command-line overrides: {' '.join(overrides)}" if overrides else "")
              + "\n# Re-run exactly with:  python run_pipeline.py --config <this file>\n\n")
    path.write_text(header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def current_run_id() -> str | None:
    """The run_id set by run_pipeline.py, or None when a script is run on its own."""
    return os.environ.get(RUN_ID_ENV)


# ---------------------------------------------------------------------------
# Environment / provenance
# ---------------------------------------------------------------------------

def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=paths.ROOT_DIR, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_info() -> dict:
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    status = _git("status", "--porcelain", "--untracked-files=no") or ""
    return {
        "available": True,
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # True = there were uncommitted code changes, so the commit alone
        # doesn't fully reproduce this run
        "uncommitted_changes": bool(status.strip()),
    }


def package_versions() -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def raw_data_fingerprint() -> dict:
    """A short hash of every raw input file's name and contents, so you can
    tell whether two runs used exactly the same data."""
    files = sorted(p for p in paths.RAW_DIR.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        digest.update(str(path.relative_to(paths.RAW_DIR)).replace("\\", "/").encode())
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        total_bytes += path.stat().st_size
    return {
        "files": len(files),
        "total_mb": round(total_bytes / 1e6, 1),
        "sha256": digest.hexdigest()[:16],
        # optional inputs change the features, so record whether they were there
        "speed_zones_present": paths.SPEED_ZONES_SHP.exists(),
        "aq_sites_present": paths.AQ_SITES_PATH.exists(),
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_run(
    run_id: str,
    config_path: Path,
    stages: list[tuple[str, str, float]],
    status: str,
    started: datetime,
    command: list[str],
    log_text: str,
) -> Path:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_names = [name for name, _, _ in stages]
    config = yaml.safe_load(Path(config_path).read_text()) or {}

    # configs (the effective config is normally already written in run_dir)
    if Path(config_path).resolve() != (run_dir / "config.yaml").resolve():
        shutil.copy2(config_path, run_dir / "config.yaml")
    if "tune" in stage_names and paths.TUNING_CONFIG_PATH.exists():
        shutil.copy2(paths.TUNING_CONFIG_PATH, run_dir / "tuning.yaml")

    # manifests -- only if this run rebuilt them or they're what it trained on
    manifest_dir = run_dir / "manifests"
    for source in MANIFEST_FILES:
        if source.exists():
            manifest_dir.mkdir(exist_ok=True)
            shutil.copy2(source, manifest_dir / source.name)

    # scores this run wrote
    rows = []
    if paths.RESULTS_PATH.exists():
        rows = [r for r in json.loads(paths.RESULTS_PATH.read_text()) if r.get("run_id") == run_id]
    (run_dir / "scores.json").write_text(json.dumps(rows, indent=2, default=str))

    # every results/ file created or updated during the run
    skip = {RUNS_DIR, paths.SAVED_MODELS_DIR}
    copied = 0
    start_ts = started.timestamp()
    for path in paths.RESULTS_DIR.rglob("*"):
        if not path.is_file() or any(s in path.parents for s in skip):
            continue
        if path == paths.RESULTS_PATH or path.stat().st_mtime < start_ts:
            continue
        target = run_dir / "outputs" / path.relative_to(paths.RESULTS_DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1

    (run_dir / "pipeline.log").write_text(log_text, encoding="utf-8")

    training = config.get("training", {})
    enabled_models = [m for m, c in (training.get("models") or {}).items() if (c or {}).get("enabled")]
    exclusions = (config.get("data", {}) or {}).get("station_exclusions") or {}
    info = {
        "run_id": run_id,
        "status": status,
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(command),

        "stages": [{"stage": n, "seconds": round(t, 1)} for n, _, t in stages],
        "grains": training.get("grain"),
        "feature_sets": training.get("feature_sets"),
        "models_enabled": enabled_models,
        "station_exclusions_enabled": bool(exclusions.get("enabled")),
        "scores_saved": len(rows),
        "output_files_saved": copied,
        "git": git_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "raw_data": raw_data_fingerprint(),
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    rebuild_index()
    return run_dir


def _best(rows: list[dict], grain: str) -> tuple[dict | None, dict | None]:
    """Best ML model on one grain by validation R2, plus its test row."""
    val = [r for r in rows if r.get("grain") == grain and r.get("split") == "val"
           and not r["model"].startswith("naive_") and r.get("r2") is not None]
    if not val:
        return None, None
    best = max(val, key=lambda r: r["r2"])
    same = lambda r: all(r.get(k) == best.get(k) for k in ("model", "stage", "grain", "feature_set"))
    test = next((r for r in rows if r.get("split") == "test" and same(r)), None)
    return best, test


def rebuild_index() -> None:
    """Rewrite results/runs/index.csv from every saved run folder -- one row
    per run, with the best ML model per grain (chosen on validation R2)."""
    rows_out = []
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if (p / "run_info.json").exists()):
        info = json.loads((run_dir / "run_info.json").read_text())
        scores_path = run_dir / "scores.json"
        scores = json.loads(scores_path.read_text()) if scores_path.exists() else []
        grains = info.get("grains", info.get("grain"))
        row = {
            "run_id": info["run_id"],
            "status": info["status"],
            "started": info["started"],
            "stages": "+".join(s["stage"] for s in info["stages"]),
            "grains": "+".join(grains) if isinstance(grains, list) else grains,
            "models_enabled": "+".join(info.get("models_enabled", [])),
            "station_exclusions": info.get("station_exclusions_enabled"),
            "speed_zones": info.get("raw_data", {}).get("speed_zones_present"),
        }
        for grain in ("daily", "hourly"):
            best, test = _best(scores, grain)
            row[f"best_{grain}_model"] = (
                f"{best['model']} ({best.get('stage')}, {best['feature_set']})" if best else "")
            row[f"best_{grain}_val_r2"] = round(best["r2"], 4) if best else None
            row[f"best_{grain}_test_r2"] = round(test["r2"], 4) if test else None
        row["git_commit"] = (info.get("git", {}).get("commit") or "")[:8]
        row["uncommitted_changes"] = info.get("git", {}).get("uncommitted_changes")
        rows_out.append(row)
    if not rows_out:
        return
    with open(INDEX_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
