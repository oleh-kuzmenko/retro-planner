"""Dated experiment-run folders: `experiments/<experiment_id>/<model_slug>/`.

`experiment_id` defaults to today's date so a run is easy to find later, but
is overridable so stages run on different days (memory constraints mean you
rarely run all of them the same day) can still be grouped under one
comparison. Every run folder is self-describing via `run_meta.json`, so
`scripts/aggregate_results.py` can discover model runs without knowing in
advance which/how many models were compared.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DEFAULT_EXPERIMENTS_ROOT = Path("experiments")
RESULTS_FILENAME = "results.json"
LOG_FILENAME = "inference_logs.json"
RUN_META_FILENAME = "run_meta.json"

_REDACT_KEY_MARKERS = ("key", "token", "secret", "password")


def default_experiment_id() -> str:
    return date.today().isoformat()


def create_experiment_run_dir(experiments_root: Path, experiment_id: str, model_slug: str) -> Path:
    run_dir = experiments_root / experiment_id / model_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _redact_cli_args(cli_args: dict) -> dict:
    return {
        key: ("***redacted***" if any(marker in key.lower() for marker in _REDACT_KEY_MARKERS) else value)
        for key, value in cli_args.items()
    }


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def start_run_meta(
    run_dir: Path,
    *,
    model_slug: str,
    script: str,
    cli_args: dict,
    input_path: Path | None,
) -> None:
    """Write an initial `run_meta.json` with `status: "running"`.

    Called at script startup, before any inference happens, so a crashed run
    leaves an honest "running" marker instead of silently looking finished.
    """
    meta = {
        "model_slug": model_slug,
        "script": script,
        "cli_args": _redact_cli_args(cli_args),
        "input_path": str(input_path) if input_path else None,
        "input_sha256": hash_file(input_path) if input_path else None,
        "record_count": None,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
    }
    save_json_atomic(run_dir / RUN_META_FILENAME, meta)


def finish_run_meta(run_dir: Path, *, record_count: int, status: str = "completed") -> None:
    """Patch `run_meta.json` with the final record count/status/timestamp."""
    meta_path = run_dir / RUN_META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["record_count"] = record_count
    meta["status"] = status
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_json_atomic(meta_path, meta)


class InferenceLogWriter:
    """Append-only inference log: exact prompt(s) and raw response(s) per call.

    Loaded once per run and rewritten in full after every record, so an
    interrupted run (OOM, Ctrl-C) keeps every entry logged up to that point.
    """

    def __init__(self, run_dir: Path):
        self.path = run_dir / LOG_FILENAME
        self.entries: list[dict] = (
            json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        )

    def append(self, entry: dict) -> None:
        self.entries.append(entry)
        save_json_atomic(self.path, self.entries)
