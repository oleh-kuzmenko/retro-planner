"""Generic crash-safe per-record inference loop shared by every model-runner script."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from retro_eval.harness.experiment import RESULTS_FILENAME, InferenceLogWriter, save_json_atomic
from retro_eval.harness.records import EvalRecord

LOGGER = logging.getLogger(__name__)


def require_modules(module_names: tuple[str, ...], extra: str) -> None:
    missing = [name for name in module_names if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            f"Missing dependencies for this script: {', '.join(missing)}. "
            f'Install them with `pip install -e ".[{extra}]"`.'
        )


def run_model_over_records(
    records: list[EvalRecord],
    run_dir: Path,
    desc: str,
    build_request: Callable[[EvalRecord], Any],
    call_model: Callable[[Any], str],
    parse_response: Callable[[str, EvalRecord], tuple[str, dict]],
) -> list[dict]:
    """Run `call_model(build_request(record))` for every record, parse, and save incrementally.

    Rewrites `run_dir/results.json` and appends to `run_dir/inference_logs.json`
    after *every* record, so an interrupted run (OOM, Ctrl-C) loses at most the
    in-flight record. `build_request` shapes what gets sent (a chat message
    list, a bare SMILES string, ...); the exact request is logged verbatim
    alongside the raw response for later qualitative citation.
    """
    results: list[dict] = []
    log = InferenceLogWriter(run_dir)
    results_path = run_dir / RESULTS_FILENAME

    for record in tqdm(records, desc=desc):
        request = build_request(record)
        entry: dict = {
            "index": record.index,
            "product_smiles": record.product_smiles,
            "request": request,
        }
        predicted, raw = "", ""
        try:
            raw = call_model(request)
            entry["raw_response"] = raw
            predicted, extra = parse_response(raw, record)
            entry.update(extra)
        except Exception as exc:
            LOGGER.exception("%s failed for index=%d", desc, record.index)
            entry["error"] = str(exc)

        results.append(
            {
                "index": record.index,
                "product_smiles": record.product_smiles,
                "reactants_smiles": record.reactants_smiles,
                "predicted_reactants_smiles": predicted,
                "raw_response": raw,
            }
        )
        save_json_atomic(results_path, results)
        log.append(entry)

    return results
