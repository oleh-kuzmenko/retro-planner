"""Discover every model run under one experiment id and build the comparison CSV.

Model identity comes entirely from directory names + `run_meta.json` --
nothing here hardcodes how many models exist or what they're called, so
adding a 5th/6th model later is just running its script with the same
`--experiment-id`, not a code change here.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from retro_eval.evaluation import is_exact_match_smiles, is_valid_smiles
from retro_eval.harness.experiment import RESULTS_FILENAME, RUN_META_FILENAME
from retro_eval.harness.records import EvalRecord

LOGGER = logging.getLogger(__name__)

FINAL_CSV_FILENAME = "final_aggregated_results.csv"


@dataclass(frozen=True)
class ModelRun:
    model_slug: str
    run_dir: Path
    meta: dict
    results_by_index: dict[int, dict]


def discover_model_runs(experiment_dir: Path) -> list[ModelRun]:
    """Find every `<experiment_dir>/<model_slug>/` with both `run_meta.json` and `results.json`."""
    if not experiment_dir.is_dir():
        raise SystemExit(f"Experiment directory not found: {experiment_dir}")

    runs: list[ModelRun] = []
    for candidate in sorted(experiment_dir.iterdir()):
        if not candidate.is_dir():
            continue
        meta_path = candidate / RUN_META_FILENAME
        results_path = candidate / RESULTS_FILENAME
        if not meta_path.exists() or not results_path.exists():
            LOGGER.warning("Skipping %s: missing %s or %s.", candidate, RUN_META_FILENAME, RESULTS_FILENAME)
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        results = json.loads(results_path.read_text(encoding="utf-8"))
        if meta.get("status") != "completed":
            LOGGER.warning(
                "%s has status=%s (not 'completed') -- including it anyway with whatever it wrote so far.",
                candidate,
                meta.get("status"),
            )
        runs.append(
            ModelRun(
                model_slug=candidate.name,
                run_dir=candidate,
                meta=meta,
                results_by_index={row["index"]: row for row in results},
            )
        )

    return sorted(runs, key=lambda run: run.model_slug)


def _check_input_hash_consistency(records_sha256: str | None, model_runs: list[ModelRun]) -> None:
    if records_sha256 is None:
        return
    for run in model_runs:
        run_hash = run.meta.get("input_sha256")
        if run_hash and run_hash != records_sha256:
            LOGGER.warning(
                "Model '%s' ran against a different --input file (sha256 mismatch) than the one "
                "passed to aggregate_results.py; ground truth may not line up with its predictions.",
                run.model_slug,
            )


def build_comparison_rows(
    records: list[EvalRecord] | None,
    model_runs: list[ModelRun],
    records_sha256: str | None = None,
) -> tuple[list[str], list[dict]]:
    """Join ground truth (if given) with every discovered model's predictions, by record index."""
    _check_input_hash_consistency(records_sha256, model_runs)

    ground_truth: dict[int, EvalRecord] = {record.index: record for record in (records or [])}
    all_indices = sorted(
        set(ground_truth) | {index for run in model_runs for index in run.results_by_index}
    )

    fieldnames = ["index", "product_smiles", "ground_truth_reactants"]
    for run in model_runs:
        fieldnames += [
            f"predicted_reactants_{run.model_slug}",
            f"exact_match_{run.model_slug}",
            f"valid_{run.model_slug}",
        ]

    rows: list[dict] = []
    for index in all_indices:
        record = ground_truth.get(index)
        product_smiles = record.product_smiles if record else next(
            (run.results_by_index[index]["product_smiles"] for run in model_runs if index in run.results_by_index),
            "",
        )
        reference = record.reactants_smiles if record else next(
            (
                run.results_by_index[index].get("reactants_smiles")
                for run in model_runs
                if index in run.results_by_index and run.results_by_index[index].get("reactants_smiles")
            ),
            None,
        )

        row = {"index": index, "product_smiles": product_smiles, "ground_truth_reactants": reference or ""}
        for run in model_runs:
            predicted = (run.results_by_index.get(index) or {}).get("predicted_reactants_smiles", "")
            row[f"predicted_reactants_{run.model_slug}"] = predicted
            row[f"valid_{run.model_slug}"] = is_valid_smiles(predicted)
            row[f"exact_match_{run.model_slug}"] = (
                is_exact_match_smiles(predicted, reference) if reference else False
            )
        rows.append(row)

    return fieldnames, rows


def write_comparison_csv(csv_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(model_runs: list[ModelRun], rows: list[dict]) -> dict[str, dict]:
    """Per-model validity%/exact-match% over the joined comparison rows."""
    summary: dict[str, dict] = {}
    for run in model_runs:
        total = len(run.results_by_index)
        if total == 0:
            continue
        valid = sum(1 for row in rows if row.get(f"valid_{run.model_slug}"))
        exact_match = sum(1 for row in rows if row.get(f"exact_match_{run.model_slug}"))
        summary[run.model_slug] = {"total": total, "valid": valid, "exact_match": exact_match}
    return summary
