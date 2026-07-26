#!/usr/bin/env python3
"""Step 6: aggregate every discovered model run under one experiment id.

Reads `experiments/<experiment_id>/<model_slug>/{run_meta.json,results.json}`
for every model_slug found (no hardcoded model list -- add a 5th/6th model
by running its script with the same `--experiment-id`, not by editing this
script), joins by record index against `--input` (if given) for ground
truth, and writes `experiments/<experiment_id>/final_aggregated_results.csv`.

Example:
    python scripts/aggregate_results.py --input data/uspto_eval_targets.json
    python scripts/aggregate_results.py --experiment-id 2026-07-25 --input data/uspto_eval_targets.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from retro_eval.harness.aggregate import (
    FINAL_CSV_FILENAME,
    build_comparison_rows,
    discover_model_runs,
    summarize,
    write_comparison_csv,
)
from retro_eval.harness.experiment import DEFAULT_EXPERIMENTS_ROOT, default_experiment_id, hash_file
from retro_eval.harness.records import load_records

LOGGER = logging.getLogger("retro_eval.aggregate_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate every model run under one experiment id into one CSV.")
    parser.add_argument("--experiment-id", default=None, help="Defaults to today's date (YYYY-MM-DD).")
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Original dataset JSON, for ground truth + canonical record ordering. Optional: "
        "without it, ground truth is pulled from whichever model run happened to log it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    experiment_id = args.experiment_id or default_experiment_id()
    experiment_dir = args.experiments_root / experiment_id

    model_runs = discover_model_runs(experiment_dir)
    if not model_runs:
        raise SystemExit(f"No completed model runs found under {experiment_dir}.")
    LOGGER.info("Discovered %d model run(s): %s", len(model_runs), [run.model_slug for run in model_runs])

    records = load_records(args.input) if args.input else None
    records_sha256 = hash_file(args.input) if args.input else None

    fieldnames, rows = build_comparison_rows(records, model_runs, records_sha256)
    csv_path = experiment_dir / FINAL_CSV_FILENAME
    write_comparison_csv(csv_path, fieldnames, rows)
    LOGGER.info("Wrote aggregated comparison for %d target(s) to %s", len(rows), csv_path)

    for model_slug, stats in summarize(model_runs, rows).items():
        LOGGER.info(
            "  %s: validity=%d/%d (%.1f%%), exact_match=%d/%d (%.1f%%), "
            "core_exact_match=%d/%d (%.1f%%)",
            model_slug,
            stats["valid"],
            stats["total"],
            100 * stats["valid"] / stats["total"],
            stats["exact_match"],
            stats["total"],
            100 * stats["exact_match"] / stats["total"],
            stats["core_exact_match"],
            stats["total"],
            100 * stats["core_exact_match"] / stats["total"],
        )


if __name__ == "__main__":
    main()
