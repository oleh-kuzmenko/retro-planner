#!/usr/bin/env python3
"""Build fresh ORD train/val (Model 1) and train/val/test (Model 2) splits from scratch.

Written for the diploma rebuild: earlier ad hoc train splits (and the RAG/LLM
comparison line of experiments) are no longer used or trusted as evidence.
This script draws ONE stratified, seeded, eval-excluded sample from ORD and
derives both downstream splits from it, so there is a single documented
source of truth:

- Model 1 (ReactionT5 retrosynthesis fine-tuning on ORD): product_smiles ->
  reactants_smiles, split into train/val. Model 1's *test* set is the
  separate, already-built `data/v2_ord_eval_targets.json`
  (`build_eval_targets_ord.py`) -- not reproduced here.
- Model 2 (reaction-condition prediction): the subset of the same pool that
  actually has at least one populated condition field (solvent/catalyst/
  temperature/yield -- ORD completeness varies a lot record to record), split
  into train/val/test.

Every eval target already sampled into `--eval-targets-file` is excluded
before sampling, by product_smiles, using the same helper
`index_ord_to_qdrant.py` uses -- so there is no leakage between this train
pool and the fixed eval set used to score Model 1.

Example:
    python scripts/build_train_data_ord.py --pool-count 60000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from indexing_common import (
    load_excluded_product_smiles,
    stratified_reservoir_sample,
    suppress_rdkit_warnings,
)
from sources_ord import (
    DEFAULT_ORD_REPO_ID,
    iter_ord_payloads,
    require_ord_dependencies,
    resolve_ord_data_dir,
)

LOGGER = logging.getLogger("retro_eval.build_train_data_ord")

REACTANTS_FIELDS = ("product_smiles", "reactants_smiles")
CONDITIONS_FIELDS = (
    "product_smiles",
    "reactants_smiles",
    "solvent",
    "temperature_celsius",
    "catalyst",
    "yield_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fresh, eval-excluded ORD train/val/test splits for Model 1 and Model 2."
    )
    parser.add_argument("--pool-count", type=int, default=60000)
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Defaults to max(1, --pool-count // 10), forcing >=10 distinct ORD source files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-targets-file",
        type=Path,
        default=Path("data/v2_ord_eval_targets.json"),
        help="Products in this file are excluded from the pool (Model 1's held-out test set).",
    )
    parser.add_argument("--val-count", type=int, default=3000, help="Model 1 (reactants) val size.")
    parser.add_argument("--conditions-val-frac", type=float, default=0.05)
    parser.add_argument("--conditions-test-frac", type=float, default=0.05)
    parser.add_argument("--ord-data-dir", type=Path, default=None)
    parser.add_argument("--ord-repo-id", default=DEFAULT_ORD_REPO_ID)
    parser.add_argument("--ord-allow-pattern", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/v2_ord_train"))
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            minimal = {key: row[key] for key in fields if key in row and row[key] is not None}
            handle.write(json.dumps(minimal, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote %d row(s) to %s", len(rows), path)


def has_any_condition(row: dict) -> bool:
    return any(
        row.get(field) not in (None, "", [])
        for field in ("solvent", "temperature_celsius", "catalyst", "yield_percent")
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    require_ord_dependencies(args.ord_data_dir)
    suppress_rdkit_warnings()

    excluded = load_excluded_product_smiles(args.eval_targets_file)

    ord_data_dir = resolve_ord_data_dir(args.ord_data_dir, args.ord_repo_id, args.ord_allow_pattern)
    max_per_source = args.max_per_source or max(1, args.pool_count // 10)

    def eval_excluded_payloads():
        for payload in iter_ord_payloads(ord_data_dir):
            if payload.get("product_smiles") in excluded:
                continue
            yield payload

    pool = stratified_reservoir_sample(
        eval_excluded_payloads(),
        args.pool_count,
        max_per_source,
        group_field="source_dataset",
        seed=args.seed,
    )
    LOGGER.info("Pool size after eval-exclusion and stratified sampling: %d", len(pool))

    rng = random.Random(args.seed)
    rng.shuffle(pool)

    # --- Model 1: reactants train/val ---
    val_count = min(args.val_count, max(1, len(pool) // 10))
    reactants_val = pool[:val_count]
    reactants_train = pool[val_count:]
    write_jsonl(args.output_dir / "reactants_train.jsonl", reactants_train, REACTANTS_FIELDS)
    write_jsonl(args.output_dir / "reactants_val.jsonl", reactants_val, REACTANTS_FIELDS)

    # --- Model 2: conditions train/val/test (subset with >=1 populated condition field) ---
    condition_rows = [row for row in pool if has_any_condition(row)]
    LOGGER.info(
        "%d/%d pooled reaction(s) have at least one populated condition field.",
        len(condition_rows),
        len(pool),
    )
    rng.shuffle(condition_rows)
    n = len(condition_rows)
    n_test = int(n * args.conditions_test_frac)
    n_val = int(n * args.conditions_val_frac)
    conditions_test = condition_rows[:n_test]
    conditions_val = condition_rows[n_test : n_test + n_val]
    conditions_train = condition_rows[n_test + n_val :]
    write_jsonl(args.output_dir / "conditions_train.jsonl", conditions_train, CONDITIONS_FIELDS)
    write_jsonl(args.output_dir / "conditions_val.jsonl", conditions_val, CONDITIONS_FIELDS)
    write_jsonl(args.output_dir / "conditions_test.jsonl", conditions_test, CONDITIONS_FIELDS)

    LOGGER.info(
        "Done. reactants: train=%d val=%d | conditions: train=%d val=%d test=%d",
        len(reactants_train),
        len(reactants_val),
        len(conditions_train),
        len(conditions_val),
        len(conditions_test),
    )


if __name__ == "__main__":
    main()
