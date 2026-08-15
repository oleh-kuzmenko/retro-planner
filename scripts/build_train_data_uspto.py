#!/usr/bin/env python3
"""Write the canonical USPTO-50K train split as JSONL, for Model 1 fine-tuning.

Uses `bisectgroup/USPTO_50K` (the same canonical 40008/5001/5007 train/val/test
mirror `build_eval_targets_uspto_test_holdout.py` scores against), restricted
to its `train` partition.

The canonical split is not perfectly disjoint: 69 products of the `test`
partition also occur in `train`, 48 of them as the identical (product,
reactants) pair -- an artifact of USPTO-50K itself, not of this sampling. The
test set is left standard so results stay comparable to the literature, and the
duplicates are dropped from the training side instead via `--exclude-file`.

Example:
    python scripts/build_train_data_uspto.py \
        --exclude-file data/v2_uspto_test_holdout.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from indexing_common import load_excluded_product_smiles, suppress_rdkit_warnings
from sources_uspto import iter_uspto_payloads, require_uspto_dependencies

LOGGER = logging.getLogger("retro_eval.build_train_data_uspto")

DEFAULT_DATASET = "bisectgroup/USPTO_50K"
FIELDS = ("product_smiles", "reactants_smiles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write USPTO-50K canonical train split as JSONL.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--split",
        default="train",
        help="Canonical partition to write: `train` for fine-tuning, `val` for the eval split.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/v2_uspto_train/reactants_train.jsonl"))
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=Path("data/v2_uspto_test_holdout.json"),
        help="Products in this eval-targets file are dropped from the training rows.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    require_uspto_dependencies()
    suppress_rdkit_warnings()

    excluded = load_excluded_product_smiles(args.exclude_file)

    seen = set()
    rows = []
    dropped = 0
    for payload in iter_uspto_payloads(args.dataset, splits=(args.split,)):
        product = payload.get("product_smiles")
        if not product or product in seen:
            continue
        if product in excluded:
            dropped += 1
            continue
        seen.add(product)
        rows.append({key: payload[key] for key in FIELDS if key in payload})
    LOGGER.info("Dropped %d row(s) whose product occurs in %s", dropped, args.exclude_file)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote %d row(s) to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
