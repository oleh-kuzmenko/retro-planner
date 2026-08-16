#!/usr/bin/env python3
"""Drop leaked rows from a Model 2 conditions test set.

The ORD pools (60k / 150k / 300k) were each sampled by an independent run of
`build_train_data_ord.py`. Sampling is seeded, but a larger pool is not a
superset of a smaller one, and the conditions train/val/test split is drawn
*inside* each pool. So the 300k pool's `conditions_test.jsonl` -- the shared
test set used to compare the two Model 2 checkpoints -- overlaps the 60k pool's
`conditions_train.jsonl`: 2,027 of its 7,714 rows are training rows for the
41,139-row checkpoint (identical on both `product_smiles` and the canonical
(product, reactants) pair).

This script writes the leak-free subset, so both checkpoints can be scored on a
test set neither of them was trained on.

Example:
    python scripts/build_clean_conditions_test.py
    python scripts/build_clean_conditions_test.py \\
        --test-file data/v2_ord_train_300k/conditions_test.jsonl \\
        --exclude-file data/v2_ord_train/conditions_train.jsonl \\
        --output data/v2_ord_conditions_test_clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger("retro_eval.build_clean_conditions_test")

DEFAULT_TEST_FILE = Path("data/v2_ord_train_300k/conditions_test.jsonl")
DEFAULT_EXCLUDE_FILES = [Path("data/v2_ord_train/conditions_train.jsonl")]
DEFAULT_OUTPUT = Path("data/v2_ord_conditions_test_clean.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the leak-free subset of a conditions test set."
    )
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument(
        "--exclude-file",
        action="append",
        type=Path,
        default=None,
        help=(
            "Rows whose product_smiles appears in this file are dropped. "
            "Repeatable; defaults to the 60k pool's conditions_train.jsonl."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    """Load either a JSON array or one JSON object per line."""
    text = path.read_text(encoding="utf-8")
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return records


def load_excluded_products(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        products = {r["product_smiles"] for r in load_records(path) if r.get("product_smiles")}
        LOGGER.info("Loaded %d product(s) to exclude from %s", len(products), path)
        excluded |= products
    return excluded


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    exclude_files = args.exclude_file or DEFAULT_EXCLUDE_FILES

    test_rows = load_records(args.test_file)
    excluded = load_excluded_products(exclude_files)

    kept = [row for row in test_rows if row.get("product_smiles") not in excluded]
    dropped = len(test_rows) - len(kept)
    LOGGER.info(
        "Test set %s: %d row(s) in, %d leaked row(s) dropped, %d kept (%.1f%%).",
        args.test_file, len(test_rows), dropped, len(kept),
        100 * len(kept) / len(test_rows) if test_rows else 0.0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote %d row(s) to %s", len(kept), args.output)


if __name__ == "__main__":
    main()
