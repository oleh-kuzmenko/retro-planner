#!/usr/bin/env python3
"""Write a leak-free USPTO-50K JSON test set from the canonical `test` split.

`build_eval_targets_uspto.py` samples from `pingzhili/uspto-50k`, which ships
only a `train` (49015 rows) / `validation` (1001 rows) split -- no `test` split
at all -- so a set built from it mixes the canonical train/val partitions back
in. This script instead samples from `bisectgroup/USPTO_50K`, a mirror of the
canonical 40008/5001/5007 division used across the retrosynthesis literature,
restricted to its `test` partition. `build_train_data_uspto.py` writes the
`train` partition of the same mirror, so the two are disjoint by construction.

Sampling is a seeded shuffle followed by first-occurrence-per-product, so a
smaller `--count` is a strict prefix of a larger one at the same `--seed`: the
1000-record comparison set is a subset of the full test set, and numbers on the
two are directly comparable.

Example:
    python scripts/build_eval_targets_uspto_test_holdout.py \
        --count 100000 --output data/v2_uspto_test_holdout.json
    python scripts/build_eval_targets_uspto_test_holdout.py \
        --count 1000 --output data/v2_uspto_test_holdout_1000.json
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from indexing_common import suppress_rdkit_warnings, take_unique_by_product, write_eval_targets
from sources_uspto import USPTO_PAYLOAD_FIELDS, iter_uspto_payloads, require_uspto_dependencies

LOGGER = logging.getLogger("retro_eval.build_eval_targets_uspto_test_holdout")

DEFAULT_DATASET = "bisectgroup/USPTO_50K"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a JSON eval set drawn only from the canonical USPTO-50K test split."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--count",
        type=int,
        default=100000,
        help="Upper bound on records; the default keeps every unique product in the split.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/v2_uspto_test_holdout.json"))
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling, kept fixed so the eval set is reproducible.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    require_uspto_dependencies()
    suppress_rdkit_warnings()

    payloads = list(iter_uspto_payloads(args.dataset, splits=("test",)))
    random.Random(args.seed).shuffle(payloads)
    targets = take_unique_by_product(iter(payloads), args.count)
    if len(targets) < args.count:
        LOGGER.info("Split yielded %d unique product(s); requested up to %d.", len(targets), args.count)

    write_eval_targets(args.output, targets, fields=USPTO_PAYLOAD_FIELDS)


if __name__ == "__main__":
    main()
