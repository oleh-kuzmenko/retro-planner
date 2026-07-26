#!/usr/bin/env python3
"""Write a genuinely held-out USPTO-50K JSON test set for ReactionT5v2.

`build_eval_targets_uspto.py` samples from `pingzhili/uspto-50k`, which
ships only a `train` (49015 rows) / `validation` (1001 rows) split --
no `test` split at all. That's unsafe to score
`sagawa/ReactionT5v2-retrosynthesis-USPTO_50k` against: its model card
says it was fine-tuned on "USPTO_50k's train split" (the canonical
40008/5001/5007 train/val/test division used across the retrosynthesis
literature), and cross-checking showed 89/100 of `uspto_eval_targets.json`
already sit in that canonical train or val split -- most "test"
predictions are memorized, not generalized (only 11/100 are in the
canonical test split; on those 11, exact_match drops from the reported
90/100 to 9/11).

This script instead samples from `bisectgroup/USPTO_50K`, a mirror of
that same canonical split, restricted to its `test` partition -- the one
sagawa's own train.csv/val.csv never touched.

Example:
    python scripts/build_eval_targets_uspto_test_holdout.py
    python scripts/build_eval_targets_uspto_test_holdout.py --count 100 --seed 0 \
        --output data/uspto_eval_targets_test_holdout.json
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
            "Write a JSON eval set for ReactionT5v2, drawn only from the canonical "
            "USPTO-50k test split its own training data never touched."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("data/uspto_eval_targets_test_holdout.json")
    )
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
        LOGGER.warning("Only found %d target(s); requested %d.", len(targets), args.count)

    write_eval_targets(args.output, targets, fields=USPTO_PAYLOAD_FIELDS)


if __name__ == "__main__":
    main()
