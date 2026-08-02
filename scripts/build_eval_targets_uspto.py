#!/usr/bin/env python3
"""USPTO-50K: write a fixed JSON test set for the independent generalization test.

Held out from training; mirrors `build_eval_targets_ord.py`.

Targets are a seeded random sample (`--seed`), not just the first N unique
products in dataset order, mirroring `build_eval_targets_ord.py`.

Example:
    python scripts/build_eval_targets_uspto.py
    python scripts/build_eval_targets_uspto.py --count 100 --seed 0 --output data/uspto_eval_targets.json
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from indexing_common import suppress_rdkit_warnings, take_unique_by_product, write_eval_targets
from sources_uspto import (
    DEFAULT_DATASET,
    USPTO_PAYLOAD_FIELDS,
    iter_uspto_payloads,
    require_uspto_dependencies,
)

LOGGER = logging.getLogger("retro_eval.build_eval_targets_uspto")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a fixed USPTO-50K JSON test set for model comparison."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/uspto_eval_targets.json"))
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

    payloads = list(iter_uspto_payloads(args.dataset))
    random.Random(args.seed).shuffle(payloads)
    targets = take_unique_by_product(iter(payloads), args.count)
    if len(targets) < args.count:
        LOGGER.warning("Only found %d target(s); requested %d.", len(targets), args.count)

    write_eval_targets(args.output, targets, fields=USPTO_PAYLOAD_FIELDS)


if __name__ == "__main__":
    main()
