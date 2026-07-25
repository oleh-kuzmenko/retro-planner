#!/usr/bin/env python3
"""Step 1 (USPTO): write a fixed JSON test set, held out for every model comparison stage.

Every stage (ReactionT5v2, ChemLLM, Qwen2.5-7B+LoRA, Llama-70B+RAG+CoT) runs
against the exact same file this produces, so `index_uspto_to_qdrant.py`
excludes it from the RAG index by default -- these targets stay unseen.

Example:
    python scripts/build_eval_targets_uspto.py
    python scripts/build_eval_targets_uspto.py --count 100 --output data/uspto_eval_targets.json
"""

from __future__ import annotations

import argparse
import logging
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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    require_uspto_dependencies()
    suppress_rdkit_warnings()

    targets = take_unique_by_product(iter_uspto_payloads(args.dataset), args.count)
    if len(targets) < args.count:
        LOGGER.warning("Only found %d target(s); requested %d.", len(targets), args.count)

    write_eval_targets(args.output, targets, fields=USPTO_PAYLOAD_FIELDS)


if __name__ == "__main__":
    main()
