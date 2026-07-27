#!/usr/bin/env python3
"""Write the canonical USPTO-50K train split as JSONL, for optional USPTO+ORD mixed fine-tuning.

Uses `bisectgroup/USPTO_50K` (the same canonical 40008/5001/5007 train/val/test
mirror `build_eval_targets_uspto_test_holdout.py` scores against), restricted
to its `train` partition -- disjoint by construction from the `test` split
used to build `data/v2_uspto_eval_targets.json`, so mixing this into Model 1's
fine-tuning data cannot leak into that eval set.

Example:
    python scripts/build_train_data_uspto.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from indexing_common import suppress_rdkit_warnings
from sources_uspto import iter_uspto_payloads, require_uspto_dependencies

LOGGER = logging.getLogger("retro_eval.build_train_data_uspto")

DEFAULT_DATASET = "bisectgroup/USPTO_50K"
FIELDS = ("product_smiles", "reactants_smiles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write USPTO-50K canonical train split as JSONL.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("data/v2_uspto_train/reactants_train.jsonl"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    require_uspto_dependencies()
    suppress_rdkit_warnings()

    seen = set()
    rows = []
    for payload in iter_uspto_payloads(args.dataset, splits=("train",)):
        product = payload.get("product_smiles")
        if not product or product in seen:
            continue
        seen.add(product)
        rows.append({key: payload[key] for key in FIELDS if key in payload})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote %d row(s) to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
