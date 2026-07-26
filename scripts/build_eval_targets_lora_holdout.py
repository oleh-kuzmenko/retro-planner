#!/usr/bin/env python3
"""Write a held-out JSON eval set for the fine-tuned Qwen+LoRA reactant model.

`build_eval_targets_uspto.py` samples straight from `pingzhili/uspto-50k`
with its own seed, independent of the train/eval/test partition that
`research/fine-tune/v2/01_prepare_reactant_data.ipynb` carved out of that
same corpus before training `oleh13/retro-reactants-qwen25-7b-lora`. That
makes it unsafe to score the LoRA model against it: cross-checking showed
91/100 of its targets already sit in the LoRA training split, so most
"test" predictions are memorized, not generalized.

This script instead reads the notebook's own `reactants_test.jsonl` --
the slice it held out of training by construction -- and reshapes it into
the same `product_smiles`/`reactants_smiles` schema `build_eval_targets_*.py`
produces, so it's a genuine held-out set for this specific model.

Example:
    python scripts/build_eval_targets_lora_holdout.py
    python scripts/build_eval_targets_lora_holdout.py --count 100 --seed 0 \
        --output data/uspto_eval_targets_lora_holdout.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Iterator

from indexing_common import take_unique_by_product, write_eval_targets

LOGGER = logging.getLogger("retro_eval.build_eval_targets_lora_holdout")

DEFAULT_INPUT = Path(
    "research/fine-tune/v2/01/content/retro_condition_agent/data/reactants_test.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a JSON eval set for the Qwen+LoRA reactant model, drawn only from "
            "the held-out test split its own training data prep produced."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("data/uspto_eval_targets_lora_holdout.json")
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling, kept fixed so the eval set is reproducible.",
    )
    return parser.parse_args()


def iter_payloads(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            user_payload = json.loads(record["messages"][1]["content"])
            assistant_payload = json.loads(record["messages"][2]["content"])
            product_smiles = user_payload.get("product_smiles")
            reactants = assistant_payload.get("reactants")
            if not product_smiles or not reactants:
                continue
            yield {
                "product_smiles": product_smiles,
                "reactants_smiles": ".".join(reactants),
            }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    if not args.input.exists():
        raise SystemExit(
            f"{args.input} not found. Run research/fine-tune/v2/01_prepare_reactant_data.ipynb "
            "first, or pass --input pointing at its reactants_test.jsonl output."
        )

    payloads = list(iter_payloads(args.input))
    random.Random(args.seed).shuffle(payloads)
    targets = take_unique_by_product(iter(payloads), args.count)
    if len(targets) < args.count:
        LOGGER.warning("Only found %d target(s); requested %d.", len(targets), args.count)

    write_eval_targets(args.output, targets, fields=("product_smiles", "reactants_smiles"))


if __name__ == "__main__":
    main()
