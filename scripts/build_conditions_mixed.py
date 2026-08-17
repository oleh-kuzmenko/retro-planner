#!/usr/bin/env python3
"""Assemble the mixed conditions corpus for the single, non-specialised Model 2.

Two earlier runs each solved half the problem. B3 trained on every row that carries any
condition field, and since only 26% of them record a temperature it learned that `?` is the
best answer there -- it abstained on 86.5% of test records. T2 trained only on rows that do
record one, which fixed the temperature (15.7% -> 51.6%) but threw away 71% of the chemistry,
and outside that subset its catalyst fell to 9.6% against B3's 29.3%.

The corpus here keeps both: every row that records a temperature, plus a seeded sample of
rows that do not, so the mandatory fields still get most of the gradient while the rest of
the chemistry stays in the training set. Rows that lack a mandatory field are not dropped
and not written as `?` -- `train_conditions_model.py --always-fields` gives them a shorter
target and a schema code in the input.

The mix is a budget decision, not a chemical one: temperature supervision costs passes over
a small set of rows, and breadth costs passes over a large one. Under a 6-hour GPU budget,
221k rows at 3 epochs buys 424k temperature examples and 664k of everything else.

Example:
    python scripts/build_conditions_mixed.py \
        --input-dir data/v2_ord_roles_900k \
        --test-file data/v2_ord_roles/conditions_test_clean.jsonl \
        --output-dir data/v2_ord_roles_mixed \
        --no-temp-rows 80000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

LOGGER = logging.getLogger("retro_eval.build_conditions_mixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Holds conditions_{train,val}.jsonl after the roles split.")
    parser.add_argument("--test-file", type=Path, required=True, help="Clean test; its products are removed from train and val.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-temp-rows", type=int, default=80000, help="How many rows without a temperature to keep in train.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Validation is subsampled in the same proportions as train, so eval_loss stays "
        "comparable across the two schemas rather than being dominated by one of them.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def has_temperature(row: dict) -> bool:
    return row.get("temperature_celsius") not in (None, "")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    rng = random.Random(args.seed)

    test_products = {row["product_smiles"] for row in read_rows(args.test_file)}
    LOGGER.info("Test products to exclude: %d", len(test_products))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, no_temp_budget in (("train", args.no_temp_rows), ("val", int(args.no_temp_rows * args.val_fraction))):
        rows = read_rows(args.input_dir / f"conditions_{split}.jsonl")
        kept = [row for row in rows if row["product_smiles"] not in test_products]
        leaked = len(rows) - len(kept)

        with_temp = [row for row in kept if has_temperature(row)]
        without = [row for row in kept if not has_temperature(row)]
        rng.shuffle(without)
        sample = without[:no_temp_budget]

        mixed = with_temp + sample
        rng.shuffle(mixed)

        out_path = args.output_dir / f"conditions_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in mixed:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        solvent = sum(1 for row in mixed if row.get("solvent") not in (None, ""))
        LOGGER.info(
            "%s: %d row(s) | with temperature %d (%.1f%%) | without %d of %d available | "
            "solvent %.1f%% | leaked rows removed %d",
            out_path,
            len(mixed),
            len(with_temp),
            100 * len(with_temp) / len(mixed),
            len(sample),
            len(without),
            100 * solvent / len(mixed),
            leaked,
        )


if __name__ == "__main__":
    main()
