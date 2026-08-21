#!/usr/bin/env python3
"""Cut a Model 2 result file down to the rows another run actually reached.

The cascade run scores a seeded random subset of the clean test -- as many rows as the
Colab session lived for -- while the reference-input run scored all 5,687. Comparing the
two summaries directly would compare different samples. This rebuilds the reference run
over exactly the cascade's rows, in the cascade's order, so `mcnemar_conditions.py` (which
pairs by position and asserts on `product_smiles`) sees two runs of the same records.

The summary is recomputed from the kept generations by `reaggregate_conditions_topk`, the
same code that reproduces the evaluator's own numbers on a full file, so both sides of the
comparison are aggregated identically.

Example:
    python scripts/models/subset_conditions_topk.py \\
        --results experiments/m3_backup/M3_conditions_fullside_clean_topk.json \\
        --like experiments/v2_model2_roles/cascade_m3_clean_topk.json \\
        --output experiments/v2_model2_roles/M3_on_cascade_subset_topk.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reaggregate_conditions_topk import recompute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="Full-test result file to cut down.")
    parser.add_argument(
        "--like",
        type=Path,
        required=True,
        help="Result file whose records carry `reference.cascade_index`; its rows and order are used.",
    )
    parser.add_argument("--index-key", default="cascade_index")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full = json.loads(args.results.read_text(encoding="utf-8"))
    like = json.loads(args.like.read_text(encoding="utf-8"))

    kept = []
    for record in like["records"]:
        index = record["reference"][args.index_key]
        source = full["records"][index]
        if source["product_smiles"] != record["product_smiles"]:
            raise SystemExit(
                f"row {index} is {source['product_smiles']} in {args.results.name} but "
                f"{record['product_smiles']} in {args.like.name}: the two runs are not the same test file"
            )
        kept.append(source)

    subset = {"summary": dict(full["summary"]), "records": kept}
    subset["summary"] = recompute(subset)
    subset["summary"]["subset_of"] = args.results.name
    subset["summary"]["subset_like"] = args.like.name
    args.output.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
    print(f"{args.output}: {len(kept)} of {len(full['records'])} record(s)")


if __name__ == "__main__":
    main()
