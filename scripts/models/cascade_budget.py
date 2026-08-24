#!/usr/bin/env python3
"""End-to-end cascade accuracy as a function of how many proposals the system shows.

The funnel of `cascade_funnel.py` scores five routes but only the rank-1 conditions of each,
so the two links are read under different protocols -- top-5 for one, top-1 for the other,
while the rest of the thesis reports top-k everywhere. This script sweeps both depths and
prints the joint accuracy for every budget, which is what subsection 3.7 reports.

A record is credited once, when any of the first k1 routes carries a left side that matches
by role and any of that route's first k2 condition sets satisfies every field the source
records. The product k1 x k2 is the number of proposals a chemist would read.

Example:
    python scripts/models/cascade_budget.py \\
        --link1 experiments/v2_model2_roles/F6_link1_topk.json \\
        --test-file data/v2_ord_cascade_test_clean299.jsonl \\
        --link2 experiments/v2_model2_roles/E2_cascade_routes_topk.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade_funnel import (
    canonical,
    conditions_match,
    decode_and_parse,
    load,
    substrate_set,
    surviving_routes,
)

ROUTE_DEPTHS = (1, 3, 5)
CONDITION_DEPTHS = (1, 2, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--link1", type=Path, required=True, help="A *_link1_topk.json.")
    parser.add_argument("--test-file", type=Path, required=True, help="The 299-row reference file.")
    parser.add_argument("--link2", type=Path, required=True, help="Model 2 scored on the expansion.")
    parser.add_argument("--output", type=Path, help="Where to write the counts as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, by_product = load(args.link1, args.test_file)
    payload = json.loads(args.link2.read_text(encoding="utf-8"))
    summary = payload["summary"]
    target_format = summary.get("target_format", "compact")
    fields = tuple(summary["condition_fields"])
    scored = payload["records"]

    # Parse once: the sweep re-reads the same candidates for every budget.
    position = 0
    per_record: list[tuple[dict, list[tuple[bool, list[dict | None]]]]] = []
    for row in rows:
        product = canonical(row["product_smiles"])
        reference = substrate_set(row["full_reactants_smiles"], product)
        routes = surviving_routes(by_product.get(row["product_smiles"])) or [""]
        entry = []
        for offset, route in enumerate(routes):
            route_ok = route != "" and substrate_set(route, product) == reference
            raw = scored[position + offset]["candidates_raw"][: max(CONDITION_DEPTHS)]
            entry.append((route_ok, [decode_and_parse(text, target_format, fields) for text in raw]))
        position += len(routes)
        per_record.append((row, entry))

    total = len(rows)
    results = []
    print(f"{'routes':>7}{'conditions':>12}{'proposals':>11}{'strict':>16}{'relaxed':>16}")
    for k1 in ROUTE_DEPTHS:
        for k2 in CONDITION_DEPTHS:
            strict = relaxed = 0
            for row, entry in per_record:
                record_strict = record_relaxed = False
                for route_ok, parsed in entry[:k1]:
                    if not route_ok:
                        continue
                    for prediction in parsed[:k2]:
                        hit, near = conditions_match(row, prediction)
                        record_strict |= hit
                        record_relaxed |= near
                strict += record_strict
                relaxed += record_relaxed
            results.append(
                {
                    "routes": k1,
                    "conditions": k2,
                    "proposals": k1 * k2,
                    "strict": strict,
                    "strict_share": strict / total,
                    "relaxed": relaxed,
                    "relaxed_share": relaxed / total,
                }
            )
            print(
                f"{k1:>7}{k2:>12}{k1 * k2:>11}"
                f"{strict:>9} ({strict / total * 100:4.1f}%)"
                f"{relaxed:>9} ({relaxed / total * 100:4.1f}%)"
            )

    if args.output:
        args.output.write_text(
            json.dumps({"records": total, "budgets": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
