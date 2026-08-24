#!/usr/bin/env python3
"""Measure what the cascade costs Model 2 by feeding it a predicted left side.

Subsection 3.7 of the thesis reports that replacing the generative link 2 with the classifier
leaves the strict end-to-end count unchanged. That is a fact about the funnel, not an
explanation, and this script supplies the explanation.

`role match` credits a left side that names the same atom donors as the reference; it does not
require the rest of the loading to agree. So a route can be "correct" and still reach Model 2
as a different string than the one the reference would have given it. This script splits the
records whose left side link 1 got right into two groups -- routes that are the very same set
of molecules as the reference, and routes that share only the donors -- and scores the rank-1
conditions of each group twice: once from the route, once from the reference left side.

The comparison needs both files scored on the same 299 records:

    python scripts/evaluate_conditions_classifier.py --model-dir <final> \\
        --test-file data/v2_ord_cascade_expanded299.jsonl --output <routes>.json
    python scripts/evaluate_conditions_classifier.py --model-dir <final> \\
        --test-file data/v2_ord_cascade_test_clean299.jsonl --output <gold>.json

Example:
    python scripts/models/cascade_input_gap.py \\
        --link1 experiments/v2_model2_roles/F6_link1_topk.json \\
        --test-file data/v2_ord_cascade_test_clean299.jsonl \\
        --routes experiments/v2_model2_roles/E2_cascade_routes_topk.json \\
        --gold experiments/v2_model2_roles/E2_gold_left_299_topk.json
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
    fragment_set,
    load,
    normalize_components,
    substrate_set,
    surviving_routes,
    to_number,
    TEMPERATURE_TOLERANCE_C,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--link1", type=Path, required=True, help="A *_link1_topk.json.")
    parser.add_argument("--test-file", type=Path, required=True, help="The 299-row reference file.")
    parser.add_argument("--routes", type=Path, required=True, help="Model 2 scored on the expansion.")
    parser.add_argument("--gold", type=Path, required=True, help="Model 2 scored on the reference rows.")
    parser.add_argument("--output", type=Path, help="Where to write the counts as JSON.")
    return parser.parse_args()


def read(path: Path) -> tuple[list[dict], str, tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    return payload["records"], summary.get("target_format", "compact"), tuple(summary["condition_fields"])


def field_hits(reference: dict, parsed: dict | None) -> dict[str, tuple[bool, bool]]:
    """(hit, applicable) per field, on the fields the reference actually names."""
    out: dict[str, tuple[bool, bool]] = {}
    for field in ("solvent", "catalyst"):
        ref = normalize_components(reference.get(field))
        out[field] = (
            (normalize_components((parsed or {}).get(field)) == ref, True) if ref is not None else (False, False)
        )
    number = to_number(reference.get("temperature_celsius"))
    if number is None:
        out["temperature_celsius"] = (False, False)
    else:
        predicted = to_number((parsed or {}).get("temperature_celsius"))
        out["temperature_celsius"] = (
            predicted is not None and abs(predicted - number) <= TEMPERATURE_TOLERANCE_C,
            True,
        )
    return out


def main() -> int:
    args = parse_args()
    rows, by_product = load(args.link1, args.test_file)
    route_records, target_format, fields = read(args.routes)
    gold_records, _, _ = read(args.gold)
    if len(gold_records) != len(rows):
        sys.exit("the gold-side file was not scored on the 299 reference rows")

    groups = {True: {"n": 0, "route": 0, "gold": 0}, False: {"n": 0, "route": 0, "gold": 0}}
    totals = {"route": {}, "gold": {}}
    position = 0
    identical = matched = 0

    for index, row in enumerate(rows):
        product = canonical(row["product_smiles"])
        reference = substrate_set(row["full_reactants_smiles"], product)
        gold_fragments = fragment_set(row["full_reactants_smiles"])
        routes = surviving_routes(by_product.get(row["product_smiles"])) or [""]

        hit = None
        for offset, route in enumerate(routes):
            if route != "" and substrate_set(route, product) == reference:
                hit = (position + offset, route)
                break
        position += len(routes)
        if hit is None:
            continue

        matched += 1
        scored_index, route = hit
        same = fragment_set(route) == gold_fragments
        identical += same

        parsed = {
            "route": decode_and_parse(route_records[scored_index]["candidates_raw"][0], target_format, fields),
            "gold": decode_and_parse(gold_records[index]["candidates_raw"][0], target_format, fields),
        }
        groups[same]["n"] += 1
        for source, prediction in parsed.items():
            strict, _ = conditions_match(row, prediction)
            groups[same][source] += strict
            bucket = totals[source]
            bucket.setdefault("full strict", [0, 0])
            bucket["full strict"][0] += strict
            bucket["full strict"][1] += 1
            for field, (ok, applicable) in field_hits(row, prediction).items():
                if not applicable:
                    continue
                bucket.setdefault(field, [0, 0])
                bucket[field][0] += ok
                bucket[field][1] += 1

    print(f"records whose left side link 1 got right: {matched}")
    print(f"  of them, the route is the same molecule set as the reference: {identical} "
          f"({identical / matched * 100:.1f}%)")
    for same, counts in groups.items():
        label = "same molecule set" if same else "same donors, other companions"
        n = counts["n"]
        print(f"  {label:<32}{n:>5} | from the route {counts['route'] / n * 100:>5.1f}% "
              f"| from the reference {counts['gold'] / n * 100:>5.1f}%")
    for source, bucket in totals.items():
        print(f"  rank-1 conditions from the {source} left side:")
        for name, (hits, total) in bucket.items():
            print(f"    {name:<24}{total:>5}{hits / total * 100:>8.1f}%")

    if args.output:
        result = {
            "records_with_a_correct_route": matched,
            "identical_molecule_set": identical,
            "groups": {("same" if key else "donors_only"): value for key, value in groups.items()},
            "totals": {source: {k: v for k, v in bucket.items()} for source, bucket in totals.items()},
        }
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
