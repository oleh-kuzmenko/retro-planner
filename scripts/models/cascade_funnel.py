#!/usr/bin/env python3
"""Run the cascade end to end from a stored link-1 output, and count the funnel.

The measurement in RESULTS.md ("F6/F6b") lived inside a Kaggle notebook, so re-running it
against a different Model 2 meant re-running the notebook. Link 1 is fixed, though, and its
generations are already on disk: `F6_link1_topk.json` holds ten candidates for each of the 299
held-out ORD records. Everything after that is deterministic bookkeeping plus one Model 2
inference, which for a classifier is a CPU matter of minutes.

Two subcommands, in the order the notebook ran them:

- `expand` turns each record into up to five rows, one per surviving route: candidates are
  canonicalized, deduplicated by fragment set, and dropped when RDKit cannot read them. A
  record with no usable route keeps one row with an empty left side -- it has failed, not
  vanished, and dropping it would quietly shrink the denominator.
- `funnel` reads a Model 2 result file scored on those rows and counts, per record, whether
  link 1 found the left side, and whether *one and the same* proposal carries both the right
  left side and the right conditions. That last condition is the point of the subsection: the
  two links are not scored independently, because a chemist runs one proposal, not a pairing
  of the best halves of two.

Link 1 is scored by the role rule of `build_conditions_roles.py` -- a predicted set matches
when it names the same molecules that donate at least one heavy atom to the product, which is
the convention both links were trained and evaluated under on ORD.

Example:
    python scripts/models/cascade_funnel.py expand \\
        --link1 experiments/v2_model2_roles/F6_link1_topk.json \\
        --test-file data/v2_ord_cascade_test_clean299.jsonl \\
        --output data/v2_ord_cascade_expanded299.jsonl
    python scripts/models/cascade_funnel.py funnel \\
        --link1 experiments/v2_model2_roles/F6_link1_topk.json \\
        --test-file data/v2_ord_cascade_test_clean299.jsonl \\
        --link2 experiments/v2_model2_roles/E1_cascade_routes_topk.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_conditions_model import (
    GROUP_CLASSIFIERS,
    NUMERIC_BUCKET_EDGES,
    TEMPERATURE_TOLERANCE_C,
    normalize_components,
    same_bucket,
    to_number,
)
from evaluate_conditions_model_topk import decode_and_parse
from substrate_role_analysis import canonical, is_substrate

MAX_ROUTES = 5
SUBSTANCE_FIELDS = ("solvent", "catalyst")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--link1", type=Path, required=True, help="A *_link1_topk.json.")
    common.add_argument("--test-file", type=Path, required=True, help="The 299-row reference file.")

    expand = sub.add_parser("expand", parents=[common], help="Write one row per surviving route.")
    expand.add_argument("--output", type=Path, required=True)

    funnel = sub.add_parser("funnel", parents=[common], help="Count the funnel.")
    funnel.add_argument("--link2", type=Path, required=True, help="Model 2 scored on the expanded rows.")
    funnel.add_argument("--output", type=Path, help="Where to write the counts as JSON.")
    return parser.parse_args()


def fragment_set(smiles: str) -> tuple[str, ...] | None:
    """Canonical fragment set, or None when RDKit cannot read a fragment -- that is the filter."""
    fragments = []
    for part in smiles.split("."):
        if not part:
            continue
        canon = canonical(part)
        if canon is None:
            return None
        fragments.append(canon)
    return tuple(sorted(fragments)) if fragments else None


def substrate_set(smiles: str, product: str) -> frozenset[str] | None:
    """The molecules of a left side that donate at least one heavy atom to the product."""
    fragments = fragment_set(smiles)
    if fragments is None:
        return None
    return frozenset(f for f in fragments if is_substrate(f, product))


def surviving_routes(record: dict | None) -> list[str]:
    routes, seen = [], set()
    for candidate in (record["candidates"] if record else []):
        key = fragment_set(candidate)
        if key is None or key in seen:
            continue
        seen.add(key)
        routes.append(".".join(key))
        if len(routes) == MAX_ROUTES:
            break
    return routes


def load(link1: Path, test_file: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in test_file.open(encoding="utf-8")]
    records = json.loads(link1.read_text(encoding="utf-8"))["records"]
    return rows, {r["product_smiles"]: r for r in records}


def do_expand(args: argparse.Namespace) -> None:
    rows, by_product = load(args.link1, args.test_file)
    expanded, counts, dropped = [], [], 0
    for row in rows:
        routes = surviving_routes(by_product.get(row["product_smiles"]))
        if not routes:
            dropped += 1
            routes = [""]
        counts.append(len(routes))
        for index, route in enumerate(routes):
            out = dict(row)
            out["full_reactants_smiles"] = route
            out["route_index"] = index
            expanded.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in expanded:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"{len(expanded)} rows from {len(rows)} records | routes per record "
        f"{sum(counts) / len(counts):.2f} | records with no usable route {dropped}"
    )
    print(f"written to {args.output}")


def conditions_match(reference: dict, parsed: dict | None) -> tuple[bool, bool]:
    """(strict, relaxed) over the fields the reference actually names.

    A record the source left blank cannot be got wrong, so it does not enter either verdict;
    a record that names nothing at all counts as matched, which is the same convention the
    per-field metrics use when their denominator excludes the row.
    """
    from retro_eval.condition_similarity import component_groups

    strict, relaxed = True, True
    for field in SUBSTANCE_FIELDS:
        ref_set = normalize_components(reference.get(field))
        if ref_set is None:
            continue
        pred_value = parsed.get(field) if parsed else None
        strict &= normalize_components(pred_value) == ref_set
        classifier = GROUP_CLASSIFIERS.get(field)
        ref_groups = component_groups(reference.get(field), classifier) if classifier else None
        pred_groups = component_groups(pred_value, classifier) if classifier else None
        relaxed &= bool(ref_groups is not None and pred_groups is not None and ref_groups <= pred_groups)

    ref_number = to_number(reference.get("temperature_celsius"))
    if ref_number is not None:
        pred_number = to_number(parsed.get("temperature_celsius")) if parsed else None
        strict &= pred_number is not None and abs(pred_number - ref_number) <= TEMPERATURE_TOLERANCE_C
        relaxed &= pred_number is not None and same_bucket(
            pred_number, ref_number, NUMERIC_BUCKET_EDGES["temperature_celsius"]
        )
    return strict, relaxed


def do_funnel(args: argparse.Namespace) -> None:
    rows, by_product = load(args.link1, args.test_file)
    link2 = json.loads(args.link2.read_text(encoding="utf-8"))
    summary = link2["summary"]
    target_format = summary.get("target_format", "compact")
    fields = tuple(summary["condition_fields"])
    scored = link2["records"]

    if len(scored) != sum(max(1, len(surviving_routes(by_product.get(r["product_smiles"])))) for r in rows):
        sys.exit("the link-2 file was not scored on this expansion: row counts disagree")

    position = 0
    link1_hits = end_to_end_strict = end_to_end_relaxed = 0
    for row in rows:
        product = canonical(row["product_smiles"])
        reference = substrate_set(row["full_reactants_smiles"], product)
        routes = surviving_routes(by_product.get(row["product_smiles"])) or [""]

        record_hit = record_strict = record_relaxed = False
        for route in routes:
            candidate = scored[position]
            position += 1
            route_ok = route != "" and substrate_set(route, product) == reference
            record_hit |= route_ok
            if not route_ok:
                continue
            # Rank 1 only: the cascade offers one condition set per route, and crediting a
            # lower-ranked one would score a list the chemist was never shown.
            parsed = decode_and_parse(candidate["candidates_raw"][0], target_format, fields)
            strict, relaxed = conditions_match(row, parsed)
            record_strict |= strict
            record_relaxed |= relaxed

        link1_hits += record_hit
        end_to_end_strict += record_strict
        end_to_end_relaxed += record_relaxed

    total = len(rows)
    result = {
        "records": total,
        "link1_top5": link1_hits,
        "link1_top5_share": link1_hits / total,
        "end_to_end_strict": end_to_end_strict,
        "end_to_end_strict_share": end_to_end_strict / total,
        "end_to_end_relaxed": end_to_end_relaxed,
        "end_to_end_relaxed_share": end_to_end_relaxed / total,
        "link2_file": str(args.link2),
    }
    print(f"records {total}")
    print(f"  link 1 found the left side (top-5): {link1_hits} ({link1_hits / total * 100:.1f}%)")
    print(f"  one proposal right end to end, strict: {end_to_end_strict} ({end_to_end_strict / total * 100:.1f}%)")
    print(f"  the same, relaxed: {end_to_end_relaxed} ({end_to_end_relaxed / total * 100:.1f}%)")
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written to {args.output}")


def main() -> None:
    args = parse_args()
    (do_expand if args.command == "expand" else do_funnel)(args)


if __name__ == "__main__":
    main()
