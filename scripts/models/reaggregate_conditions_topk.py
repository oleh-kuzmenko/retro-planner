#!/usr/bin/env python3
"""Recompute a Model 2 top-k summary from the generations already stored in the file.

`evaluate_conditions_model_topk.py` grew its record-level scales (`{field}_absent_share`,
`{field}_silence_correct`, `{field}_record_level_top{k}`) after the B2 runs were scored, so
their summaries carry only the conditional metrics. The generations themselves are in the
file -- `records[*].candidates_raw`, already deduplicated and in rank order -- so the missing
scales are a re-parse away and need neither a checkpoint nor a GPU.

The recomputation covers *every* metric, not just the missing ones, and the script refuses to
write a file whose recomputed values disagree with the stored ones. That check is the point:
it proves the offline parse reproduces what the evaluator did on the model's own output, and
only then are the newly added scales trustworthy.

Adds one metric the evaluator does not report: `{field}_abstain_share_top1` for numeric
fields -- the share of *all* test records whose rank-1 generation names no number. On a corpus
where the field is sparsely annotated this is the number that shows the model learned to stay
silent, which the conditional accuracy hides.

Example:
    python scripts/models/reaggregate_conditions_topk.py \\
        experiments/v2_model2_roles/B2_conditions_t5small_3f_clean_topk.json --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_conditions_model import (
    GROUP_CLASSIFIERS,
    NUMERIC_BUCKET_EDGES,
    TEMPERATURE_TOLERANCE_C,
    YIELD_TOLERANCE_PCT,
    normalize_components,
    same_bucket,
    to_number,
)
from evaluate_conditions_model_topk import decode_and_parse

STRING_FIELD_ORDER = ("reagents", "solvent", "catalyst")
NUMERIC_FIELD_ORDER = ("temperature_celsius", "yield_percent")
TOLERANCES = {
    "temperature_celsius": TEMPERATURE_TOLERANCE_C,
    "yield_percent": YIELD_TOLERANCE_PCT,
}
# Rounding in the stored file is float64 division of integer counters, so an exact match is
# expected; the slack only absorbs JSON round-tripping.
EPSILON = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path, help="A *_topk.json written by evaluate_conditions_model_topk.py")
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the file with the recomputed summary. Omit to only report.",
    )
    parser.add_argument("--in-place", action="store_true", help="Rewrite the input file itself.")
    return parser.parse_args()


def recompute(data: dict) -> dict:
    from retro_eval.condition_similarity import component_groups

    stored = data["summary"]
    target_format = stored.get("target_format", "json")
    fields = tuple(stored["condition_fields"])
    records = data["records"]
    n = len(records)

    string_fields = tuple(f for f in STRING_FIELD_ORDER if f in fields)
    numeric_fields = tuple(f for f in NUMERIC_FIELD_ORDER if f in fields)
    ks = [k for k in (1, 3, 5) if k <= stored["num_beams"]]

    match_totals = {
        f"{field}_{kind}_top{k}": 0
        for field in string_fields
        for kind in ("exact_match", "same_group")
        for k in ks
    }
    match_expected = {field: 0 for field in string_fields}
    same_group_classifiable = {field: 0 for field in string_fields}
    record_level = {f"{field}_top{k}": 0 for field in string_fields for k in ks}
    record_absent = {field: 0 for field in string_fields}
    record_silence_ok = {field: 0 for field in string_fields}

    numeric_within_tol = {f"{field}_top{k}": 0 for field in numeric_fields for k in ks}
    numeric_same_bucket = {f"{field}_top{k}": 0 for field in numeric_fields for k in ks}
    numeric_expected = {field: 0 for field in numeric_fields}
    numeric_abstain_top1 = {field: 0 for field in numeric_fields}

    json_valid_top1 = 0

    for record in records:
        row = record["reference"]
        parsed_candidates = [
            decode_and_parse(candidate, target_format, fields) for candidate in record["candidates_raw"]
        ]
        if parsed_candidates and parsed_candidates[0] is not None:
            json_valid_top1 += 1

        for field in string_fields:
            ref_value = row.get(field)
            ref_set = normalize_components(ref_value)
            classifier = GROUP_CLASSIFIERS.get(field)
            ref_groups = component_groups(ref_value, classifier) if classifier else None

            if ref_set is not None:
                match_expected[field] += 1
            if ref_groups is not None:
                same_group_classifiable[field] += 1

            exact_hit_at = None
            group_hit_at = None
            for rank, parsed in enumerate(parsed_candidates[: max(ks)], start=1):
                if parsed is None:
                    continue
                pred_set = normalize_components(parsed.get(field))
                if exact_hit_at is None and ref_set is not None and pred_set is not None and pred_set == ref_set:
                    exact_hit_at = rank
                pred_groups = component_groups(parsed.get(field), classifier) if classifier else None
                if (
                    group_hit_at is None
                    and ref_groups is not None
                    and pred_groups is not None
                    and ref_groups <= pred_groups
                ):
                    group_hit_at = rank

            for k in ks:
                if ref_set is not None and exact_hit_at is not None and exact_hit_at <= k:
                    match_totals[f"{field}_exact_match_top{k}"] += 1
                if ref_groups is not None and group_hit_at is not None and group_hit_at <= k:
                    match_totals[f"{field}_same_group_top{k}"] += 1

            if ref_set is None:
                record_absent[field] += 1
                top1 = parsed_candidates[0] if parsed_candidates else None
                if top1 is not None and normalize_components(top1.get(field)) is None:
                    record_silence_ok[field] += 1
                    for k in ks:
                        record_level[f"{field}_top{k}"] += 1
            else:
                for k in ks:
                    if exact_hit_at is not None and exact_hit_at <= k:
                        record_level[f"{field}_top{k}"] += 1

        for field in numeric_fields:
            top1 = parsed_candidates[0] if parsed_candidates else None
            if top1 is None or to_number(top1.get(field)) is None:
                numeric_abstain_top1[field] += 1

            ref_num = to_number(row.get(field))
            if ref_num is None:
                continue
            numeric_expected[field] += 1
            tolerance = TOLERANCES[field]
            edges = NUMERIC_BUCKET_EDGES[field]
            tol_hit_at = None
            bucket_hit_at = None
            for rank, parsed in enumerate(parsed_candidates[: max(ks)], start=1):
                if parsed is None:
                    continue
                pred_num = to_number(parsed.get(field))
                if pred_num is None:
                    continue
                if tol_hit_at is None and abs(pred_num - ref_num) <= tolerance:
                    tol_hit_at = rank
                if bucket_hit_at is None and same_bucket(pred_num, ref_num, edges):
                    bucket_hit_at = rank
            for k in ks:
                if tol_hit_at is not None and tol_hit_at <= k:
                    numeric_within_tol[f"{field}_top{k}"] += 1
                if bucket_hit_at is not None and bucket_hit_at <= k:
                    numeric_same_bucket[f"{field}_top{k}"] += 1

    summary = {
        "total": n,
        "num_beams": stored["num_beams"],
        "target_format": target_format,
        "condition_fields": list(fields),
    }
    for carried in ("reactants_field", "force_numeric"):
        if carried in stored:
            summary[carried] = stored[carried]
    summary["json_valid_rate_top1"] = json_valid_top1 / n if n else 0.0

    for field in string_fields:
        expected = match_expected[field]
        classifiable = same_group_classifiable[field]
        summary[f"{field}_expected_count"] = expected
        summary[f"{field}_same_group_classifiable_count"] = classifiable
        summary[f"{field}_absent_share"] = record_absent[field] / n if n else None
        summary[f"{field}_silence_correct"] = (
            record_silence_ok[field] / record_absent[field] if record_absent[field] else None
        )
        for k in ks:
            summary[f"{field}_record_level_top{k}"] = record_level[f"{field}_top{k}"] / n if n else None
        for k in ks:
            summary[f"{field}_exact_match_top{k}"] = (
                match_totals[f"{field}_exact_match_top{k}"] / expected if expected else None
            )
            summary[f"{field}_same_group_top{k}"] = (
                match_totals[f"{field}_same_group_top{k}"] / classifiable if classifiable else None
            )

    for field in numeric_fields:
        expected = numeric_expected[field]
        summary[f"{field}_expected_count"] = expected
        summary[f"{field}_abstain_share_top1"] = numeric_abstain_top1[field] / n if n else None
        for k in ks:
            summary[f"{field}_within_tol_top{k}"] = (
                numeric_within_tol[f"{field}_top{k}"] / expected if expected else None
            )
            summary[f"{field}_same_bucket_top{k}"] = (
                numeric_same_bucket[f"{field}_top{k}"] / expected if expected else None
            )

    return summary


def report(stored: dict, recomputed: dict) -> list[str]:
    """Keys the stored summary already carried, and where the two disagree."""
    disagreements = []
    for key, old in stored.items():
        if key not in recomputed:
            continue
        new = recomputed[key]
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            if abs(old - new) > EPSILON:
                disagreements.append(f"{key}: stored {old!r} vs recomputed {new!r}")
        elif old != new:
            disagreements.append(f"{key}: stored {old!r} vs recomputed {new!r}")
    return disagreements


def main() -> None:
    args = parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    stored = data["summary"]
    recomputed = recompute(data)

    disagreements = report(stored, recomputed)
    added = [key for key in recomputed if key not in stored]

    print(f"{args.results.name}: {len(data['records'])} records, {len(stored)} stored keys")
    if disagreements:
        print("MISMATCH -- the offline parse does not reproduce the stored numbers:")
        for line in disagreements:
            print(f"  {line}")
        sys.exit(1)
    print(f"  all {len(stored) - len(added) if added else len(stored)} shared keys reproduced exactly")
    if added:
        print(f"  {len(added)} scale(s) added:")
        for key in added:
            value = recomputed[key]
            shown = f"{value:.4f}" if isinstance(value, float) else value
            print(f"    {key} = {shown}")

    destination = args.results if args.in_place else args.output
    if destination is None:
        return
    data["summary"] = recomputed
    destination.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"  written to {destination}")


if __name__ == "__main__":
    main()
