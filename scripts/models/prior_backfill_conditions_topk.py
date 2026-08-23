#!/usr/bin/env python3
"""Rebuild a Model 2 candidate list field by field, backfilled from the corpus frequency prior.

Beam search over a joint condition sequence spends its slots badly: the beams of a Model 2 run
differ in how a formula is written far more often than in which substance is named, so a
5-beam run offers only 2.84 distinct solvents and 1.18 distinct catalysts per record. The
strict top-5 pays for that -- it sits below what naming the five most frequent sets of the
training corpus would score without reading the reaction at all.

This script measures how much of that gap is the list rather than the model, and it needs
neither a checkpoint nor a GPU: `records[*].candidates_raw` already holds the generations.
Per field it keeps the distinct values in rank order and backfills the empty slots with the
most frequent sets of the training corpus. That is what a classification head gives for free
-- a softmax cannot spend two of its five slots on one substance -- so the result is a lower
bound on what the encoder-with-heads reformulation could win.

**Rank 1 is never touched.** Slot 1 stays the model's own generation, verbatim, which keeps
`json_valid_rate_top1`, the silence scales and every top-1 metric identical to the source run;
only slots 2..k are rebuilt. `--prior-topn 0` disables the backfill and leaves a pure per-field
dedup, which is the self-check: top-1 and top-5 must then reproduce the stored summary to the
digit, since dropping a duplicate from a five-long list changes neither what rank 1 is nor
which values the five slots hold. Top-3 is the one scale dedup may move, and only upward -- a
hit that sat at rank 4 behind a repeated value moves up when the repeat goes. The script
refuses to write if a top-1 or top-5 number shifts, or if a top-3 number falls.

The output keeps the shape of an `evaluate_conditions_model_topk.py` file (slot i joins the
i-th value of each field back into one target string, which is sound because the fields are
scored independently), so `mcnemar_conditions.py` and `reaggregate_conditions_topk.py` read it
unchanged. The candidates in it are assembled, not generated: never feed such a file to the
cascade demo.

Example:
    python scripts/models/prior_backfill_conditions_topk.py \\
        experiments/v2_model2_roles/F1_conditions_full486k_clean_topk.json \\
        --train-file data/v2_ord_roles_full/conditions_train.jsonl \\
        --output experiments/v2_model2_roles/F1_priorfill_topk.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_conditions_model import normalize_components, to_number
from evaluate_conditions_model_topk import decode_and_parse
from train_conditions_model import format_target
from reaggregate_conditions_topk import EPSILON, recompute

NUMERIC_FIELDS = ("temperature_celsius", "yield_percent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path, help="A *_topk.json written by evaluate_conditions_model_topk.py")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=ROOT / "data/v2_ord_roles_full/conditions_train.jsonl",
        help="Corpus the frequency prior is counted on (the run's own training file).",
    )
    parser.add_argument(
        "--fields",
        default="solvent,catalyst",
        help="Fields to backfill, comma-separated. Others are only deduplicated.",
    )
    parser.add_argument("--k", type=int, default=5, help="Length of the rebuilt candidate list.")
    parser.add_argument(
        "--prior-topn",
        type=int,
        default=5,
        help="How many prior entries may be used per field. 0 disables the backfill and turns "
        "the run into the self-check.",
    )
    parser.add_argument(
        "--prior-only",
        action="store_true",
        help="Ignore the generations entirely and score the frequency prior alone (the honest "
        "top-k trivial baseline). Skips the self-check.",
    )
    parser.add_argument("--output", type=Path, help="Where to write the rebuilt file.")
    return parser.parse_args()


def field_key(field: str, value):
    """The identity two candidate values are considered the same under -- the metric's own."""
    if field in NUMERIC_FIELDS:
        return to_number(value)
    return normalize_components(value)


def build_prior(train_file: Path, fields: tuple[str, ...], top_n: int) -> dict[str, list[str]]:
    """Most frequent value of each field in the corpus, as text a candidate may carry."""
    counters = {field: collections.Counter() for field in fields}
    representatives: dict[str, dict] = {field: {} for field in fields}
    with train_file.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for field in fields:
                key = field_key(field, row.get(field))
                if key is None:
                    continue
                counters[field][key] += 1
                representatives[field].setdefault(key, row.get(field))
    prior = {}
    for field in fields:
        prior[field] = [representatives[field][key] for key, _ in counters[field].most_common(top_n)]
    return prior


def rebuild_field(field: str, parsed: list, prior: list[str], k: int, prior_only: bool) -> list:
    """Distinct values in rank order, backfilled from the prior, length k."""
    values: list = []
    keys: list = []
    if not prior_only:
        for index, candidate in enumerate(parsed):
            value = candidate.get(field) if candidate is not None else None
            key = field_key(field, value)
            # Rank 1 is kept whatever it is, silence included; later slots must be new.
            if index == 0:
                values.append(value)
                keys.append(key)
                continue
            if key is None or key in keys:
                continue
            values.append(value)
            keys.append(key)
    for value in prior:
        if len(values) >= k:
            break
        key = field_key(field, value)
        if key in keys:
            continue
        values.append(value)
        keys.append(key)
    if not values:
        return [None] * k
    # Slots past the end repeat the last value: a repeat can never add a hit, so it leaves
    # every metric where the shorter list left it.
    return (values + [values[-1]] * k)[:k]


def rebuild(data: dict, prior: dict[str, list[str]], k: int, prior_only: bool) -> None:
    stored = data["summary"]
    target_format = stored.get("target_format", "json")
    fields = tuple(stored["condition_fields"])
    for record in data["records"]:
        raw_candidates = record["candidates_raw"]
        parsed = [decode_and_parse(raw, target_format, fields) for raw in raw_candidates]
        columns = {
            field: rebuild_field(field, parsed, prior.get(field, []), k, prior_only)
            for field in fields
        }
        rebuilt = [
            format_target({field: columns[field][slot] for field in fields}, target_format, fields)
            for slot in range(k)
        ]
        if not prior_only and raw_candidates:
            rebuilt[0] = raw_candidates[0]
        record["candidates_raw"] = rebuilt


def check_dedup(stored: dict, recomputed: dict) -> tuple[list[str], list[str]]:
    """Split the dedup-only differences into the forbidden ones and the expected ones.

    Forbidden: any shift at top-1 or top-5, or in a count, share or validity scale. Expected:
    a top-3 number that rose, which is dedup doing its job -- it cannot fall, because removing
    a duplicate only ever moves a later distinct value forward.
    """
    broken: list[str] = []
    promoted: list[str] = []
    for key, old in stored.items():
        if key not in recomputed:
            continue
        new = recomputed[key]
        if not (isinstance(old, (int, float)) and isinstance(new, (int, float))):
            if old != new:
                broken.append(f"{key}: stored {old!r} vs recomputed {new!r}")
            continue
        if abs(old - new) <= EPSILON:
            continue
        if key.endswith("_top3"):
            if new < old:
                broken.append(f"{key}: fell, stored {old!r} vs recomputed {new!r}")
            else:
                promoted.append(f"{key}: {old * 100:.1f} -> {new * 100:.1f}%")
            continue
        broken.append(f"{key}: stored {old!r} vs recomputed {new!r}")
    return broken, promoted


def main() -> None:
    args = parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    stored = dict(data["summary"])
    fields = tuple(stored["condition_fields"])

    requested = tuple(name.strip() for name in args.fields.split(",") if name.strip())
    unknown = [name for name in requested if name not in fields]
    if unknown:
        sys.exit(f"--fields names {unknown}, which this run does not carry: {list(fields)}")

    top_n = 0 if (args.prior_topn <= 0 and not args.prior_only) else max(args.prior_topn, args.k)
    prior = build_prior(args.train_file, requested, top_n) if top_n else {}

    rebuild(data, prior, args.k, args.prior_only)
    recomputed = recompute(data)

    self_check = args.prior_topn <= 0 and not args.prior_only
    if self_check:
        broken, promoted = check_dedup(stored, recomputed)
        if broken:
            print("MISMATCH -- the per-field dedup does not reproduce the stored numbers:")
            for line in broken:
                print(f"  {line}")
            sys.exit(1)
        print(f"{args.results.name}: dedup-only rebuild reproduces every top-1 and top-5 number")
        for line in promoted:
            print(f"  top-3 promoted by dedup, {line}")

    recomputed["candidate_strategy"] = (
        "prior_only" if args.prior_only else ("per_field_dedup" if not top_n else "per_field_dedup_prior_backfill")
    )
    recomputed["prior_topn"] = top_n
    recomputed["prior_fields"] = list(requested)
    recomputed["prior_source"] = str(args.train_file)
    recomputed["candidates_assembled"] = True

    print(f"{args.results.name} -> {recomputed['candidate_strategy']} (k={args.k})")
    for field in fields:
        for key in (f"{field}_exact_match_top5", f"{field}_same_group_top5", f"{field}_within_tol_top5"):
            if key in recomputed and recomputed[key] is not None:
                was = stored.get(key)
                shown = f"{was * 100:.1f} -> " if isinstance(was, float) else ""
                print(f"  {key}: {shown}{recomputed[key] * 100:.1f}%")

    if args.output is None:
        return
    data["summary"] = recomputed
    args.output.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"  written to {args.output}")


if __name__ == "__main__":
    main()
