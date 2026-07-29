#!/usr/bin/env python3
"""Evaluate a fine-tuned Model 2 (reaction-conditions) checkpoint with top-1/3/5 accuracy.

`evaluate_conditions_model.py` only keeps the single best (greedy/1-beam)
generation per record. Mirroring `run_reactiont5_topk.py`'s rationale for
Model 1: a chemist reviewing a short list of candidate condition sets is a
realistic usage pattern, and generating `--num-beams` candidates per record
via beam search typically raises the hit rate over top-1 alone. This script
reports, for k=1,3,5 (capped at --num-beams), whether *any* of the top-k
candidates matches the reference on:

- solvent / catalyst: exact_match_topk (canonicalized component-set equality,
  same normalization as evaluate_conditions_model.py) and
  same_group_topk (chemistry-informed relaxation via retro_eval.condition_similarity).
- temperature_celsius / yield_percent: within_tol_topk (|predicted - reference|
  within the same tolerances as the single-shot script: 10C / 10 percentage points).

Candidates are deduplicated by their exact decoded JSON text before ranking,
so k reflects k distinct generations, not k beams that happen to decode
identically. json_valid_rate_top1 reports the single-shot JSON-validity rate
for reference (should match evaluate_conditions_model.py's json_valid_rate
on the same checkpoint/data, since beam candidate 0 there is a 4-beam best
generation vs this script's rank-0 of num_beams -- close but not required to
be numerically identical).

Example:
    python scripts/evaluate_conditions_model_topk.py \\
        --model-dir /content/drive/MyDrive/retro-planner-checkpoints/model2_conditions/final \\
        --test-file data/v2_ord_train/conditions_test.jsonl \\
        --num-beams 10 --output conditions_topk_results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_conditions_model import GROUP_CLASSIFIERS, TEMPERATURE_TOLERANCE_C, YIELD_TOLERANCE_PCT, normalize_components, to_number
from train_conditions_model import CONDITION_FIELDS, format_input

LOGGER = logging.getLogger("retro_eval.evaluate_conditions_model_topk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="Fine-tuned Model 2 checkpoint (local path or HF id).")
    parser.add_argument("--test-file", type=Path, default=Path("data/v2_ord_train/conditions_test.jsonl"))
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("conditions_topk_results.json"))
    return parser.parse_args()


def decode_and_parse(raw: str) -> dict | None:
    for candidate in (raw, "{" + raw + "}"):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from retro_eval.condition_similarity import component_groups

    args = parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows = [json.loads(line) for line in args.test_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    LOGGER.info("Evaluating on %d row(s) from %s", len(rows), args.test_file)

    LOGGER.info("Loading model=%s on device=%s ...", args.model_dir, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(args.device)
    model.eval()

    ks = [k for k in (1, 3, 5) if k <= args.num_beams]

    match_totals = {
        f"{field}_{kind}_top{k}": 0
        for field in ("solvent", "catalyst")
        for kind in ("exact_match", "same_group")
        for k in ks
    }
    match_expected = {field: 0 for field in ("solvent", "catalyst")}
    same_group_classifiable = {field: 0 for field in ("solvent", "catalyst")}

    numeric_within_tol = {f"{field}_top{k}": 0 for field in ("temperature_celsius", "yield_percent") for k in ks}
    numeric_expected = {field: 0 for field in ("temperature_celsius", "yield_percent")}

    json_valid_top1 = 0
    detailed = []

    for i, row in enumerate(rows):
        prompt = format_input(row)
        inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=256).to(args.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                max_length=args.max_target_length,
                min_length=1,
            )
        candidates_raw = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        seen = set()
        deduped_raw = []
        for candidate in candidates_raw:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped_raw.append(candidate)

        parsed_candidates = [decode_and_parse(c) for c in deduped_raw]
        if parsed_candidates and parsed_candidates[0] is not None:
            json_valid_top1 += 1

        for field in ("solvent", "catalyst"):
            ref_value = row.get(field)
            ref_set = normalize_components(ref_value)
            classifier = GROUP_CLASSIFIERS[field]
            ref_groups = component_groups(ref_value, classifier)

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
                pred_groups = component_groups(parsed.get(field), classifier)
                if group_hit_at is None and ref_groups is not None and pred_groups is not None and ref_groups <= pred_groups:
                    group_hit_at = rank

            for k in ks:
                if ref_set is not None and exact_hit_at is not None and exact_hit_at <= k:
                    match_totals[f"{field}_exact_match_top{k}"] += 1
                if ref_groups is not None and group_hit_at is not None and group_hit_at <= k:
                    match_totals[f"{field}_same_group_top{k}"] += 1

        for field, tolerance in (("temperature_celsius", TEMPERATURE_TOLERANCE_C), ("yield_percent", YIELD_TOLERANCE_PCT)):
            ref_num = to_number(row.get(field))
            if ref_num is None:
                continue
            numeric_expected[field] += 1
            hit_at = None
            for rank, parsed in enumerate(parsed_candidates[: max(ks)], start=1):
                if parsed is None:
                    continue
                pred_num = to_number(parsed.get(field))
                if pred_num is not None and abs(pred_num - ref_num) <= tolerance:
                    hit_at = rank
                    break
            for k in ks:
                if hit_at is not None and hit_at <= k:
                    numeric_within_tol[f"{field}_top{k}"] += 1

        detailed.append(
            {
                "product_smiles": row["product_smiles"],
                "reference": row,
                "candidates_raw": deduped_raw,
                "candidates_parsed": parsed_candidates,
            }
        )
        if (i + 1) % 25 == 0:
            LOGGER.info("Processed %d/%d", i + 1, len(rows))

    del model, tokenizer
    gc.collect()

    n = len(rows)
    summary = {"total": n, "num_beams": args.num_beams, "json_valid_rate_top1": json_valid_top1 / n if n else 0.0}

    for field in ("solvent", "catalyst"):
        expected = match_expected[field]
        classifiable = same_group_classifiable[field]
        summary[f"{field}_expected_count"] = expected
        summary[f"{field}_same_group_classifiable_count"] = classifiable
        for k in ks:
            summary[f"{field}_exact_match_top{k}"] = (
                match_totals[f"{field}_exact_match_top{k}"] / expected if expected else None
            )
            summary[f"{field}_same_group_top{k}"] = (
                match_totals[f"{field}_same_group_top{k}"] / classifiable if classifiable else None
            )

    for field in ("temperature_celsius", "yield_percent"):
        expected = numeric_expected[field]
        summary[f"{field}_expected_count"] = expected
        for k in ks:
            summary[f"{field}_within_tol_top{k}"] = (
                numeric_within_tol[f"{field}_top{k}"] / expected if expected else None
            )

    LOGGER.info("Summary: %s", json.dumps(summary, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": detailed}, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote detailed results to %s", args.output)


if __name__ == "__main__":
    main()
