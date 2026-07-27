#!/usr/bin/env python3
"""Evaluate a fine-tuned Model 2 (reaction-conditions) checkpoint on the held-out test split.

Scores `data/v2_ord_train/conditions_test.jsonl` (built by `build_train_data_ord.py`,
never seen during `train_conditions_model.py` training) against the fine-tuned
checkpoint, reporting:

- json_valid_rate: fraction of generations that parse as JSON with the four
  expected keys (solvent, catalyst, temperature_celsius, yield_percent).
- {field}_present_rate: fraction where the model predicted something other
  than "not specified" for a field whose reference value is also populated
  (i.e. the model attempted *an* answer where one was expected -- this does
  NOT check whether that answer is correct; see below).
- {field}_exact_match_rate / {field}_any_overlap_rate (solvent, catalyst):
  the actual correctness check. Reference and predicted values are split on
  "."/"," into components, each canonicalized as a SMILES string via RDKit
  (falling back to the raw, lowercased text if it doesn't parse as SMILES --
  ORD sometimes stores a compound *name* instead of SMILES), and compared as
  sets: exact_match requires the sets to be identical, any_overlap only
  requires at least one shared component. Earlier iterations of this script
  only reported the "present" rate above, which looked deceptively strong
  (e.g. 99.6% for solvent) purely because the model reliably outputs *some*
  plausible common solvent -- not because it predicts the *correct* one; spot
  checks showed most of those "present" predictions do not actually match
  the reference. Report exact_match/any_overlap, not present_rate, as the
  correctness numbers in the thesis.
- temperature_within_20c_rate / yield_within_15pct_rate: fraction of
  populated-reference rows where the predicted numeric value is within a
  practical tolerance of the reference (both are inherently approximate
  quantities reported inconsistently across ORD source papers). These two
  were always genuine correctness checks, unlike the solvent/catalyst
  "present" rate.

Example:
    python scripts/evaluate_conditions_model.py \\
        --model-dir /content/drive/MyDrive/retro-planner-checkpoints/model2_conditions/final \\
        --test-file data/v2_ord_train/conditions_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from train_conditions_model import CONDITION_FIELDS, format_input

LOGGER = logging.getLogger("retro_eval.evaluate_conditions_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Fine-tuned Model 2 checkpoint (local path or HF id).")
    parser.add_argument("--test-file", type=Path, default=Path("data/v2_ord_train/conditions_test.jsonl"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("conditions_eval_v2.json"))
    return parser.parse_args()


def to_number(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        digits = "".join(ch for ch in str(value) if ch.isdigit() or ch in ".-")
        return float(digits) if digits not in ("", "-", ".") else None
    except ValueError:
        return None


def normalize_components(value) -> frozenset[str] | None:
    """Split a solvent/catalyst value into a set of canonicalized components.

    ORD stores these as SMILES when available, else a plain compound name;
    components are separated inconsistently with "." or ", " across records.
    Each component is canonicalized via RDKit when it parses as SMILES,
    otherwise kept as lowercased/stripped text so at least literal name
    matches (e.g. "water" == "water") still register.
    """
    if value in (None, "", "not specified"):
        return None
    from retro_eval.chemistry import canonicalize_smiles

    parts = [p.strip() for p in str(value).replace(",", ".").split(".") if p.strip()]
    if not parts:
        return None
    normalized = set()
    for part in parts:
        canonical = canonicalize_smiles(part)
        normalized.add(canonical if canonical else part.lower())
    return frozenset(normalized)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(args.device)
    model.eval()

    rows = [json.loads(line) for line in args.test_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    LOGGER.info("Evaluating on %d row(s) from %s", len(rows), args.test_file)

    totals = {f"{field}_present_expected": 0 for field in CONDITION_FIELDS}
    totals.update({f"{field}_present_predicted": 0 for field in CONDITION_FIELDS})
    match_totals = {
        f"{field}_{kind}": 0
        for field in ("solvent", "catalyst")
        for kind in ("expected", "exact_match", "any_overlap")
    }
    json_valid = 0
    temp_within_tol = 0
    temp_expected = 0
    yield_within_tol = 0
    yield_expected = 0
    records = []

    for row in rows:
        prompt = format_input(row)
        inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=256).to(args.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=args.max_target_length, num_beams=4)
        raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # t5-small's default (C4-derived) vocabulary has no single-token
        # encoding for "{"/"}"; both collapse to the same <unk> id, which
        # skip_special_tokens=True strips from the decoded string. Since the
        # training target is always exactly `{...}`, the braces are the very
        # first/last characters generated -- re-wrapping recovers valid JSON
        # without retraining. See diploma methodology notes (Rozdil 3.5).
        parsed = None
        for candidate in (raw, "{" + raw + "}"):
            try:
                parsed = json.loads(candidate)
                json_valid += 1
                break
            except json.JSONDecodeError:
                continue

        record = {"product_smiles": row["product_smiles"], "raw_response": raw, "reference": row}

        if isinstance(parsed, dict):
            for field in CONDITION_FIELDS:
                reference_value = row.get(field)
                predicted_value = parsed.get(field)
                reference_present = reference_value not in (None, "", "not specified")
                predicted_present = predicted_value not in (None, "", "not specified")
                if reference_present:
                    totals[f"{field}_present_expected"] += 1
                    if predicted_present:
                        totals[f"{field}_present_predicted"] += 1

            for field in ("solvent", "catalyst"):
                ref_set = normalize_components(row.get(field))
                if ref_set is None:
                    continue
                match_totals[f"{field}_expected"] += 1
                pred_set = normalize_components(parsed.get(field))
                if pred_set is not None and pred_set == ref_set:
                    match_totals[f"{field}_exact_match"] += 1
                if pred_set is not None and (pred_set & ref_set):
                    match_totals[f"{field}_any_overlap"] += 1

            ref_temp = to_number(row.get("temperature_celsius"))
            pred_temp = to_number(parsed.get("temperature_celsius"))
            if ref_temp is not None:
                temp_expected += 1
                if pred_temp is not None and abs(pred_temp - ref_temp) <= 20:
                    temp_within_tol += 1

            ref_yield = to_number(row.get("yield_percent"))
            pred_yield = to_number(parsed.get("yield_percent"))
            if ref_yield is not None:
                yield_expected += 1
                if pred_yield is not None and abs(pred_yield - ref_yield) <= 15:
                    yield_within_tol += 1

        records.append(record)

    n = len(rows)
    summary = {"total": n, "json_valid": json_valid, "json_valid_rate": json_valid / n if n else 0.0}
    for field in CONDITION_FIELDS:
        expected = totals[f"{field}_present_expected"]
        predicted = totals[f"{field}_present_predicted"]
        summary[f"{field}_present_rate_of_expected"] = predicted / expected if expected else None
        summary[f"{field}_expected_count"] = expected

    summary["temperature_within_20c_rate"] = temp_within_tol / temp_expected if temp_expected else None
    summary["yield_within_15pct_rate"] = yield_within_tol / yield_expected if yield_expected else None

    for field in ("solvent", "catalyst"):
        expected = match_totals[f"{field}_expected"]
        summary[f"{field}_exact_match_rate"] = match_totals[f"{field}_exact_match"] / expected if expected else None
        summary[f"{field}_any_overlap_rate"] = match_totals[f"{field}_any_overlap"] / expected if expected else None
        summary[f"{field}_match_expected_count"] = expected

    LOGGER.info("Summary: %s", json.dumps(summary, indent=2))
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote detailed results to %s", args.output)


if __name__ == "__main__":
    main()
