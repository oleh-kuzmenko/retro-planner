#!/usr/bin/env python3
"""Evaluate a fine-tuned Model 2 (reaction-conditions) checkpoint on the held-out test split.

Scores `data/v2_ord_train/conditions_test.jsonl` (built by `build_train_data_ord.py`,
never seen during `train_conditions_model.py` training) against the fine-tuned
checkpoint, reporting:

- json_valid_rate: fraction of generations that parse as JSON with the four
  expected keys (solvent, catalyst, temperature_celsius, yield_percent).
- {field}_present_rate: fraction where the model predicted something other
  than "not specified" for a field whose reference value is also populated
  (i.e. the model attempted an answer where one was expected).
- temperature_within_20c_rate / yield_within_15pct_rate: fraction of
  populated-reference rows where the predicted numeric value is within a
  practical tolerance of the reference (both are inherently approximate
  quantities reported inconsistently across ORD source papers).

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

        parsed = None
        try:
            parsed = json.loads(raw)
            json_valid += 1
        except json.JSONDecodeError:
            pass

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

    LOGGER.info("Summary: %s", json.dumps(summary, indent=2))
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote detailed results to %s", args.output)


if __name__ == "__main__":
    main()
