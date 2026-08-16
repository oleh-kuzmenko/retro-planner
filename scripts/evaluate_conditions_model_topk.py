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

from evaluate_conditions_model import (
    GROUP_CLASSIFIERS,
    NUMERIC_BUCKET_EDGES,
    TEMPERATURE_TOLERANCE_C,
    YIELD_TOLERANCE_PCT,
    normalize_components,
    same_bucket,
    to_number,
)
from train_conditions_model import (
    COMPACT_MISSING,
    COMPACT_SEPARATOR,
    CONDITION_FIELDS,
    FORMAT_MARKER_FILE,
    TARGET_FORMATS,
    format_input,
    parse_condition_fields,
)

LOGGER = logging.getLogger("retro_eval.evaluate_conditions_model_topk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="Fine-tuned Model 2 checkpoint (local path or HF id).")
    parser.add_argument("--test-file", type=Path, default=Path("data/v2_ord_train/conditions_test.jsonl"))
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Records per generate() call. Mirrors run_reactiont5_topk.py's --batch-size: "
            "batch=1 leaves a GPU mostly idle, raise this (e.g. 16-32) for a real speedup. "
            "CPU runs typically see little benefit and can even slow down from padding "
            "overhead, so leave at 1 there."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--target-format",
        choices=("auto", *TARGET_FORMATS),
        default="auto",
        help="How to parse generations. `auto` reads the marker the trainer wrote next to the "
        "checkpoint, falling back to `json` for checkpoints saved before that marker existed.",
    )
    parser.add_argument(
        "--force-numeric",
        action="store_true",
        help="For numeric fields, rank only the candidates that actually carry a number, so "
        "abstention never occupies a top-k slot. See the comment at the scoring loop.",
    )
    parser.add_argument(
        "--max-source-length",
        type=int,
        default=256,
        help="Must match training. Substrate-only inputs run to 234 tokens at p99 under "
        "CompoundT5's vocabulary, so a shorter cap silently drops reactants from the prompt.",
    )
    parser.add_argument(
        "--condition-fields",
        default=None,
        help="Comma-separated fields the checkpoint emits. Defaults to the list in the format "
        "marker, and to all four for checkpoints saved before the marker carried one.",
    )
    parser.add_argument("--output", type=Path, default=Path("conditions_topk_results.json"))
    return parser.parse_args()


def fix_legacy_tokenizer_config(model_dir: Path) -> None:
    """Patch `extra_special_tokens` from a flat list (how this project's earlier
    training runs saved it) to the `{token: token}` dict form current `transformers`
    expects (`tokenization_utils_base.py`'s `_set_model_specific_special_tokens` calls
    `.keys()` on it). Only touches local checkpoint directories -- HF Hub model ids
    aren't a `Path` that `.is_dir()` would find. In-place: this is a forward-compat
    format fix, not a behavior change (verified via round-trip tokenization: same
    ids before/after for a plain SMILES string), so no need to keep the old file.
    """
    config_path = model_dir / "tokenizer_config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        config["extra_special_tokens"] = {tok: tok for tok in extra}
        config_path.write_text(json.dumps(config), encoding="utf-8")
        LOGGER.info("Patched legacy extra_special_tokens list->dict in %s", config_path)


def fix_tie_word_embeddings(model_dir: Path) -> None:
    """Fix `config.json`'s `tie_word_embeddings` from ground truth (do `lm_head.weight`
    and `shared.weight` actually hold the same values in the saved checkpoint?), rather
    than trusting whatever the field says. Found by direct debugging: for checkpoints
    where the field says `true` but the two tensors are in fact distinct, generation
    silently degenerates into repeating one token -- some `transformers` versions
    auto-detect and override this at load time (with a warning), but this was observed
    NOT to happen reliably (confirmed: identical symptom reproduced and fixed this way
    on a real checkpoint). No-ops (and doesn't load the ~750MB safetensors file) unless
    `tie_word_embeddings` is currently `true`, since a `false` value is never wrong to
    load with two separate tensors present.
    """
    config_path = model_dir / "config.json"
    safetensors_path = model_dir / "model.safetensors"
    if not config_path.is_file() or not safetensors_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("tie_word_embeddings"):
        return

    from safetensors import safe_open

    with safe_open(safetensors_path, framework="pt") as f:
        keys = f.keys()
        if "lm_head.weight" not in keys or "shared.weight" not in keys:
            return
        tied = bool((f.get_tensor("lm_head.weight") == f.get_tensor("shared.weight")).all())

    if not tied:
        config["tie_word_embeddings"] = False
        config_path.write_text(json.dumps(config), encoding="utf-8")
        LOGGER.info(
            "Patched tie_word_embeddings true->false in %s "
            "(lm_head.weight and shared.weight are distinct in the saved checkpoint)",
            config_path,
        )


def decode_and_parse(
    raw: str, target_format: str = "json", fields: tuple[str, ...] = CONDITION_FIELDS
) -> dict | None:
    if target_format == "compact":
        # Spaces are decoding noise, never data: SMILES contain none, and the only
        # spaces in a reference value follow the ", " that joins components, which
        # `normalize_components` collapses anyway. They appear because every token
        # `ensure_full_char_coverage` adds is decoded as its own segment, and
        # SentencePiece re-prefixes the segment that follows it -- `[Pd]` comes back
        # as `[Pd ]`, which RDKit will not parse. `run_reactiont5_topk.py:189` strips
        # them the same way for Model 1.
        parts = raw.replace(" ", "").split(COMPACT_SEPARATOR)
        if len(parts) != len(fields):
            return None
        return {
            field: None if value.strip() in ("", COMPACT_MISSING) else value.strip()
            for field, value in zip(fields, parts)
        }
    for candidate in (raw, "{" + raw + "}"):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def read_format_marker(model_dir: Path) -> dict:
    marker = model_dir / FORMAT_MARKER_FILE
    if marker.is_dir() or not marker.exists():
        return {}
    return json.loads(marker.read_text(encoding="utf-8"))


def resolve_target_format(model_dir: Path, requested: str) -> str:
    """Honour an explicit `--target-format`, else read the marker the trainer wrote."""
    if requested != "auto":
        return requested
    return read_format_marker(model_dir).get("target_format", "json")


def resolve_condition_fields(model_dir: Path, requested: str | None) -> tuple[str, ...]:
    """Which fields the checkpoint was trained to emit, in CONDITION_FIELDS order.

    Checkpoints trained before `--condition-fields` existed carry no `fields` key, and
    those all predicted the full four, so the fallback is CONDITION_FIELDS. Getting this
    wrong is not a soft failure: under `compact` the field count is what splits the
    generation, so a mismatch makes every record unparseable.
    """
    spec = requested or ",".join(read_format_marker(model_dir).get("fields", CONDITION_FIELDS))
    return parse_condition_fields(spec)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from retro_eval.condition_similarity import component_groups

    args = parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_dir = Path(args.model_dir)
    if model_dir.is_dir():
        fix_legacy_tokenizer_config(model_dir)
        fix_tie_word_embeddings(model_dir)

    target_format = resolve_target_format(model_dir, args.target_format)
    fields = resolve_condition_fields(model_dir, args.condition_fields)
    LOGGER.info("Parsing generations as target_format=%s over fields %s", target_format, ", ".join(fields))
    string_fields = tuple(f for f in ("reagents", "solvent", "catalyst") if f in fields)
    numeric_fields = tuple(f for f in ("temperature_celsius", "yield_percent") if f in fields)

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
        for field in string_fields
        for kind in ("exact_match", "same_group")
        for k in ks
    }
    match_expected = {field: 0 for field in string_fields}
    same_group_classifiable = {field: 0 for field in string_fields}

    # Conditional accuracy answers "which catalyst", counted only where the record has one --
    # 941 of 5,687. It says nothing about "is a catalyst needed at all", which the model gets
    # right in 88.9% of records against a 83.5% always-silent baseline. So the record-level
    # counters below score every record: correct silence where the reference is empty, exact
    # match where it is not, and zero when the two disagree about whether the field applies.
    # Report it beside `{field}_absent_share` -- on catalyst the record-level number sits
    # *below* that trivial baseline, and quoting it alone would overstate the system.
    record_level = {f"{field}_top{k}": 0 for field in string_fields for k in ks}
    record_absent = {field: 0 for field in string_fields}
    record_silence_ok = {field: 0 for field in string_fields}

    numeric_within_tol = {f"{field}_top{k}": 0 for field in numeric_fields for k in ks}
    numeric_same_bucket = {f"{field}_top{k}": 0 for field in numeric_fields for k in ks}
    numeric_expected = {field: 0 for field in numeric_fields}

    json_valid_top1 = 0
    detailed = []

    processed = 0
    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start : batch_start + args.batch_size]
        prompts = [format_input(row) for row in batch]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_source_length
        ).to(args.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                max_length=args.max_target_length,
                min_length=1,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for row_idx, row in enumerate(batch):
            candidates_raw = decoded[row_idx * args.num_beams : (row_idx + 1) * args.num_beams]

            seen = set()
            deduped_raw = []
            for candidate in candidates_raw:
                if candidate in seen:
                    continue
                seen.add(candidate)
                deduped_raw.append(candidate)

            parsed_candidates = [decode_and_parse(c, target_format, fields) for c in deduped_raw]
            if parsed_candidates and parsed_candidates[0] is not None:
                json_valid_top1 += 1

            for field in string_fields:
                ref_value = row.get(field)
                ref_set = normalize_components(ref_value)
                # `reagents` has no group taxonomy: solvents and catalysts fall into a few
                # dozen chemical families where a near-miss is still useful to a chemist,
                # but a base, a coupling agent and a halide source share no such axis.
                # It is scored on exact match only, and its same_group counters stay at 0.
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
                    if group_hit_at is None and ref_groups is not None and pred_groups is not None and ref_groups <= pred_groups:
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

            tolerances = {"temperature_celsius": TEMPERATURE_TOLERANCE_C, "yield_percent": YIELD_TOLERANCE_PCT}
            for field in numeric_fields:
                tolerance = tolerances[field]
                ref_num = to_number(row.get(field))
                if ref_num is None:
                    continue
                numeric_expected[field] += 1
                edges = NUMERIC_BUCKET_EDGES[field]
                tol_hit_at = None
                bucket_hit_at = None
                # Under --force-numeric the ranking skips candidates that abstain, so rank 1
                # is the model's best *number* rather than its best answer. Temperature is
                # populated in 27.5% of training rows, which makes `?` the loss-minimizing
                # output: the model abstained on 85.6% of top-1 generations while being
                # accurate when it did commit (median error 0 C). The number was already in
                # the beam list -- forcing the choice lifted top-1 from 6.1% to 27.9% with no
                # retraining. Both readings are real: abstention is the right behaviour when
                # a wrong number is worse than none, forcing is the right one when the user
                # needs a starting point regardless.
                ranked = parsed_candidates[: max(ks)]
                if args.force_numeric:
                    ranked = [p for p in ranked if p is not None and to_number(p.get(field)) is not None]
                for rank, parsed in enumerate(ranked, start=1):
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

            detailed.append(
                {
                    "product_smiles": row["product_smiles"],
                    "reference": row,
                    "candidates_raw": deduped_raw,
                    "candidates_parsed": parsed_candidates,
                }
            )

        processed += len(batch)
        if processed % 25 == 0 or processed == len(rows):
            LOGGER.info("Processed %d/%d", processed, len(rows))

    del model, tokenizer
    gc.collect()

    n = len(rows)
    # json_valid_rate_top1 keeps its name across formats: it is the share of records whose
    # rank-0 generation parsed into the expected number of fields, whatever the serialization.
    summary = {
        "total": n,
        "num_beams": args.num_beams,
        "target_format": target_format,
        "condition_fields": list(fields),
        "force_numeric": bool(args.force_numeric),
        "json_valid_rate_top1": json_valid_top1 / n if n else 0.0,
    }

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
        for k in ks:
            summary[f"{field}_within_tol_top{k}"] = (
                numeric_within_tol[f"{field}_top{k}"] / expected if expected else None
            )
            summary[f"{field}_same_bucket_top{k}"] = (
                numeric_same_bucket[f"{field}_top{k}"] / expected if expected else None
            )

    LOGGER.info("Summary: %s", json.dumps(summary, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": detailed}, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote detailed results to %s", args.output)


if __name__ == "__main__":
    main()
