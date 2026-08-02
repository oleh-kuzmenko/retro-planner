#!/usr/bin/env python3
"""Evaluate a ReactionT5-family checkpoint with top-1/top-3/top-5 (not just top-1) accuracy.

Most published
retrosynthesis results report top-k accuracy (a prediction counts as correct
if the reference appears anywhere in the top k beam candidates) precisely
because a chemist reviewing a short list of candidate routes is a realistic
usage pattern, and top-k is typically far higher than top-1 (e.g. RetroKNN:
57.2% top-1 vs 92.7% top-10 on USPTO-50K, unknown reaction class). This
script generates `--num-beams` candidates per record via beam search and
reports exact_match/core_exact_match at k=1,3,5 (capped at --num-beams),
deduplicating candidates by canonical SMILES before ranking so k reflects k
*distinct* chemical answers, not k beams that happen to decode to the same
molecule.

Example:
    python scripts/models/run_reactiont5_topk.py \\
        --input data/v2_ord_eval_targets.json \\
        --t5-model /content/drive/MyDrive/retro-planner-checkpoints/model1_reactant_v3/final \\
        --num-beams 10 --output topk_results.json

--num-beam-groups/--diversity-penalty switch on HF's diverse beam search: `--num-beams`
candidates are split into that many equal-size groups, and a token chosen by an earlier
group is penalized (by `--diversity-penalty`) for later groups at the same decoding step.
Plain beam search (the default here, groups=1) tends to fill the beam with near-duplicate
variants of the top 1-2 hypotheses; diverse beam search trades a bit of per-candidate
quality for genuinely different candidates, which is exactly what top-k accuracy rewards.
`--num-beams` must be divisible by `--num-beam-groups`.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

LOGGER = logging.getLogger("retro_eval.run_reactiont5_topk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="USPTO/ORD-format eval-targets JSON.")
    parser.add_argument("--t5-model", required=True)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--num-beam-groups", type=int, default=1, help="1 = plain beam search (default).")
    parser.add_argument("--diversity-penalty", type=float, default=0.0, help="Only used if --num-beam-groups > 1.")
    parser.add_argument("--max-length", type=int, default=150)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Products per generate() call. The default (1) matches this script's original "
            "behavior. On GPU, batch=1 leaves the device mostly idle (confirmed: <10%% "
            "utilization on a T4) since beam search for a single product is too small a "
            "workload to saturate it -- raise this (e.g. 16-32) for a real GPU speedup. "
            "CPU runs typically see little benefit and can even slow down from padding "
            "overhead, so leave at 1 there."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from retro_eval.evaluation import (
        canonical_precursor_set,
        is_core_exact_match,
        is_exact_match,
        is_valid_smiles,
        split_fragments,
        strip_salts_and_catalysts,
    )

    model_dir = Path(args.t5_model)
    if model_dir.is_dir():
        fix_legacy_tokenizer_config(model_dir)
        fix_tie_word_embeddings(model_dir)

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]
    LOGGER.info("Loaded %d record(s) from %s", len(records), args.input)

    LOGGER.info("Loading model=%s on device=%s ...", args.t5_model, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.t5_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.t5_model).to(args.device)
    model.eval()

    ks = [k for k in (1, 3, 5) if k <= args.num_beams]
    exact_hits = {k: 0 for k in ks}
    core_hits = {k: 0 for k in ks}
    valid_top1 = 0
    detailed = []

    generate_kwargs = dict(
        num_beams=args.num_beams,
        num_return_sequences=args.num_beams,
        max_length=args.max_length,
        min_length=1,
    )
    if args.num_beam_groups > 1:
        generate_kwargs["num_beam_groups"] = args.num_beam_groups
        generate_kwargs["diversity_penalty"] = args.diversity_penalty
        # transformers >=4.6x moved group beam search to a community-maintained
        # custom_generate repo (transformers-community/group-beam-search, HF-org
        # maintained); trust_remote_code is required to fetch and run it.
        generate_kwargs["trust_remote_code"] = True

    processed = 0
    for batch_start in range(0, len(records), args.batch_size):
        batch = records[batch_start : batch_start + args.batch_size]
        products = [r["product_smiles"] for r in batch]

        inputs = tokenizer(products, return_tensors="pt", padding=True).to(args.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generate_kwargs)
        decoded = [c.strip().replace(" ", "") for c in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]

        for row_idx, record in enumerate(batch):
            reference = record["reactants_smiles"]
            candidates_raw = decoded[row_idx * args.num_beams : (row_idx + 1) * args.num_beams]

            seen_sets = set()
            deduped = []
            for candidate in candidates_raw:
                canonical_set = canonical_precursor_set(split_fragments(candidate))
                key = canonical_set if canonical_set is not None else candidate
                if key in seen_sets:
                    continue
                seen_sets.add(key)
                deduped.append(candidate)

            if deduped:
                valid_top1 += is_valid_smiles(deduped[0])

            for k in ks:
                top_k = deduped[:k]
                exact_hits[k] += any(is_exact_match(split_fragments(c), split_fragments(reference)) for c in top_k)
                core_hits[k] += any(
                    is_exact_match(
                        strip_salts_and_catalysts(split_fragments(c)),
                        strip_salts_and_catalysts(split_fragments(reference)),
                    )
                    for c in top_k
                )

            detailed.append(
                {"product_smiles": record["product_smiles"], "reactants_smiles": reference, "candidates": deduped}
            )

        processed += len(batch)
        if processed % 25 == 0 or processed == len(records):
            LOGGER.info("Processed %d/%d", processed, len(records))

    del model, tokenizer
    gc.collect()

    n = len(records)
    summary = {
        "total": n,
        "num_beams": args.num_beams,
        "num_beam_groups": args.num_beam_groups,
        "diversity_penalty": args.diversity_penalty,
        "top1_valid_rate": valid_top1 / n if n else 0.0,
    }
    for k in ks:
        summary[f"exact_match_top{k}"] = exact_hits[k] / n if n else 0.0
        summary[f"core_exact_match_top{k}"] = core_hits[k] / n if n else 0.0

    LOGGER.info("Summary: %s", json.dumps(summary, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": detailed}, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote detailed results to %s", args.output)


if __name__ == "__main__":
    main()
