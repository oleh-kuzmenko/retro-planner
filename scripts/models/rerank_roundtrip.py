#!/usr/bin/env python3
"""Rerank top-k retrosynthesis candidates via round-trip (forward) consistency.

No new retrosynthesis inference: takes an already-generated `run_reactiont5_topk.py`
(or `ensemble_topk.py`) output file. For each record, feeds every candidate reactant
set through `sagawa/ReactionT5v2-forward` (ORD-pretrained forward-reaction
checkpoint, zero-shot -- no fine-tuning) to predict the product, and checks whether
that predicted product canonically matches the record's real target product. This
does not require the correct *reactants* to be known -- only the target product,
which is always available at inference time -- so it's a legitimate reranking
signal, not test-set leakage.

Candidates whose predicted product matches the real product ("verified") are moved
to the front, in their original relative order; unverified candidates keep their
original order after. If a record has zero verified candidates, its ranking is
unchanged. Reports the same top-1/3/5 metrics as `run_reactiont5_topk.py`.

Input format for the forward model: `REACTANT:<candidate>` (REAGENT: omitted --
this project's reactants_smiles already mixes reactant/reagent roles into one
field, matching how the forward model's REACTANT section is used when reagents
are unknown).

Example:
    python scripts/models/rerank_roundtrip.py \\
        --input experiments/v2_model1_topk/ensemble_v2_v7_ord.json \\
        --max-candidates 10 \\
        --output experiments/v2_model1_topk/roundtrip_ensemble_ord.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

LOGGER = logging.getLogger("retro_eval.rerank_roundtrip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="run_reactiont5_topk.py / ensemble_topk.py output.")
    parser.add_argument("--forward-model", default="sagawa/ReactionT5v2-forward")
    parser.add_argument("--max-candidates", type=int, default=10, help="Only rerank within the first N candidates.")
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from retro_eval.chemistry import canonicalize_smiles
    from retro_eval.evaluation import (
        is_core_exact_match,
        is_exact_match,
        is_valid_smiles,
        split_fragments,
        strip_salts_and_catalysts,
    )

    records = json.loads(args.input.read_text(encoding="utf-8"))["records"]
    if args.limit:
        records = records[: args.limit]
    LOGGER.info("Loaded %d record(s) from %s", len(records), args.input)

    LOGGER.info("Loading forward model=%s on device=%s ...", args.forward_model, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.forward_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.forward_model).to(args.device)
    model.eval()

    ks = [1, 3, 5]
    exact_hits = {k: 0 for k in ks}
    core_hits = {k: 0 for k in ks}
    valid_top1 = 0
    verified_at_least_one = 0
    detailed = []

    for i, record in enumerate(records):
        product = record["product_smiles"]
        reference = record["reactants_smiles"]
        candidates = record["candidates"][: args.max_candidates]
        reference_product_canonical = canonicalize_smiles(product) or product

        verified_flags = []
        for candidate in candidates:
            prompt = f"REACTANT:{candidate}"
            inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=150).to(args.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_length=args.max_length, num_beams=1, min_length=1)
            raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(" ", "").rstrip(".")
            predicted_canonical = canonicalize_smiles(raw)
            verified_flags.append(predicted_canonical is not None and predicted_canonical == reference_product_canonical)

        if any(verified_flags):
            verified_at_least_one += 1

        reranked = [c for c, v in zip(candidates, verified_flags) if v] + [
            c for c, v in zip(candidates, verified_flags) if not v
        ] + record["candidates"][args.max_candidates :]

        if reranked:
            valid_top1 += is_valid_smiles(reranked[0])

        for k in ks:
            top_k = reranked[:k]
            exact_hits[k] += any(is_exact_match(split_fragments(c), split_fragments(reference)) for c in top_k)
            core_hits[k] += any(
                is_exact_match(
                    strip_salts_and_catalysts(split_fragments(c)),
                    strip_salts_and_catalysts(split_fragments(reference)),
                )
                for c in top_k
            )

        detailed.append(
            {
                "product_smiles": product,
                "reactants_smiles": reference,
                "candidates_reranked": reranked,
                "verified_flags": verified_flags,
            }
        )
        if (i + 1) % 25 == 0:
            LOGGER.info("Processed %d/%d", i + 1, len(records))

    del model, tokenizer
    gc.collect()

    n = len(records)
    summary = {
        "total": n,
        "top1_valid_rate": valid_top1 / n if n else 0.0,
        "verified_at_least_one_rate": verified_at_least_one / n if n else 0.0,
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
