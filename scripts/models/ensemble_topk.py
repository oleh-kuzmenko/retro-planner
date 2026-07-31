#!/usr/bin/env python3
"""Merge top-k candidate lists from two ReactionT5 checkpoints into one ranked pool.

No new model inference: both inputs are already-generated `run_reactiont5_topk.py`
output files (same `--input` eval-targets file, so records are in the same order).
For each record, candidates are **interleaved** (primary[0], secondary[0],
primary[1], secondary[1], ...) and re-deduplicated by canonical precursor set,
keeping the first (highest-ranked) occurrence. Plain concatenation (primary's full
list first) was tried first and made no difference at all -- a single model's own
`--num-beams 10` search already fills every top-5 slot with distinct candidates, so
anything appended after slot 10 never enters the top-5 window. Interleaving gives
the secondary model's candidates an actual chance to displace a primary candidate
within top-3/5, at the cost of primary's own top-1 no longer being guaranteed to
stay top-1 (secondary's top-1 can now win that slot too, if the primary model
already has multiple valid distinct candidates ranked above where a merge decision
matters). Reports the same top-1/3/5 metrics as `run_reactiont5_topk.py`.

Example:
    python scripts/models/ensemble_topk.py \\
        --primary experiments/v2_model1_eval/model1_v2_topk_ord.json \\
        --secondary experiments/v2_model1_topk/ord150k_v2cfg_topk.json \\
        --output experiments/v2_model1_topk/ensemble_v2_v7_ord.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primary", type=Path, required=True, help="run_reactiont5_topk.py output; own top-1 is kept first.")
    parser.add_argument("--secondary", type=Path, required=True, help="Second run_reactiont5_topk.py output on the same --input.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    from retro_eval.evaluation import (
        canonical_precursor_set,
        is_core_exact_match,
        is_exact_match,
        is_valid_smiles,
        split_fragments,
        strip_salts_and_catalysts,
    )

    args = parse_args()
    primary = json.loads(args.primary.read_text(encoding="utf-8"))["records"]
    secondary = json.loads(args.secondary.read_text(encoding="utf-8"))["records"]

    assert len(primary) == len(secondary), "record count mismatch -- did both runs use the same --input?"
    assert [r["product_smiles"] for r in primary] == [r["product_smiles"] for r in secondary], (
        "record order mismatch -- did both runs use the same --input?"
    )

    ks = [1, 3, 5]
    exact_hits = {k: 0 for k in ks}
    core_hits = {k: 0 for k in ks}
    valid_top1 = 0
    detailed = []

    for p_rec, s_rec in zip(primary, secondary):
        product = p_rec["product_smiles"]
        reference = p_rec["reactants_smiles"]

        interleaved = []
        for pair in zip(p_rec["candidates"], s_rec["candidates"]):
            interleaved.extend(pair)
        # tail of whichever list is longer
        interleaved.extend(p_rec["candidates"][len(s_rec["candidates"]):])
        interleaved.extend(s_rec["candidates"][len(p_rec["candidates"]):])

        seen_sets = set()
        merged = []
        for candidate in interleaved:
            canonical_set = canonical_precursor_set(split_fragments(candidate))
            key = canonical_set if canonical_set is not None else candidate
            if key in seen_sets:
                continue
            seen_sets.add(key)
            merged.append(candidate)

        if merged:
            valid_top1 += is_valid_smiles(merged[0])

        for k in ks:
            top_k = merged[:k]
            exact_hits[k] += any(is_exact_match(split_fragments(c), split_fragments(reference)) for c in top_k)
            core_hits[k] += any(
                is_exact_match(
                    strip_salts_and_catalysts(split_fragments(c)),
                    strip_salts_and_catalysts(split_fragments(reference)),
                )
                for c in top_k
            )

        detailed.append({"product_smiles": product, "reactants_smiles": reference, "candidates": merged})

    n = len(primary)
    summary = {"total": n, "top1_valid_rate": valid_top1 / n if n else 0.0}
    for k in ks:
        summary[f"exact_match_top{k}"] = exact_hits[k] / n if n else 0.0
        summary[f"core_exact_match_top{k}"] = core_hits[k] / n if n else 0.0

    print(json.dumps(summary, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": detailed}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
