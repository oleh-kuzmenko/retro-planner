#!/usr/bin/env python3
"""Replace the reference reactant side of the conditions test with Model 1's answer.

The two models were always measured apart: Model 1 on products it must reconstruct
precursors for, Model 2 on precursors handed to it from ORD's record. Served as one
system, Model 2 never sees ORD's record -- it sees whatever Model 1 wrote, salts,
misread charge and all. This script produces the test file for that second reading:
every row keeps its reference conditions, and only the field Model 2 reads as the
reactant side is overwritten with the prediction that survived the RDKit filter.
The file is then scored by `evaluate_conditions_model_topk.py` unchanged, so the two
numbers differ in exactly one thing -- where the reactant side came from.

Rows whose candidates all fail the filter keep an empty reactant side rather than
being dropped: a cascade that proposes nothing usable has failed that record, and
dropping it would quietly raise the score by shrinking the denominator. The count is
reported separately so the two failure modes stay distinguishable.

Every finished row is appended immediately. A Colab session that dies mid-run costs
the current batch, not the run: pointing `--output` at the partial file resumes it.

Example:
    python scripts/models/cascade_stage1_substitute.py \
        --model1-dir /content/model1/final \
        --test-file data/v2_ord_roles_mixed/conditions_test_clean.jsonl \
        --limit 2000 --seed 0 --num-beams 5 --batch-size 32 --device cuda \
        --output /content/out/cascade_stage1_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from run_reactiont5_topk import fix_legacy_tokenizer_config, fix_tie_word_embeddings

LOGGER = logging.getLogger("retro_eval.cascade_stage1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model1-dir", required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument(
        "--reactants-field",
        default="full_reactants_smiles",
        help="The field Model 2 reads as the reactant side; must match its format marker.",
    )
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score a random subset of this size instead of the whole test file. Rows are "
        "visited in a seeded random order and keep their original index, so the reference-input "
        "run can be restricted to exactly the rows this one reached -- whether it finished or "
        "the session died halfway -- and the comparison stays paired over a uniform sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="Substituted test file (JSONL, appended to).")
    return parser.parse_args()


def canonical_key(smiles: str):
    """The comparison key for stage 1: the set of canonical fragments, or None if unparseable."""
    from retro_eval.evaluation import canonical_precursor_set, split_fragments

    return canonical_precursor_set(split_fragments(smiles))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from retro_eval.evaluation import is_exact_match, split_fragments, strip_salts_and_catalysts

    rows = [json.loads(line) for line in args.test_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)  # so a run cut short is still a uniform sample
    if args.limit:
        indices = indices[: args.limit]
    LOGGER.info("Stage 1 over %d of %d row(s) from %s", len(indices), len(rows), args.test_file)

    done = set()
    if args.output.is_file():
        done = {
            json.loads(line)["cascade_index"]
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        LOGGER.info("Resuming: %d row(s) already written to %s", len(done), args.output)
    indices = [i for i in indices if i not in done]

    model_dir = Path(args.model1_dir)
    if model_dir.is_dir():
        fix_legacy_tokenizer_config(model_dir)
        fix_tie_word_embeddings(model_dir)

    LOGGER.info("Loading Model 1 = %s on %s ...", args.model1_dir, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model1_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model1_dir).to(args.device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("a", encoding="utf-8") as sink:
        for batch_start in range(0, len(indices), args.batch_size):
            batch = indices[batch_start : batch_start + args.batch_size]
            products = [rows[i]["product_smiles"] for i in batch]
            inputs = tokenizer(products, return_tensors="pt", padding=True).to(args.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    max_length=args.max_length,
                    min_length=1,
                )
            decoded = [c.strip().replace(" ", "") for c in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]

            for position, index in enumerate(batch):
                row = dict(rows[index])
                reference = row[args.reactants_field]
                candidates = decoded[position * args.num_beams : (position + 1) * args.num_beams]

                seen, valid = set(), []
                for candidate in candidates:
                    key = canonical_key(candidate)
                    if key is None or key in seen:  # unparseable is what the filter drops
                        continue
                    seen.add(key)
                    valid.append(candidate)

                served = valid[0] if valid else ""
                row[args.reactants_field] = served
                row["cascade_index"] = index
                row["cascade_reference_reactants"] = reference
                row["cascade_candidates_generated"] = len(candidates)
                row["cascade_candidates_valid"] = len(valid)
                row["cascade_stage1_exact"] = bool(
                    served and is_exact_match(split_fragments(served), split_fragments(reference))
                )
                row["cascade_stage1_core_exact"] = bool(
                    served
                    and is_exact_match(
                        strip_salts_and_catalysts(split_fragments(served)),
                        strip_salts_and_catalysts(split_fragments(reference)),
                    )
                )
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            sink.flush()
            LOGGER.info("Written %d/%d", written, len(indices))

    LOGGER.info("Stage 1 file: %s", args.output)


if __name__ == "__main__":
    main()
