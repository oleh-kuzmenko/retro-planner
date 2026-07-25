#!/usr/bin/env python3
"""Step 2: ReactionT5v2 -- local HF seq2seq retrosynthesis, no reasoning trace.

Beam search (`num_beams=5`), top-1 SMILES out. Runs standalone so only this
one model is ever resident in memory; explicitly frees it
(`del model; gc.collect()`) before exiting. Defaults to CPU (float32) but
accepts `--device cuda` for the Colab GPU notebook (`colab/`).

Example:
    pip install -e ".[local-models,eval-runner]"
    python scripts/models/run_reactiont5.py --input data/uspto_eval_targets.json --limit 10
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

from retro_eval.harness.experiment import (
    DEFAULT_EXPERIMENTS_ROOT,
    create_experiment_run_dir,
    default_experiment_id,
    finish_run_meta,
    start_run_meta,
)
from retro_eval.harness.records import load_records
from retro_eval.harness.stage_runner import require_modules, run_model_over_records

LOGGER = logging.getLogger("retro_eval.run_reactiont5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2: ReactionT5v2 local seq2seq retrosynthesis.")
    parser.add_argument("--input", type=Path, required=True, help="USPTO/ORD-format dataset JSON.")
    parser.add_argument(
        "--model-slug", default="reactiont5v2", help="Folder name under experiments/<experiment_id>/."
    )
    parser.add_argument("--t5-model", default="sagawa/ReactionT5v2-retrosynthesis-USPTO_50k")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=150)
    parser.add_argument("--device", default="cpu", help="e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records.")
    parser.add_argument("--experiment-id", default=None, help="Defaults to today's date (YYYY-MM-DD).")
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    require_modules(("torch", "transformers"), "local-models")
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    records = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    LOGGER.info("Loaded %d record(s) from %s", len(records), args.input)

    experiment_id = args.experiment_id or default_experiment_id()
    run_dir = create_experiment_run_dir(args.experiments_root, experiment_id, args.model_slug)
    start_run_meta(
        run_dir,
        model_slug=args.model_slug,
        script="scripts/models/run_reactiont5.py",
        cli_args=vars(args),
        input_path=args.input,
    )
    LOGGER.info("Run directory: %s", run_dir)

    LOGGER.info("Loading ReactionT5v2 model=%s on device=%s ...", args.t5_model, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.t5_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.t5_model, dtype=torch.float32)
    model.to(args.device)
    model.eval()

    def build_request(record):
        return record.product_smiles

    def call_model(product_smiles: str) -> str:
        inputs = tokenizer([product_smiles], return_tensors="pt").to(args.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, num_beams=args.num_beams, max_length=args.max_length, min_length=1
            )
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    def parse_response(raw: str, record) -> tuple[str, dict]:
        return raw.strip().replace(" ", ""), {}

    results = run_model_over_records(
        records, run_dir, "Step 2: ReactionT5v2", build_request, call_model, parse_response
    )

    del model, tokenizer
    gc.collect()

    finish_run_meta(run_dir, record_count=len(results))
    LOGGER.info("Finished: %d result(s) written to %s", len(results), run_dir)


if __name__ == "__main__":
    main()
