#!/usr/bin/env python3
"""Step 3: ChemLLM-20B-Chat-SFT (GGUF, quantized), zero-shot retrosynthesis.

Bespoke `llama-cpp-python` in-process load -- the GGUF quantization brings
20B params comfortably under a 15GB Colab GPU (`--n-gpu-layers -1` offloads
every layer). ChemLLM follows its native concise `Answer: <SMILES>` contract;
prose without that final line is never scored as a SMILES prediction.

Example:
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
    python scripts/models/run_chemllm.py --input data/uspto_eval_targets.json --limit 10
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
from retro_eval.harness.parsing import parse_chemllm_answer
from retro_eval.harness.records import EvalRecord, load_records
from retro_eval.harness.stage_runner import require_modules, run_model_over_records

LOGGER = logging.getLogger("retro_eval.run_chemllm")

REPO_ID = "mradermacher/ChemLLM-20B-Chat-SFT-i1-GGUF"
FILENAME = "ChemLLM-20B-Chat-SFT.i1-Q4_K_M.gguf"

SYSTEM_PROMPT = """You are an expert organic chemist performing one single-step retrosynthesis prediction.
Return exactly one final line in this format:
Answer: <reactant_1_SMILES>.<reactant_2_SMILES>
Return only the reactant SMILES needed immediately before the target. Do not include reagents, solvents, explanations, Markdown, atom maps, or any text after the Answer line."""


def build_chemllm_prompt(product_smiles: str) -> str:
    return f"Target product (canonical SMILES): {product_smiles}\nReturn the required final Answer line now."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3: ChemLLM-20B-Chat-SFT GGUF, zero-shot retrosynthesis.")
    parser.add_argument("--input", type=Path, required=True, help="USPTO/ORD-format dataset JSON.")
    parser.add_argument("--model-slug", default="chemllm_20b_gguf", help="Folder name under experiments/<experiment_id>/.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--filename", default=FILENAME)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 offloads every layer to the GPU.")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records.")
    parser.add_argument("--experiment-id", default=None, help="Defaults to today's date (YYYY-MM-DD).")
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    require_modules(("llama_cpp",), "local-models")
    from llama_cpp import Llama

    records = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    LOGGER.info("Loaded %d record(s) from %s", len(records), args.input)

    experiment_id = args.experiment_id or default_experiment_id()
    run_dir = create_experiment_run_dir(args.experiments_root, experiment_id, args.model_slug)
    start_run_meta(
        run_dir,
        model_slug=args.model_slug,
        script="scripts/models/run_chemllm.py",
        cli_args=vars(args),
        input_path=args.input,
    )
    LOGGER.info("Run directory: %s", run_dir)

    LOGGER.info("Loading %s (%s), n_gpu_layers=%d ...", args.repo_id, args.filename, args.n_gpu_layers)
    llm = Llama.from_pretrained(
        repo_id=args.repo_id,
        filename=args.filename,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )

    def build_request(record: EvalRecord) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_chemllm_prompt(record.product_smiles)},
        ]

    def call_model(messages: list[dict]) -> str:
        output = llm.create_chat_completion(
            messages=messages, temperature=args.temperature, max_tokens=args.max_tokens
        )
        return output["choices"][0]["message"]["content"].strip()

    def parse_response(raw: str, record: EvalRecord) -> tuple[str, dict]:
        return parse_chemllm_answer(raw, record.product_smiles)

    results = run_model_over_records(
        records, run_dir, "Step 3: ChemLLM-20B-Chat-SFT (GGUF, Answer-line contract)", build_request, call_model, parse_response
    )

    del llm
    gc.collect()

    finish_run_meta(run_dir, record_count=len(results))
    LOGGER.info("Finished: %d result(s) written to %s", len(results), run_dir)


if __name__ == "__main__":
    main()
