#!/usr/bin/env python3
"""Zero-shot retrosynthesis against any OpenAI-compatible chat endpoint.

An Ollama-served alternative to `run_chemllm.py` (Step 3) and
`run_qwen_lora_peft.py` (Step 4) for machines without a usable GPU: both are
"POST chat messages to an OpenAI-compatible endpoint, zero-shot, parse the
response," differing only in the response contract (`--response-format`).
Adding a future model server this way is a new invocation, not a new script.

Example (ChemLLM via Ollama, CoT tags):
    pip install -e ".[eval-runner]"
    ollama serve &
    python scripts/models/run_chat_zero_shot.py --input data/uspto_eval_targets.json \\
        --model-slug chemllm_ollama --model chemllm --response-format cot

Example (a merged/quantized Qwen+LoRA checkpoint served as e.g. "qwen25-retro-lora" in Ollama,
matching the fine-tuned JSON reactants contract):
    python scripts/models/run_chat_zero_shot.py --input data/uspto_eval_targets.json \\
        --model-slug qwen25_7b_lora_ollama --model qwen25-retro-lora --response-format json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from retro_eval.harness.experiment import (
    DEFAULT_EXPERIMENTS_ROOT,
    create_experiment_run_dir,
    default_experiment_id,
    finish_run_meta,
    start_run_meta,
)
from retro_eval.harness.parsing import parse_cot_answer, parse_json_reactants_response
from retro_eval.harness.records import EvalRecord, load_records
from retro_eval.harness.stage_runner import run_model_over_records
from retro_eval.providers.chat_api import OpenAICompatibleLLMProvider
from retro_eval.providers.retrying import RetryingProvider

LOGGER = logging.getLogger("retro_eval.run_chat_zero_shot")

DEFAULT_COT_SYSTEM_PROMPT = """You are an expert computational organic chemist specialized in single-step retrosynthesis analysis.
Given a target molecule's SMILES string, propose the most chemically plausible set of starting reactants that would produce it in a single reaction step, favoring well-precedented named reactions.
Think step by step inside <think>...</think> tags: identify the key functional groups, the most likely disconnection (retron), and the named reaction type.
Then output ONLY the reactant SMILES strings, separated by a dot ('.'), inside <answer>...</answer> tags, with no additional commentary outside these tags."""

DEFAULT_JSON_SYSTEM_PROMPT = """You are a chemistry reaction prediction model.
Return valid compact JSON only, of the shape {"reactants": ["<SMILES>", ...], "reaction_class": "<short name>"}.
Predict the reactants and reaction class for a single-step synthesis of the given target product.
Use canonical SMILES strings when possible. Do not include conditions or explanations."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot retrosynthesis against any OpenAI-compatible chat endpoint (Ollama, llama.cpp server, ...)."
    )
    parser.add_argument("--input", type=Path, required=True, help="USPTO/ORD-format dataset JSON.")
    parser.add_argument("--model-slug", required=True, help="Folder name under experiments/<experiment_id>/.")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", required=True, help="Server-side model tag/name.")
    parser.add_argument("--api-key", default="ollama", help="Ollama ignores this; some servers require a non-empty value.")
    parser.add_argument("--response-format", choices=("cot", "json"), default="cot")
    parser.add_argument("--system-prompt", default=None, help="Override the default system prompt for --response-format.")
    parser.add_argument("--system-prompt-file", type=Path, default=None, help="Read the system prompt from a file instead.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records.")
    parser.add_argument("--experiment-id", default=None, help="Defaults to today's date (YYYY-MM-DD).")
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    LOGGER.info("Loaded %d record(s) from %s", len(records), args.input)

    experiment_id = args.experiment_id or default_experiment_id()
    run_dir = create_experiment_run_dir(args.experiments_root, experiment_id, args.model_slug)
    start_run_meta(
        run_dir,
        model_slug=args.model_slug,
        script="scripts/models/run_chat_zero_shot.py",
        cli_args=vars(args),
        input_path=args.input,
    )
    LOGGER.info("Run directory: %s", run_dir)

    if args.system_prompt_file:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8")
    elif args.system_prompt:
        system_prompt = args.system_prompt
    else:
        system_prompt = DEFAULT_JSON_SYSTEM_PROMPT if args.response_format == "json" else DEFAULT_COT_SYSTEM_PROMPT

    provider = RetryingProvider(
        OpenAICompatibleLLMProvider(api_key=args.api_key, base_url=args.base_url),
        max_attempts=args.max_retries,
    )

    def build_request(record: EvalRecord) -> list[dict]:
        if args.response_format == "json":
            user_content = json.dumps(
                {"task": "predict_reactants_and_class", "product_smiles": record.product_smiles},
                separators=(",", ":"),
            )
        else:
            user_content = (
                f"Target molecule (SMILES): {record.product_smiles}\n\n"
                "Propose the reactant(s) needed to synthesize this target in a single retrosynthetic step."
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def call_model(messages: list[dict]) -> str:
        return provider.generate(messages=messages, model=args.model, temperature=args.temperature)

    def parse_response(raw: str, record: EvalRecord) -> tuple[str, dict]:
        if args.response_format == "json":
            return parse_json_reactants_response(raw)
        return parse_cot_answer(raw, record.product_smiles)

    results = run_model_over_records(
        records, run_dir, f"{args.model_slug} (zero-shot, {args.response_format})", build_request, call_model, parse_response
    )

    finish_run_meta(run_dir, record_count=len(results))
    LOGGER.info("Finished: %d result(s) written to %s", len(results), run_dir)


if __name__ == "__main__":
    main()
