#!/usr/bin/env python3
"""Step 4: Qwen2.5-7B-Instruct + a retrosynthesis LoRA adapter, local HF `peft`.

Defaults to CPU with no bitsandbytes/4-bit quantization -- CPU-only bnb
support is unreliable, so the base model loads directly in `--dtype`
(float32/bfloat16/float16; bfloat16 by default needs ~15GB for weights
alone, float32 ~28GB). Pass `--device cuda` for the Colab GPU notebook
(`colab/`); on GPUs too small for ~15GB of bfloat16 weights (e.g. a Colab
T4, ~14.5GB usable), also pass `--load-in-4bit` to load nf4-quantized via
bitsandbytes, matching this adapter's QLoRA training recipe. Greedy
decoding (deterministic, reproducible).

Two `--prompt-style`s:
  - "json" (default): the JSON-in/JSON-out contract this project's own
    fine-tuned adapter (`oleh13/retro-reactants-qwen25-7b-lora`) was trained
    on.
  - "cot": generic `<think>/<answer>` instruction-following, for a
    differently fine-tuned (or base, un-adapted) model.

Alternative if this doesn't fit in memory or is too slow: merge the LoRA into
the base model, quantize to GGUF, serve it with Ollama/llama.cpp, and run
`scripts/models/run_chat_zero_shot.py --response-format json` (or `cot`)
against that endpoint instead -- same prompt/parse contracts, no in-process
model loading.

Example:
    pip install -e ".[local-models,eval-runner]"
    python scripts/models/run_qwen_lora_peft.py --input data/uspto_eval_targets.json --limit 10
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
from retro_eval.harness.parsing import parse_cot_answer, parse_json_reactants_response
from retro_eval.harness.records import EvalRecord, load_records
from retro_eval.harness.stage_runner import require_modules, run_model_over_records
from retro_eval.prompting import build_cot_prompt

LOGGER = logging.getLogger("retro_eval.run_qwen_lora_peft")

# This project's own fine-tuned reactants/class adapter (see research/fine-tune/v2/).
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
REACTANT_LORA_ID = "oleh13/retro-reactants-qwen25-7b-lora"
REACTANT_SYSTEM = """You are a chemistry reaction prediction model.
Return valid compact JSON only.
Predict reactants and reaction class for a single-step synthesis of the target product.
Use canonical SMILES strings when possible.
Do not include conditions or explanations."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4: Qwen2.5-7B + LoRA, local HF peft.")
    parser.add_argument("--input", type=Path, required=True, help="USPTO/ORD-format dataset JSON.")
    parser.add_argument(
        "--model-slug", default="qwen25_7b_lora_peft", help="Folder name under experiments/<experiment_id>/."
    )
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--lora-adapter", default=REACTANT_LORA_ID)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--prompt-style", choices=("json", "cot"), default="json")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--device", default="cpu", help="e.g. cpu, cuda, cuda:0.")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help=(
            "Load the base model 4-bit-quantized via bitsandbytes (nf4, matching this "
            "adapter's QLoRA training recipe). Needed on GPUs that can't fit ~15GB of "
            "bfloat16 weights, e.g. a Colab T4 (~14.5GB usable). Requires --device cuda."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records.")
    parser.add_argument("--experiment-id", default=None, help="Defaults to today's date (YYYY-MM-DD).")
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    required = ("torch", "transformers", "peft")
    if args.load_in_4bit:
        if args.device == "cpu":
            raise SystemExit("--load-in-4bit requires --device cuda (CPU-only bnb is unreliable).")
        required += ("bitsandbytes",)
    require_modules(required, "local-models")
    import json as json_module

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    records = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    LOGGER.info("Loaded %d record(s) from %s", len(records), args.input)

    experiment_id = args.experiment_id or default_experiment_id()
    run_dir = create_experiment_run_dir(args.experiments_root, experiment_id, args.model_slug)
    start_run_meta(
        run_dir,
        model_slug=args.model_slug,
        script="scripts/models/run_qwen_lora_peft.py",
        cli_args=vars(args),
        input_path=args.input,
    )
    LOGGER.info("Run directory: %s", run_dir)

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]
    LOGGER.info(
        "Loading base=%s + LoRA adapter=%s dtype=%s on device=%s (needs ~%s for weights alone).",
        args.base_model,
        args.lora_adapter,
        args.dtype,
        args.device,
        "~28GB" if dtype is torch.float32 else "~15GB",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"dtype": dtype, "trust_remote_code": True}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": args.device}
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, args.lora_adapter)
    if not args.load_in_4bit:
        model.to(args.device)
    model.eval()

    def build_request(record: EvalRecord) -> list[dict]:
        if args.prompt_style == "json":
            return [
                {"role": "system", "content": REACTANT_SYSTEM},
                {
                    "role": "user",
                    "content": json_module.dumps(
                        {"task": "predict_reactants_and_class", "product_smiles": record.product_smiles},
                        separators=(",", ":"),
                    ),
                },
            ]
        return [{"role": "user", "content": build_cot_prompt(record.product_smiles, None)}]

    def call_model(messages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to(args.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()

    def parse_response(raw: str, record: EvalRecord) -> tuple[str, dict]:
        if args.prompt_style == "json":
            return parse_json_reactants_response(raw)
        return parse_cot_answer(raw, record.product_smiles)

    results = run_model_over_records(
        records, run_dir, f"Step 4: Qwen2.5-7B + LoRA ({args.prompt_style})", build_request, call_model, parse_response
    )

    del model, base_model, tokenizer
    gc.collect()

    finish_run_meta(run_dir, record_count=len(results))
    LOGGER.info("Finished: %d result(s) written to %s", len(results), run_dir)


if __name__ == "__main__":
    main()
