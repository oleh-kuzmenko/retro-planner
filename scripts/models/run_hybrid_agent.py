#!/usr/bin/env python3
"""Hybrid Retrieve-Generate-Rerank pipeline: ReactionT5v2 + Qdrant RAG + LLM Chemist-Agent.

Combines three components already used separately elsewhere in this codebase into one
per-record pipeline:

1. **Generate** -- ReactionT5v2 (`run_reactiont5.py`'s model, local HF seq2seq, CPU) proposes
   `--num-candidates` beam-search reactant-SMILES hypotheses for the target product.
2. **Retrieve** -- Qdrant hybrid RAG (`retro_eval.retrieval.hybrid_retrieve_reactions_for_smiles`,
   the same call `run_rag_cot_llm.py` (step 5) makes) fetches `--rag-top-k` literature
   precedent reactions for the same target.
3. **Rerank** -- a hosted LLM (`retro_eval.prompting.build_rerank_prompt`) is shown the
   target, the retrieved precedents, and the T5 candidates, and asked to pick the most
   chemically plausible one inside `<answer>` tags (reasoning inside `<think>`), parsed by
   `retro_eval.harness.parsing.parse_rerank_answer`. Unlike the step-5 CoT contract (the LLM
   proposes its own disconnection), the LLM here only ever selects among T5's candidates --
   if `<answer>` is missing or chemically invalid, the pipeline falls back to the first
   RDKit-*valid* T5 candidate (not blindly the top beam), and reports the record as an
   honest failure (empty prediction) only if none of the candidates parse at all.

**Hardware note (CPU-only, 16 GB RAM laptop):** records are processed strictly one at a
time -- generate, retrieve, rerank, write, next record -- so only one target's tensors and
one in-flight API call ever exist at once; nothing is batched or threaded. ReactionT5v2
stays resident for the whole run (freeing/reloading it per record would be far more
expensive than keeping ~1 GB of weights loaded) and is explicitly released
(`del model; gc.collect()`) only once the loop finishes, same as `run_reactiont5.py`.

**Output layout matches every other model-runner script exactly** --
`experiments/<experiment_id>/<model_slug>/{results.json,inference_logs.json,run_meta.json}`
via `retro_eval.harness.experiment` -- so `scripts/aggregate_results.py` discovers this run
automatically alongside ReactionT5v2/ChemLLM/Qwen-LoRA/RAG+CoT runs under the same
`--experiment-id`, with no code changes to the aggregator. `results.json` rows use the
same `predicted_reactants_smiles` key every other script writes (the aggregator's join key);
`inference_logs.json` rows additionally carry `t5_candidates`, `retrieved_reactions`,
`prompt_sent_to_llm`, and `raw_llm_response` for qualitative citation. (This intentionally
does not use separate `--output`/`--logs` file paths or a `predicted_reactants` key, since
this repo's actual aggregator only understands the dated run-folder layout above.)

Talks to any OpenAI-compatible chat-completions endpoint via `--base-url`/`--api-key`/
`--model` for the reranking call -- e.g. Groq's `llama-3.3-70b-versatile`, or any other
OpenAI/Cerebras/OpenRouter-hosted model. Rate limits and long-pause/resume behavior are
identical to `run_rag_cot_llm.py`/`run_cot_llm.py`.

Example:
    pip install -e ".[local-models,eval-runner]"
    docker compose up -d qdrant
    python scripts/index_uspto_to_qdrant.py     # if not already indexed
    python scripts/models/run_hybrid_agent.py --input data/uspto_eval_targets.json --limit 10 \\
        --base-url https://api.groq.com/openai/v1 --api-key $GROQ_API_KEY \\
        --model llama-3.3-70b-versatile
    # If it pauses on a rate limit, just rerun the same command later.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from retro_eval.config import QDRANT_HOST, QDRANT_PORT
from retro_eval.harness.experiment import (
    DEFAULT_EXPERIMENTS_ROOT,
    RESULTS_FILENAME,
    InferenceLogWriter,
    create_experiment_run_dir,
    default_experiment_id,
    finish_run_meta,
    save_json_atomic,
    start_run_meta,
)
from retro_eval.harness.parsing import parse_rerank_answer
from retro_eval.harness.records import load_records
from retro_eval.harness.stage_runner import require_modules
from retro_eval.prompting import build_rerank_prompt
from retro_eval.providers.chat_api import OpenAICompatibleLLMProvider
from retro_eval.providers.retrying import ProviderPaused, RetryingProvider
from retro_eval.retrieval import RetrievalConfig, create_qdrant_client, hybrid_retrieve_reactions_for_smiles
from tqdm import tqdm

LOGGER = logging.getLogger("retro_eval.run_hybrid_agent")


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = [f"{hours}h" for _ in [0] if hours] + [f"{minutes}m" for _ in [0] if minutes] + [f"{seconds}s"]
    return "".join(parts)


def resume_pointer_path(experiments_root: Path, model_slug: str, input_path: Path) -> Path:
    return experiments_root / ".resume" / f"{model_slug}__{input_path.stem}.json"


def load_resume_pointer(pointer_path: Path) -> dict | None:
    if not pointer_path.exists():
        return None
    try:
        return json.loads(pointer_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_resume_pointer(pointer_path: Path, experiment_id: str, run_dir: Path) -> None:
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    save_json_atomic(
        pointer_path,
        {
            "experiment_id": experiment_id,
            "run_dir": str(run_dir),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def generate_t5_candidates(
    tokenizer,
    model,
    product_smiles: str,
    num_candidates: int,
    max_length: int,
    device: str,
    torch,
) -> list[str]:
    """Beam-search `num_candidates` reactant-SMILES hypotheses for one product SMILES.

    `num_beams == num_return_sequences == num_candidates` so beam search returns exactly
    that many distinct hypotheses ranked by sequence score, matching the parameters
    `run_reactiont5.py` uses for its single top-1 prediction.
    """
    inputs = tokenizer([product_smiles], return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=num_candidates,
            num_return_sequences=num_candidates,
            max_length=max_length,
            min_length=1,
        )
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [candidate.strip().replace(" ", "") for candidate in decoded]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid pipeline: ReactionT5v2 candidates + Qdrant RAG + LLM-API reranking."
    )
    parser.add_argument("--input", type=Path, required=True, help="USPTO/ORD-format dataset JSON.")
    parser.add_argument(
        "--model-slug", default="hybrid_t5_rag_rerank", help="Folder name under experiments/<experiment_id>/."
    )
    parser.add_argument("--t5-model", default="sagawa/ReactionT5v2-retrosynthesis-USPTO_50k")
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=5,
        help="Number of ReactionT5v2 beam-search candidates shown to the reranker "
        "(sets num_beams == num_return_sequences).",
    )
    parser.add_argument("--max-length", type=int, default=150)
    parser.add_argument("--device", default="cpu", help="e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--qdrant-host", default=QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=QDRANT_PORT)
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument(
        "--base-url",
        required=True,
        help="Any OpenAI-compatible chat-completions endpoint for the reranker call, e.g. "
        "https://api.groq.com/openai/v1 (Groq), https://openrouter.ai/api/v1 (OpenRouter), "
        "https://api.cerebras.ai/v1 (Cerebras), or a local Ollama server.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_API_KEY", ""),
        help="API key for --base-url. Defaults to $LLM_API_KEY.",
    )
    parser.add_argument("--model", default="llama-3.3-70b-versatile", help="Reranker model id.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records.")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Defaults to today's date, or to a saved resume pointer's experiment_id if one exists.",
    )
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument(
        "--rate-limit-auto-wait-seconds",
        type=float,
        default=300,
        help="Sleep through a 429 automatically if its suggested wait is at most this long.",
    )
    parser.add_argument(
        "--rate-limit-default-wait-seconds",
        type=float,
        default=3600,
        help="Fallback wait when a 429 response doesn't include a parseable Retry-After.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any saved resume pointer and start a brand-new run (does not delete old results).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.api_key:
        raise SystemExit("This script requires --api-key or the LLM_API_KEY environment variable.")

    require_modules(("torch", "transformers"), "local-models")
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    records = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    LOGGER.info("Loaded %d record(s) from %s", len(records), args.input)

    pointer_path = resume_pointer_path(args.experiments_root, args.model_slug, args.input)
    resume_pointer = None if args.fresh else load_resume_pointer(pointer_path)

    if args.experiment_id:
        experiment_id = args.experiment_id
    elif resume_pointer:
        experiment_id = resume_pointer["experiment_id"]
        LOGGER.info("Found resume pointer -> reusing experiment_id=%s", experiment_id)
    else:
        experiment_id = default_experiment_id()

    run_dir = create_experiment_run_dir(args.experiments_root, experiment_id, args.model_slug)
    save_resume_pointer(pointer_path, experiment_id, run_dir)
    start_run_meta(
        run_dir,
        model_slug=args.model_slug,
        script="scripts/models/run_hybrid_agent.py",
        cli_args=vars(args),
        input_path=args.input,
    )
    LOGGER.info("Run directory: %s", run_dir)

    LOGGER.info("Loading ReactionT5v2 model=%s on device=%s ...", args.t5_model, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.t5_model)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(args.t5_model, dtype=torch.float32)
    t5_model.to(args.device)
    t5_model.eval()

    provider = RetryingProvider(
        OpenAICompatibleLLMProvider(api_key=args.api_key, base_url=args.base_url),
        max_attempts=args.max_retries,
        rate_limit_auto_wait_seconds=args.rate_limit_auto_wait_seconds,
        rate_limit_default_wait_seconds=args.rate_limit_default_wait_seconds,
    )
    retrieval_config = RetrievalConfig(host=args.qdrant_host, port=args.qdrant_port)
    qdrant_client = create_qdrant_client(retrieval_config)

    results_path = run_dir / RESULTS_FILENAME
    results: list[dict] = (
        json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    )
    done_indices = {result["index"] for result in results}
    if done_indices:
        pending = [record for record in records if record.index not in done_indices]
        LOGGER.info(
            "Resuming: %d/%d record(s) already completed, %d remaining.",
            len(done_indices),
            len(records),
            len(pending),
        )
        records = pending

    log = InferenceLogWriter(run_dir)

    for record in tqdm(records, desc=f"Hybrid T5+RAG+Rerank ({args.model})"):
        entry: dict = {"index": record.index, "product_smiles": record.product_smiles}
        predicted, raw = "", ""
        try:
            candidates = generate_t5_candidates(
                tokenizer,
                t5_model,
                record.product_smiles,
                args.num_candidates,
                args.max_length,
                args.device,
                torch,
            )
            entry["t5_candidates"] = candidates

            retrieval_result = hybrid_retrieve_reactions_for_smiles(
                record.product_smiles,
                top_k=args.rag_top_k,
                client=qdrant_client,
                config=retrieval_config,
            )
            reactions = retrieval_result.reactions
            entry["retrieved_reactions"] = reactions
            entry["retrieval_warnings"] = retrieval_result.warnings

            prompt = build_rerank_prompt(record.product_smiles, reactions, candidates)
            entry["prompt_sent_to_llm"] = prompt

            raw = provider.generate(
                messages=[{"role": "user", "content": prompt}],
                model=args.model,
                temperature=args.temperature,
                json_mode=False,
            )
            entry["raw_llm_response"] = raw

            predicted, extra = parse_rerank_answer(raw, candidates)
            entry.update(extra)
        except ProviderPaused as exc:
            LOGGER.warning(
                "Rate limit requires a long pause (~%s) at index=%d. "
                "Progress so far is saved in %s -- rerun the exact same command "
                "later to resume (it will skip everything already completed).",
                format_duration(exc.wait_seconds),
                record.index,
                run_dir,
            )
            finish_run_meta(run_dir, record_count=len(results), status="paused_rate_limited")
            raise SystemExit(
                f"Paused on a rate limit at index={record.index}; "
                f"suggested wait ~{format_duration(exc.wait_seconds)}. "
                "Rerun the same command later to resume."
            ) from exc
        except Exception as exc:
            LOGGER.exception("Hybrid inference failed for index=%d", record.index)
            entry["error"] = str(exc)

        results.append(
            {
                "index": record.index,
                "product_smiles": record.product_smiles,
                "reactants_smiles": record.reactants_smiles,
                "predicted_reactants_smiles": predicted,
                "raw_response": raw,
            }
        )
        save_json_atomic(results_path, results)
        log.append(entry)

    del t5_model, tokenizer
    gc.collect()

    finish_run_meta(run_dir, record_count=len(results))
    pointer_path.unlink(missing_ok=True)
    LOGGER.info("Finished: %d result(s) written to %s", len(results), run_dir)


if __name__ == "__main__":
    main()
