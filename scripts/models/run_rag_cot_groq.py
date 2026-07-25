#!/usr/bin/env python3
"""Step 5: Qdrant hybrid RAG retrieval + Chain-of-Thought via Groq's Llama-3.3-70B.

The proposed hybrid system: Morgan + reaction-transform fingerprint retrieval
(`retro_eval.retrieval.hybrid_retrieve_reactions_for_smiles`) feeds retrieved
precedent reactions into the same 4-block CoT prompt/repair-retry pipeline
the rest of the codebase uses (`retro_eval.planning.generate_single_step`),
called through the Groq API.

If a record carries ORD condition metadata (`solvent`/`temperature_celsius`/
`catalyst`/`yield_percent`/`reaction_id`), a SEPARATE follow-up call also asks
the model to propose reaction conditions -- logged for qualitative review,
never scored programmatically (comparing predicted vs. reference conditions
is not something this script attempts).

Example:
    pip install -e ".[eval-runner]"
    docker compose up -d qdrant
    GROQ_API_KEY=... python scripts/models/run_rag_cot_groq.py --input data/ord_eval_targets.json --limit 10
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict
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
from retro_eval.harness.records import EvalRecord, load_records
from retro_eval.planning import GenerationRequest, generate_single_step
from retro_eval.providers.chat_api import GroqLLMProvider
from retro_eval.providers.retrying import RetryingProvider
from retro_eval.retrieval import RetrievalConfig, create_qdrant_client, hybrid_retrieve_reactions_for_smiles
from tqdm import tqdm

LOGGER = logging.getLogger("retro_eval.run_rag_cot_groq")


def build_condition_prompt(product_smiles: str, predicted_reactants: str, retrieved_reactions: list[dict]) -> str:
    precedent_lines = [
        f"- solvent={reaction.get('solvent') or 'unknown'}, "
        f"temperature_celsius={reaction.get('temperature_celsius')}, "
        f"catalyst={reaction.get('catalyst') or 'none'}, "
        f"yield_percent={reaction.get('yield_percent')}"
        for reaction in retrieved_reactions
    ]
    precedent_block = "\n".join(precedent_lines) or "No condition precedents were retrieved."
    return f"""[System] You are an expert organic chemist estimating practical reaction conditions.
[Context] Conditions reported for similar retrieved reaction precedents:
{precedent_block}
[Instruction] Given the product and reactants below, propose practical reaction conditions: solvent(s), an approximate temperature in Celsius, and a catalyst if one is typically required. Respond with a compact JSON object with keys "solvent", "temperature_celsius", "catalyst" and nothing else.
[Input] Product (SMILES): {product_smiles}
Reactants (SMILES): {predicted_reactants or "unknown"}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5: Qdrant hybrid RAG + CoT via Groq's Llama-3.3-70B.")
    parser.add_argument("--input", type=Path, required=True, help="USPTO/ORD-format dataset JSON.")
    parser.add_argument(
        "--model-slug", default="rag_cot_llama70b", help="Folder name under experiments/<experiment_id>/."
    )
    parser.add_argument("--groq-api-key", default=os.getenv("GROQ_API_KEY", ""))
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--qdrant-host", default=QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=QDRANT_PORT)
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records.")
    parser.add_argument("--experiment-id", default=None, help="Defaults to today's date (YYYY-MM-DD).")
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.groq_api_key:
        raise SystemExit("This script requires --groq-api-key or the GROQ_API_KEY environment variable.")

    records = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    LOGGER.info("Loaded %d record(s) from %s", len(records), args.input)

    experiment_id = args.experiment_id or default_experiment_id()
    run_dir = create_experiment_run_dir(args.experiments_root, experiment_id, args.model_slug)
    start_run_meta(
        run_dir,
        model_slug=args.model_slug,
        script="scripts/models/run_rag_cot_groq.py",
        cli_args=vars(args),
        input_path=args.input,
    )
    LOGGER.info("Run directory: %s", run_dir)

    provider = RetryingProvider(GroqLLMProvider(args.groq_api_key), max_attempts=args.max_retries)
    retrieval_config = RetrievalConfig(host=args.qdrant_host, port=args.qdrant_port)
    qdrant_client = create_qdrant_client(retrieval_config)

    results: list[dict] = []
    log = InferenceLogWriter(run_dir)
    results_path = run_dir / RESULTS_FILENAME

    for record in tqdm(records, desc="Step 5: RAG+CoT (Llama-3.3-70B via Groq)"):
        entry: dict = {"index": record.index, "product_smiles": record.product_smiles}
        predicted, raw, predicted_conditions = "", "", None
        try:
            retrieval_result = hybrid_retrieve_reactions_for_smiles(
                record.product_smiles,
                top_k=args.rag_top_k,
                client=qdrant_client,
                config=retrieval_config,
            )
            reactions = retrieval_result.reactions
            entry["retrieved_reactions"] = reactions

            step = generate_single_step(
                GenerationRequest(
                    target_smiles=record.product_smiles,
                    llm_provider=provider,
                    model=args.groq_model,
                    reactions=reactions,
                    temperature=args.temperature,
                )
            )
            entry["attempts"] = [asdict(attempt) for attempt in step.attempts]
            entry["think"] = step.think
            entry["warnings"] = step.warnings
            entry["errors"] = step.errors
            predicted = ".".join(step.precursors) if step.precursors else ""
            raw = step.raw_response

            if record.is_ord:
                condition_prompt = build_condition_prompt(record.product_smiles, predicted, reactions)
                condition_raw = provider.generate(
                    messages=[{"role": "user", "content": condition_prompt}],
                    model=args.groq_model,
                    temperature=0.0,
                    json_mode=False,
                )
                entry["condition_prompt"] = condition_prompt
                entry["condition_raw_response"] = condition_raw
                predicted_conditions = condition_raw
        except Exception as exc:
            LOGGER.exception("Step 5 inference failed for index=%d", record.index)
            entry["error"] = str(exc)

        results.append(
            {
                "index": record.index,
                "product_smiles": record.product_smiles,
                "reactants_smiles": record.reactants_smiles,
                "predicted_reactants_smiles": predicted,
                "raw_response": raw,
                "predicted_conditions": predicted_conditions,
            }
        )
        save_json_atomic(results_path, results)
        log.append(entry)

    finish_run_meta(run_dir, record_count=len(results))
    LOGGER.info("Finished: %d result(s) written to %s", len(results), run_dir)


if __name__ == "__main__":
    main()
