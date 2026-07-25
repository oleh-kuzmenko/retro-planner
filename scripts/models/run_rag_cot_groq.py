#!/usr/bin/env python3
"""Step 5: Qdrant hybrid RAG retrieval + Chain-of-Thought via Groq's GPT-OSS-120B.

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

Groq's free tier returns HTTP 429 once its per-minute/per-day quota is used
up. Short rate limits (<= --rate-limit-auto-wait-seconds) are slept through
automatically by `RetryingProvider`. Longer ones (e.g. a daily quota reset,
which Groq may report as hours away) instead pause the whole run: progress
already written to `results.json` is kept, a resume pointer is saved under
`<experiments-root>/.resume/`, and the script exits so it isn't left
blocking a terminal for hours. Re-running the exact same command later picks
the same experiment run back up and skips every record already completed.
Pass --fresh to ignore any saved resume pointer and start a new run.

Example:
    pip install -e ".[eval-runner]"
    docker compose up -d qdrant
    GROQ_API_KEY=... python scripts/models/run_rag_cot_groq.py --input data/ord_eval_targets.json --limit 10
    # If it pauses on a Groq rate limit, just rerun the same command later:
    GROQ_API_KEY=... python scripts/models/run_rag_cot_groq.py --input data/ord_eval_targets.json --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
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
from retro_eval.harness.records import EvalRecord, load_records
from retro_eval.planning import GenerationRequest, generate_single_step
from retro_eval.providers.chat_api import GroqLLMProvider
from retro_eval.providers.retrying import ProviderPaused, RetryingProvider
from retro_eval.retrieval import RetrievalConfig, create_qdrant_client, hybrid_retrieve_reactions_for_smiles
from tqdm import tqdm

LOGGER = logging.getLogger("retro_eval.run_rag_cot_groq")


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
    parser = argparse.ArgumentParser(description="Step 5: Qdrant hybrid RAG + CoT via Groq's GPT-OSS-120B.")
    parser.add_argument("--input", type=Path, required=True, help="USPTO/ORD-format dataset JSON.")
    parser.add_argument(
        "--model-slug", default="rag_cot_gptoss120b", help="Folder name under experiments/<experiment_id>/."
    )
    parser.add_argument("--groq-api-key", default=os.getenv("GROQ_API_KEY", ""))
    parser.add_argument("--groq-model", default="openai/gpt-oss-120b")
    parser.add_argument("--qdrant-host", default=QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=QDRANT_PORT)
    parser.add_argument("--rag-top-k", type=int, default=3)
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
        help="Sleep through a Groq 429 automatically if its suggested wait is at most this long.",
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

    if not args.groq_api_key:
        raise SystemExit("This script requires --groq-api-key or the GROQ_API_KEY environment variable.")

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
        script="scripts/models/run_rag_cot_groq.py",
        cli_args=vars(args),
        input_path=args.input,
    )
    LOGGER.info("Run directory: %s", run_dir)

    provider = RetryingProvider(
        GroqLLMProvider(args.groq_api_key),
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

    for record in tqdm(records, desc="Step 5: RAG+CoT (GPT-OSS-120B via Groq)"):
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
        except ProviderPaused as exc:
            LOGGER.warning(
                "Groq rate limit requires a long pause (~%s) at index=%d. "
                "Progress so far is saved in %s -- rerun the exact same command "
                "later to resume (it will skip everything already completed).",
                format_duration(exc.wait_seconds),
                record.index,
                run_dir,
            )
            finish_run_meta(run_dir, record_count=len(results), status="paused_rate_limited")
            raise SystemExit(
                f"Paused on a Groq rate limit at index={record.index}; "
                f"suggested wait ~{format_duration(exc.wait_seconds)}. "
                "Rerun the same command later to resume."
            ) from exc
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
    pointer_path.unlink(missing_ok=True)
    LOGGER.info("Finished: %d result(s) written to %s", len(results), run_dir)


if __name__ == "__main__":
    main()
