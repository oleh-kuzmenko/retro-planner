#!/usr/bin/env python3
"""Automated Zero-shot vs RAG+CoT evaluation on USPTO-50K (PZ section 4.1-4.2).

For a slice of the USPTO-50K test split, generates ranked single-step
retrosynthesis candidates through any registered `LLMProvider` under one or
both configurations and prints a Table-4.1-style markdown report: Top-1/3/5
exact match plus Structure Success Rate. One call = one retrosynthetic step
(see `planning.generate_single_step`); "top-k" candidates for a target come
from calling the pipeline `k` times, not from one multi-route response.

Example:
    GROQ_API_KEY=... python scripts/evaluate_retrosynthesis.py \\
        --provider groq --limit 25 --modes zero_shot,rag_cot
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from index_uspto50k_to_qdrant import normalize_row

from retro_planner.config import QDRANT_HOST, QDRANT_PORT
from retro_planner.evaluation import (
    format_results_table,
    structure_success_rate,
    top_k_exact_match,
)
from retro_planner.planning import GenerationRequest, generate_single_step
from retro_planner.providers import LLM_PROVIDER_REGISTRY, LLMProvider
from retro_planner.retrieval import RetrievalConfig, create_qdrant_client, hybrid_retrieve_reactions_for_smiles


LOGGER = logging.getLogger("retro_planner.evaluate")

MODE_ZERO_SHOT = "zero_shot"
MODE_RAG_COT = "rag_cot"
ALL_MODES = (MODE_ZERO_SHOT, MODE_RAG_COT)
MODE_LABELS = {MODE_ZERO_SHOT: "Zero-shot", MODE_RAG_COT: "RAG+CoT"}


@dataclass(frozen=True)
class EvalTarget:
    reaction_id: str
    product_smiles: str
    reference_precursors: list[str]


def load_test_targets(dataset_name: str, split: str, limit: int | None) -> list[EvalTarget]:
    """Load (product, reference precursors) pairs from a USPTO-50K split.

    Reuses `normalize_row` from the indexing script so both tools agree on
    dataset-schema handling and SMILES canonicalization.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency 'datasets'. Install it with "
            '`pip install -e ".[indexing]"` or `pip install datasets`.'
        ) from exc

    LOGGER.info("Loading dataset=%s split=%s", dataset_name, split)
    dataset = load_dataset(dataset_name)
    if split not in dataset:
        raise SystemExit(
            f"Split '{split}' not found in {dataset_name}; available splits: {sorted(dataset.keys())}"
        )

    targets: list[EvalTarget] = []
    for idx, row in enumerate(dataset[split]):
        normalized = normalize_row(row, split, idx)
        if normalized is None:
            continue
        reference = [part for part in normalized["reactants_smiles"].split(".") if part]
        if not reference:
            continue
        targets.append(
            EvalTarget(
                reaction_id=normalized["reaction_id"],
                product_smiles=normalized["product_smiles"],
                reference_precursors=reference,
            )
        )
        if limit is not None and len(targets) >= limit:
            break

    LOGGER.info("Loaded %d evaluation targets from %s/%s", len(targets), dataset_name, split)
    return targets


def generate_candidates(
    target: EvalTarget,
    mode: str,
    provider: LLMProvider,
    model: str,
    temperature: float,
    num_candidates: int,
    qdrant_client,
    retrieval_config: RetrievalConfig,
    retrieval_top_k: int,
) -> tuple[list[list[str]], list[str]]:
    """Run the single-step pipeline `num_candidates` times for one target.

    Returns (ranked candidate precursor lists, raw structure predictions for
    Structure Success Rate). RAG context is retrieved once per target and
    reused for every candidate, matching the Streamlit "Generate another
    candidate" behavior.
    """
    reactions: list[dict] | None = None
    if mode == MODE_RAG_COT:
        try:
            result = hybrid_retrieve_reactions_for_smiles(
                target.product_smiles,
                top_k=retrieval_top_k,
                client=qdrant_client,
                config=retrieval_config,
            )
            reactions = result.reactions
        except Exception as exc:
            LOGGER.warning("RAG retrieval failed for target=%s: %s", target.product_smiles, exc)
            reactions = []

    candidates: list[list[str]] = []
    structure_predictions: list[str] = []
    for _ in range(num_candidates):
        step = generate_single_step(
            GenerationRequest(
                target_smiles=target.product_smiles,
                llm_provider=provider,
                model=model,
                reactions=reactions,
                temperature=temperature,
            )
        )
        candidates.append(step.precursors)
        structure_predictions.append(".".join(step.precursors) if step.precursors else "")

    return candidates, structure_predictions


def run_mode(
    mode: str,
    targets: list[EvalTarget],
    provider: LLMProvider,
    model: str,
    temperature: float,
    k_values: list[int],
    qdrant_client,
    retrieval_config: RetrievalConfig,
    retrieval_top_k: int,
) -> dict[str, float]:
    max_k = max(k_values)
    predictions: list[list[list[str]]] = []
    references: list[list[str]] = []
    structure_predictions: list[str] = []

    for i, target in enumerate(targets, start=1):
        LOGGER.info("[%s] %d/%d target=%s", mode, i, len(targets), target.reaction_id)
        candidates, structures = generate_candidates(
            target,
            mode,
            provider,
            model,
            temperature,
            max_k,
            qdrant_client,
            retrieval_config,
            retrieval_top_k,
        )
        predictions.append(candidates)
        references.append(target.reference_precursors)
        structure_predictions.extend(structures)

    metrics: dict[str, float] = {
        f"top_{k}": top_k_exact_match(predictions, references, k) for k in k_values
    }
    metrics["structure_success_rate"] = structure_success_rate(structure_predictions)
    metrics["num_targets"] = len(targets)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Zero-shot vs RAG+CoT single-step retrosynthesis on a "
            "USPTO-50K split (PZ section 4)."
        )
    )
    parser.add_argument("--dataset", default="pingzhili/uspto-50k")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of test targets to evaluate. Use 0 for the full split.",
    )
    parser.add_argument("--k", default="1,3,5", help="Comma-separated top-k values, e.g. 1,3,5.")
    parser.add_argument(
        "--modes",
        default="zero_shot,rag_cot",
        help=f"Comma-separated modes to run, from {ALL_MODES}.",
    )
    parser.add_argument(
        "--provider",
        default="groq",
        choices=sorted(LLM_PROVIDER_REGISTRY),
        help="LLM_PROVIDER_REGISTRY key.",
    )
    parser.add_argument("--model", default=None, help="Defaults to the provider's default_model.")
    parser.add_argument("--api-key", default=None, help="Defaults to the provider's env var.")
    parser.add_argument("--base-url", default=None, help="For OpenAI-compatible/local endpoints.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature; non-zero so repeated calls yield diverse candidates.",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--qdrant-host", default=QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=QDRANT_PORT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Also write the markdown table to this file.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    limit = None if args.limit == 0 else args.limit
    k_values = sorted({int(value) for value in args.k.split(",") if value.strip()})
    if not k_values:
        raise SystemExit("--k must list at least one positive integer.")

    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    unknown_modes = [mode for mode in modes if mode not in ALL_MODES]
    if unknown_modes:
        raise SystemExit(f"Unknown mode(s) {unknown_modes}; choose from {ALL_MODES}.")
    if not modes:
        raise SystemExit("--modes must list at least one mode.")

    provider_config = LLM_PROVIDER_REGISTRY[args.provider]
    api_key = args.api_key or os.getenv(provider_config.api_key_env_var, "")
    if provider_config.api_key_required and not api_key:
        raise SystemExit(
            f"Missing API key for provider '{args.provider}'; pass --api-key or "
            f"set {provider_config.api_key_env_var}."
        )
    base_url = (
        args.base_url
        or (os.getenv(provider_config.base_url_env_var) if provider_config.base_url_env_var else None)
        or provider_config.default_base_url
    )
    provider = provider_config.create_provider(api_key, base_url)
    model = args.model or os.getenv(provider_config.model_env_var) or provider_config.default_model

    targets = load_test_targets(args.dataset, args.split, limit)
    if not targets:
        raise SystemExit("No evaluation targets loaded from the dataset.")

    qdrant_client = None
    retrieval_config = RetrievalConfig(host=args.qdrant_host, port=args.qdrant_port)
    if MODE_RAG_COT in modes:
        qdrant_client = create_qdrant_client(retrieval_config)

    results: dict[str, dict[str, float]] = {}
    for mode in modes:
        started_at = time.perf_counter()
        metrics = run_mode(
            mode,
            targets,
            provider,
            model,
            args.temperature,
            k_values,
            qdrant_client,
            retrieval_config,
            args.retrieval_top_k,
        )
        results[MODE_LABELS[mode]] = metrics
        LOGGER.info(
            "Mode %s finished in %.1fs: %s",
            mode,
            time.perf_counter() - started_at,
            metrics,
        )

    table = format_results_table(results, k_values)
    print(table)
    if args.output:
        args.output.write_text(table + "\n")
        LOGGER.info("Wrote results table to %s", args.output)


if __name__ == "__main__":
    main()
