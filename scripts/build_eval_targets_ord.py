#!/usr/bin/env python3
"""Step 1 (ORD): write a fixed JSON test set, held out for every model comparison stage.

Every stage (ReactionT5v2, ChemLLM, Qwen2.5-7B+LoRA, Llama-70B+RAG+CoT) runs
against the exact same file this produces, so `index_ord_to_qdrant.py`
excludes it from the RAG index by default -- these targets stay unseen.

Targets are drawn via `stratified_reservoir_sample`, capped per ORD source
file (`--max-per-source`), rather than the first N unique-product reactions
encountered. Each ORD file is typically one contributed paper/patent
dataset, so an uncapped pull tends to land almost entirely on whichever
file(s) sort first -- one narrow reaction/catalyst family, not a
representative slice of retrosynthesis chemistry.

Example:
    python scripts/build_eval_targets_ord.py
    python scripts/build_eval_targets_ord.py --ord-data-dir /path/to/ord-data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from indexing_common import (
    stratified_reservoir_sample,
    suppress_rdkit_warnings,
    write_eval_targets,
)
from sources_ord import (
    DEFAULT_ORD_REPO_ID,
    ORD_PAYLOAD_FIELDS,
    iter_ord_payloads,
    require_ord_dependencies,
    resolve_ord_data_dir,
)

LOGGER = logging.getLogger("retro_eval.build_eval_targets_ord")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a fixed ORD JSON test set for model comparison."
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/ord_eval_targets.json"))
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help=(
            "Max targets from any single ORD source file (one contributed "
            "dataset). Defaults to max(1, --count // 10), forcing the sample "
            "to span at least ~10 distinct sources."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling, kept fixed so the eval set is reproducible.",
    )
    parser.add_argument(
        "--ord-data-dir",
        type=Path,
        default=None,
        help="Local ORD data directory or single .pb.gz file. If omitted, download from Hugging Face.",
    )
    parser.add_argument("--ord-repo-id", default=DEFAULT_ORD_REPO_ID)
    parser.add_argument(
        "--ord-allow-pattern",
        action="append",
        default=None,
        help="Hugging Face allow pattern for ORD files, e.g. data/4d/*.pb.gz. Repeatable.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    require_ord_dependencies(args.ord_data_dir)
    suppress_rdkit_warnings()

    ord_data_dir = resolve_ord_data_dir(args.ord_data_dir, args.ord_repo_id, args.ord_allow_pattern)
    max_per_source = args.max_per_source or max(1, args.count // 10)
    targets = stratified_reservoir_sample(
        iter_ord_payloads(ord_data_dir),
        args.count,
        max_per_source,
        group_field="source_dataset",
        seed=args.seed,
    )
    if len(targets) < args.count:
        LOGGER.warning("Only found %d target(s); requested %d.", len(targets), args.count)

    write_eval_targets(args.output, targets, fields=ORD_PAYLOAD_FIELDS)


if __name__ == "__main__":
    main()
