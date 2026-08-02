#!/usr/bin/env python3
"""ORD: write a fixed JSON test set for Model 1, held out from training.

`build_train_data_ord.py` excludes every product in this file from the training
pool by default, so these targets stay unseen.

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
import json
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
    parser.add_argument(
        "--exclude-file",
        action="append",
        default=None,
        help=(
            "Product_smiles in this file are excluded from sampling -- e.g. an "
            "existing eval-targets JSON or a training pool's reactants_train.jsonl/"
            "reactants_val.jsonl. Repeatable, so a new (larger) eval set can be built "
            "leak-free against every training pool sampled so far, not just one."
        ),
    )
    return parser.parse_args()


def load_excluded_products(path: Path) -> set[str]:
    """Load `product_smiles` from either an eval-targets JSON array or a
    reactants_{train,val}.jsonl pool file (one JSON object per line)."""
    text = path.read_text(encoding="utf-8")
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    excluded = {r["product_smiles"] for r in records if r.get("product_smiles")}
    LOGGER.info("Loaded %d product(s) to exclude from %s", len(excluded), path)
    return excluded


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    require_ord_dependencies(args.ord_data_dir)
    suppress_rdkit_warnings()

    ord_data_dir = resolve_ord_data_dir(args.ord_data_dir, args.ord_repo_id, args.ord_allow_pattern)
    max_per_source = args.max_per_source or max(1, args.count // 10)

    excluded: set[str] = set()
    for exclude_path in args.exclude_file or []:
        excluded |= load_excluded_products(Path(exclude_path))
    LOGGER.info("Total excluded product(s): %d", len(excluded))

    def excluded_filtered_payloads():
        for payload in iter_ord_payloads(ord_data_dir):
            if payload.get("product_smiles") in excluded:
                continue
            yield payload

    targets = stratified_reservoir_sample(
        excluded_filtered_payloads(),
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
