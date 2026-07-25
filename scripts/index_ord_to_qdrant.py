#!/usr/bin/env python3
"""Populate the Qdrant RAG collections from ORD, excluding held-out eval targets.

Run `build_eval_targets_ord.py` first to create the eval-targets file this
script excludes by default.

Example:
    python scripts/build_eval_targets_ord.py
    python scripts/index_ord_to_qdrant.py --limit 1000
    python scripts/index_ord_to_qdrant.py --ord-data-dir /path/to/ord-data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from indexing_common import (
    LOGGER,
    add_common_index_args,
    index_payloads,
    load_excluded_product_smiles,
    recreate_collection,
    suppress_rdkit_warnings,
)
from sources_ord import (
    DEFAULT_ORD_REPO_ID,
    ORD_PAYLOAD_FIELDS,
    iter_ord_payloads,
    require_ord_dependencies,
    resolve_ord_data_dir,
)

from retro_eval.config import PRODUCT_COLLECTION_NAME, TRANSFORM_COLLECTION_NAME


DEFAULT_INDEX_LIMIT = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Qdrant RAG collections from Open Reaction Database (ORD) reactions."
    )
    add_common_index_args(
        parser,
        default_collection=PRODUCT_COLLECTION_NAME,
        default_transform_collection=TRANSFORM_COLLECTION_NAME,
        default_limit=DEFAULT_INDEX_LIMIT,
        default_eval_targets_file="data/ord_eval_targets.json",
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
        "--max-per-source",
        type=int,
        default=None,
        help=(
            "Cap reactions read from any single ORD source file before moving to "
            "the next one, so --limit doesn't just land on whichever large file(s) "
            "sort first. Unset means no cap (read every file in full, subject to "
            "--limit)."
        ),
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        args.limit = None
    return args


def main() -> None:
    from qdrant_client import QdrantClient

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    require_ord_dependencies(args.ord_data_dir)
    suppress_rdkit_warnings()

    exclude = (
        set()
        if args.no_exclude_eval_targets
        else load_excluded_product_smiles(args.eval_targets_file)
    )

    client = QdrantClient(host=args.host, port=args.port)
    recreate_collection(client, args.collection)
    recreate_collection(client, args.transform_collection)

    ord_data_dir = resolve_ord_data_dir(args.ord_data_dir, args.ord_repo_id, args.ord_allow_pattern)

    indexed, skipped = index_payloads(
        client=client,
        payloads=iter_ord_payloads(ord_data_dir, max_per_source=args.max_per_source),
        product_collection=args.collection,
        transform_collection=args.transform_collection,
        batch_size=args.batch_size,
        limit=args.limit,
        source_name="ORD",
        payload_fields=ORD_PAYLOAD_FIELDS,
        exclude_product_smiles=exclude,
    )

    LOGGER.info("Done.")
    LOGGER.info("Indexed: %s", indexed)
    LOGGER.info("Skipped (unparseable): %s", skipped)
    LOGGER.info("Excluded (held-out eval targets): %s", len(exclude))
    LOGGER.info("Product collection: %s", args.collection)
    LOGGER.info("Transform collection: %s", args.transform_collection)


if __name__ == "__main__":
    main()
