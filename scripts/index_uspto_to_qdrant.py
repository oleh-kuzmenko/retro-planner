#!/usr/bin/env python3
"""Populate the Qdrant RAG collections from USPTO-50K, excluding held-out eval targets.

Run `build_eval_targets_uspto.py` first to create the eval-targets file this
script excludes by default.

Example:
    python scripts/build_eval_targets_uspto.py
    python scripts/index_uspto_to_qdrant.py --limit 1000
"""

from __future__ import annotations

import argparse
import logging

from indexing_common import (
    LOGGER,
    add_common_index_args,
    index_payloads,
    load_excluded_product_smiles,
    recreate_collection,
    suppress_rdkit_warnings,
)
from sources_uspto import (
    DEFAULT_DATASET,
    USPTO_PAYLOAD_FIELDS,
    iter_uspto_payloads,
    require_uspto_dependencies,
)

from retro_eval.config import PRODUCT_COLLECTION_NAME, TRANSFORM_COLLECTION_NAME


DEFAULT_INDEX_LIMIT = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Qdrant RAG collections from USPTO-50K reactions."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    add_common_index_args(
        parser,
        default_collection=PRODUCT_COLLECTION_NAME,
        default_transform_collection=TRANSFORM_COLLECTION_NAME,
        default_limit=DEFAULT_INDEX_LIMIT,
        default_eval_targets_file="data/uspto_eval_targets.json",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        args.limit = None
    return args


def main() -> None:
    from qdrant_client import QdrantClient

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    require_uspto_dependencies()
    suppress_rdkit_warnings()

    exclude = (
        set()
        if args.no_exclude_eval_targets
        else load_excluded_product_smiles(args.eval_targets_file)
    )

    client = QdrantClient(host=args.host, port=args.port)
    recreate_collection(client, args.collection)
    recreate_collection(client, args.transform_collection)

    indexed, skipped = index_payloads(
        client=client,
        payloads=iter_uspto_payloads(args.dataset),
        product_collection=args.collection,
        transform_collection=args.transform_collection,
        batch_size=args.batch_size,
        limit=args.limit,
        source_name="USPTO",
        payload_fields=USPTO_PAYLOAD_FIELDS,
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
