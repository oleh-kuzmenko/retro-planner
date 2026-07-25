#!/usr/bin/env python3
"""Shared Qdrant indexing helpers for the per-source indexing CLIs.

`index_uspto_to_qdrant.py` and `index_ord_to_qdrant.py` each build the same
two Qdrant collections (product Morgan fingerprints + reaction-transform
fingerprints) from one dataset. This module holds the logic that does not
depend on which source is being read: optional-dependency checks, collection
management, batched upserts, and holding out a slice of target molecules for
evaluation instead of indexing them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Optional

from retro_eval.chemistry import morgan_vector, reaction_transform_vector


LOGGER = logging.getLogger("retro_eval.indexer")
DEFAULT_EVAL_TARGETS_COUNT = 100


def optional_dependency_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def require_modules(required_modules: set[str], package_by_module: dict[str, str]) -> None:
    missing = sorted(
        module for module in required_modules if not optional_dependency_available(module)
    )
    if not missing:
        return

    packages = sorted({package_by_module[module] for module in missing})
    missing_modules = ", ".join(missing)
    missing_packages = ", ".join(packages)
    raise SystemExit(
        "Missing optional indexing dependencies: "
        f"{missing_modules}. Install them with `pip install -e \".[indexing]\"` "
        f"or install the package(s) directly: {missing_packages}."
    )


def suppress_rdkit_warnings() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")


def recreate_collection(client, collection_name: str) -> None:
    """(Re)create a Qdrant collection with Cosine-distance ANN indexing.

    Cosine here is only used by Qdrant to shortlist nearby candidates
    quickly; it is not the similarity score shown to the user or used for
    final ranking. `retrieval.py` rescores the shortlist with an exact
    Tanimoto coefficient over the raw fingerprint vectors, per PZ section 3.2.
    """
    from qdrant_client.models import Distance, VectorParams

    from retro_eval.config import VECTOR_SIZE

    collections = client.get_collections().collections
    exists = any(collection.name == collection_name for collection in collections)

    if exists:
        LOGGER.info("Dropping existing Qdrant collection: %s", collection_name)
        client.delete_collection(collection_name)

    LOGGER.info("Creating Qdrant collection: %s", collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )


def flush_batch(
    client,
    product_collection: str,
    transform_collection: str,
    product_points: list,
    transform_points: list,
) -> int:
    if not product_points:
        return 0

    client.upsert(collection_name=product_collection, points=product_points)
    client.upsert(collection_name=transform_collection, points=transform_points)
    flushed = len(product_points)
    product_points.clear()
    transform_points.clear()
    return flushed


def index_payloads(
    client,
    payloads: Iterable[dict],
    product_collection: str,
    transform_collection: str,
    batch_size: int,
    limit: Optional[int],
    source_name: str,
    payload_fields: Optional[Iterable[str]] = None,
) -> tuple[int, int]:
    """Batch-embed and upsert `payloads` into both Qdrant collections.

    `payload_fields`, if given, restricts what gets stored on each Qdrant
    point to that key subset (e.g. just `product_smiles`/`reactants_smiles`
    for USPTO) instead of the full normalized record, so fields the app
    never reads aren't persisted alongside the vectors. `product_smiles`
    and `reactants_smiles` are still required on every payload to compute
    the fingerprints, regardless of what gets stored.
    """
    from qdrant_client.models import PointStruct

    indexed = 0
    skipped = 0
    product_points: list[PointStruct] = []
    transform_points: list[PointStruct] = []

    for payload in payloads:
        if limit is not None and indexed >= limit:
            break

        product_vector = morgan_vector(payload["product_smiles"])
        transform_vector = reaction_transform_vector(
            payload["product_smiles"],
            payload["reactants_smiles"],
        )
        if product_vector is None or transform_vector is None:
            skipped += 1
            continue

        stored_payload = (
            {key: payload[key] for key in payload_fields if key in payload}
            if payload_fields is not None
            else payload
        )

        point_id = str(uuid.uuid4())
        product_points.append(
            PointStruct(id=point_id, vector=product_vector, payload=stored_payload)
        )
        transform_points.append(
            PointStruct(id=point_id, vector=transform_vector, payload=stored_payload)
        )

        if len(product_points) >= batch_size:
            indexed += flush_batch(
                client,
                product_collection,
                transform_collection,
                product_points,
                transform_points,
            )
            LOGGER.info("%s indexed=%s skipped=%s", source_name, indexed, skipped)

    indexed += flush_batch(
        client,
        product_collection,
        transform_collection,
        product_points,
        transform_points,
    )
    LOGGER.info("%s complete: indexed=%s skipped=%s", source_name, indexed, skipped)
    return indexed, skipped


def split_eval_targets(
    payloads: Iterator[dict], count: int
) -> tuple[list[dict], Iterator[dict]]:
    """Peel the first `count` payloads off `payloads` as a held-out eval set.

    The returned list is consumed eagerly (so it can be written to disk
    before indexing starts); the returned iterator resumes exactly where the
    hold-out left off, so those payloads are never passed to `index_payloads`
    and can't leak into the RAG index they're meant to be evaluated against.
    """
    iterator = iter(payloads)
    held_out: list[dict] = []
    for _ in range(max(count, 0)):
        try:
            held_out.append(next(iterator))
        except StopIteration:
            break
    return held_out, iterator


DEFAULT_EVAL_TARGET_FIELDS = ("product_smiles", "reactants_smiles")


def write_eval_targets(
    path: Path,
    targets: list[dict],
    fields: Iterable[str] = DEFAULT_EVAL_TARGET_FIELDS,
) -> None:
    """Write held-out targets, keeping only `fields` from each record.

    Mirrors the `payload_fields` filtering `index_payloads` applies to
    Qdrant points, so the eval-target file only carries what's actually
    used for testing instead of the full indexing payload.
    """
    minimal_targets = [
        {key: target[key] for key in fields if key in target} for target in targets
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(minimal_targets, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Wrote %d held-out evaluation target(s) to %s", len(minimal_targets), path)


def add_common_index_args(
    parser: argparse.ArgumentParser,
    *,
    default_collection: str,
    default_transform_collection: str,
    default_limit: int,
    default_eval_targets_file: str,
) -> None:
    parser.add_argument("--collection", default=default_collection)
    parser.add_argument("--transform-collection", default=default_transform_collection)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--limit",
        type=int,
        default=default_limit,
        help=(
            "Indexed reaction limit, counted after the --eval-targets-count "
            "hold-out has been set aside. Use --limit 0 to index all remaining "
            "reactions."
        ),
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Accepted for backwards compatibility; collections are always recreated.",
    )
    parser.add_argument(
        "--eval-targets-count",
        type=int,
        default=DEFAULT_EVAL_TARGETS_COUNT,
        help=(
            "Number of target molecules to set aside into --eval-targets-file "
            "instead of indexing them, so they stay unseen for evaluation. "
            "Use 0 to disable (index everything up to --limit)."
        ),
    )
    parser.add_argument(
        "--eval-targets-file",
        type=Path,
        default=Path(default_eval_targets_file),
        help="Where to write the held-out evaluation targets as JSON.",
    )
