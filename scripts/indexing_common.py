#!/usr/bin/env python3
"""Source-agnostic sampling/IO helpers shared by the ORD and USPTO data builders.

Stratified reservoir sampling, eval-target exclusion, JSONL/JSON writing and RDKit
warning suppression used by `build_train_data_{ord,uspto}.py` and
`build_eval_targets_{ord,uspto}.py` -- the logic that does not depend on which source
is being read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Optional

from retro_eval.chemistry import morgan_vector, reaction_transform_vector


LOGGER = logging.getLogger("retro_eval.indexer")

DEFAULT_EVAL_TARGET_FIELDS = ("product_smiles", "reactants_smiles")


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
    """Drop and recreate a Qdrant collection with Cosine-distance ANN indexing.

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
    exclude_product_smiles: Optional[set[str]] = None,
) -> tuple[int, int]:
    """Batch-embed and upsert `payloads` into both Qdrant collections.

    `payload_fields`, if given, restricts what gets stored on each Qdrant
    point to that key subset (e.g. just `product_smiles`/`reactants_smiles`
    for USPTO) instead of the full normalized record. `exclude_product_smiles`,
    if given, skips any payload whose (already-canonicalized) `product_smiles`
    is in that set, so held-out evaluation targets never enter the RAG index.
    """
    from qdrant_client.models import PointStruct

    exclude = exclude_product_smiles or set()
    indexed = 0
    skipped = 0
    excluded = 0
    product_points: list[PointStruct] = []
    transform_points: list[PointStruct] = []

    for payload in payloads:
        if limit is not None and indexed >= limit:
            break

        if payload["product_smiles"] in exclude:
            excluded += 1
            continue

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
            LOGGER.info(
                "%s indexed=%s skipped=%s excluded=%s", source_name, indexed, skipped, excluded
            )

    indexed += flush_batch(
        client,
        product_collection,
        transform_collection,
        product_points,
        transform_points,
    )
    LOGGER.info(
        "%s complete: indexed=%s skipped=%s excluded=%s", source_name, indexed, skipped, excluded
    )
    return indexed, skipped


def take_unique_by_product(payloads: Iterator[dict], count: int) -> list[dict]:
    """Collect up to `count` payloads with distinct `product_smiles`.

    `product_smiles` repeats across reactions (the same product made via
    different reactants/conditions), so a plain prefix of the source
    iterator can yield duplicate targets in the eval set. This keeps the
    first occurrence of each product instead, preserving source order.
    """
    seen: set[str] = set()
    targets: list[dict] = []
    for payload in payloads:
        product = payload.get("product_smiles")
        if not product or product in seen:
            continue
        seen.add(product)
        targets.append(payload)
        if len(targets) >= count:
            break
    return targets


def stratified_reservoir_sample(
    payloads: Iterator[dict],
    count: int,
    max_per_group: int,
    group_field: str = "source_dataset",
    seed: int = 0,
) -> list[dict]:
    """Randomly sample up to `count` payloads with distinct `product_smiles`,
    capping how many can come from any single `group_field` value (e.g. one
    ORD source file, which is typically one contributed paper/patent dataset
    and therefore one narrow slice of chemistry -- see PZ analysis: an
    uncapped first-N pull put 71/100 targets on a single Pd/phosphine
    coupling system). The cap forces the sample to span at least
    `ceil(count / max_per_group)` distinct sources.

    Runs in a single streaming pass over `payloads` and exits as soon as
    `count` capped, deduped candidates have been collected, so it does not
    need to read the full corpus. Within each group, kept payloads are
    chosen via reservoir sampling (Algorithm R) so they're a uniform random
    subset of what that group produced rather than just the first ones
    encountered -- except possibly the group that is mid-read when the early
    exit fires, which only reflects what had been seen of it so far.
    """
    rng = random.Random(seed)
    reservoirs: dict[str, list[dict]] = {}
    seen_in_group: dict[str, int] = {}
    seen_products: set[str] = set()
    pooled_size = 0

    for payload in payloads:
        product = payload.get("product_smiles")
        if not product or product in seen_products:
            continue
        seen_products.add(product)

        group = payload.get(group_field) or "unknown"
        reservoir = reservoirs.setdefault(group, [])
        seen_in_group[group] = seen_in_group.get(group, 0) + 1

        if len(reservoir) < max_per_group:
            reservoir.append(payload)
            pooled_size += 1
        else:
            j = rng.randrange(seen_in_group[group])
            if j < max_per_group:
                reservoir[j] = payload

        if pooled_size >= count:
            break

    pooled = [payload for reservoir in reservoirs.values() for payload in reservoir]
    if len(pooled) > count:
        pooled = rng.sample(pooled, count)
    else:
        rng.shuffle(pooled)

    LOGGER.info(
        "Sampled %d target(s) from %d source group(s) (max %d per group).",
        len(pooled),
        len(reservoirs),
        max_per_group,
    )
    return pooled


def write_eval_targets(
    path: Path,
    targets: list[dict],
    fields: Iterable[str] = DEFAULT_EVAL_TARGET_FIELDS,
) -> None:
    """Write held-out eval targets, keeping only `fields` from each record."""
    minimal_targets = [
        {key: target[key] for key in fields if key in target} for target in targets
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(minimal_targets, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Wrote %d evaluation target(s) to %s", len(minimal_targets), path)


def load_excluded_product_smiles(path: Path) -> set[str]:
    """Load `product_smiles` from an eval-targets file so indexing can skip them.

    Returns an empty set (with a warning) if `path` doesn't exist yet -- the
    eval-targets file is optional input, produced by a separate
    `build_eval_targets_*.py` run that may not have happened.
    """
    if not path.exists():
        LOGGER.warning(
            "Eval-targets file %s not found; indexing will not exclude any "
            "held-out targets. Run build_eval_targets_uspto.py/build_eval_targets_ord.py "
            "first if you want them excluded.",
            path,
        )
        return set()

    targets = json.loads(path.read_text(encoding="utf-8"))
    excluded = {target["product_smiles"] for target in targets if target.get("product_smiles")}
    LOGGER.info("Loaded %d product(s) to exclude from indexing, from %s", len(excluded), path)
    return excluded


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
        help="Indexed reaction limit. Use --limit 0 to index all remaining reactions.",
    )
    parser.add_argument(
        "--eval-targets-file",
        type=Path,
        default=Path(default_eval_targets_file),
        help=(
            "Eval-targets JSON produced by build_eval_targets_*.py. Any reaction whose "
            "product_smiles appears here is excluded from indexing. Pass --no-exclude-"
            "eval-targets to index everything regardless."
        ),
    )
    parser.add_argument(
        "--no-exclude-eval-targets",
        action="store_true",
        help="Index every reaction, ignoring --eval-targets-file entirely.",
    )
