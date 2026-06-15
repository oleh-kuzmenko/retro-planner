#!/usr/bin/env python3

from __future__ import annotations

import argparse
import uuid
from typing import Optional

from retro_planner.chemistry import (
    canonicalize_smiles,
    morgan_vector,
    reaction_transform_vector,
)
from retro_planner.config import (
    PRODUCT_COLLECTION_NAME,
    TRANSFORM_COLLECTION_NAME,
    VECTOR_SIZE,
)
from retro_planner.reaction_classes import normalize_reaction_class


COLLECTION_NAME = PRODUCT_COLLECTION_NAME


def normalize_row(row: dict, split: str, idx: int) -> Optional[dict]:
    """
    Support common USPTO-50K schemas:
    - reaction_smiles / reaction / rxn_smiles
    - reactants / product
    - reactants_smiles / product_smiles
    """
    reaction_smiles = (
        row.get("reaction_smiles")
        or row.get("reaction")
        or row.get("rxn_smiles")
    )

    reactants = (
        row.get("reactants_smiles")
        or row.get("reactants")
        or row.get("source")
    )

    product = (
        row.get("product_smiles")
        or row.get("product")
        or row.get("target")
    )

    if reaction_smiles and ">>" in reaction_smiles:
        left, right = reaction_smiles.split(">>", maxsplit=1)
        reactants = reactants or left
        product = product or right

    if not product or not reactants:
        return None

    product_canonical = canonicalize_smiles(product)
    if not product_canonical:
        return None

    reaction_id = (
        row.get("reaction_id")
        or row.get("id")
        or f"uspto50k_{split}_{idx}"
    )

    reaction_class = (
        row.get("class")
        or row.get("reaction_class")
        or row.get("label")
    )

    return {
        "reaction_id": str(reaction_id),
        "split": split,
        "reaction_class": str(reaction_class) if reaction_class is not None else None,
        "reaction_class_normalized": normalize_reaction_class(reaction_class),
        "reactants_smiles": reactants,
        "product_smiles": product_canonical,
        "reaction_smiles": reaction_smiles or f"{reactants}>>{product_canonical}",
        "source": "pingzhili/uspto-50k",
        "solvent": None,
        "temperature_celsius": None,
        "pressure_atm": None,
        "reaction_time_hours": None,
        "yield_percent": None,
    }


def recreate_collection(client: QdrantClient, collection_name: str) -> None:
    from qdrant_client.models import Distance, VectorParams

    collections = client.get_collections().collections
    exists = any(collection.name == collection_name for collection in collections)

    if exists:
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    from qdrant_client.models import Distance, VectorParams

    collections = client.get_collections().collections
    exists = any(collection.name == collection_name for collection in collections)

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )


def flush_batch(
    client: QdrantClient,
    collection_name: str,
    points: list[PointStruct],
) -> None:
    if not points:
        return

    client.upsert(
        collection_name=collection_name,
        points=points,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pingzhili/uspto-50k")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--transform-collection", default=TRANSFORM_COLLECTION_NAME)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def main() -> None:
    from datasets import load_dataset
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
    from tqdm import tqdm

    args = parse_args()
    client = QdrantClient(host=args.host, port=args.port)

    if args.recreate:
        recreate_collection(client, args.collection)
        recreate_collection(client, args.transform_collection)
    else:
        ensure_collection(client, args.collection)
        ensure_collection(client, args.transform_collection)

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)

    total_indexed = 0
    total_skipped = 0
    product_points: list[PointStruct] = []
    transform_points: list[PointStruct] = []

    for split_name, split_data in dataset.items():
        print(f"Processing split: {split_name}, rows: {len(split_data)}")

        for idx, row in enumerate(tqdm(split_data)):
            if args.limit is not None and total_indexed >= args.limit:
                break

            normalized = normalize_row(row, split_name, idx)
            if normalized is None:
                total_skipped += 1
                continue

            product_vector = morgan_vector(normalized["product_smiles"])
            if product_vector is None:
                total_skipped += 1
                continue

            transform_vector = reaction_transform_vector(
                normalized["product_smiles"],
                normalized["reactants_smiles"],
            )
            if transform_vector is None:
                total_skipped += 1
                continue

            point_id = str(uuid.uuid4())
            product_points.append(
                PointStruct(
                    id=point_id,
                    vector=product_vector,
                    payload=normalized,
                )
            )
            transform_points.append(
                PointStruct(
                    id=point_id,
                    vector=transform_vector,
                    payload=normalized,
                )
            )

            if len(product_points) >= args.batch_size:
                flush_batch(client, args.collection, product_points)
                flush_batch(client, args.transform_collection, transform_points)
                total_indexed += len(product_points)
                product_points.clear()
                transform_points.clear()

        if args.limit is not None and total_indexed >= args.limit:
            break

    if product_points:
        flush_batch(client, args.collection, product_points)
        flush_batch(client, args.transform_collection, transform_points)
        total_indexed += len(product_points)

    print("Done.")
    print(f"Indexed: {total_indexed}")
    print(f"Skipped: {total_skipped}")
    print(f"Product collection: {args.collection}")
    print(f"Transform collection: {args.transform_collection}")


if __name__ == "__main__":
    main()
