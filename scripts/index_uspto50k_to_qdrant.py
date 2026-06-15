#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import logging
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Optional

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


LOGGER = logging.getLogger("retro_planner.indexer")
COLLECTION_NAME = PRODUCT_COLLECTION_NAME
DEFAULT_INDEX_LIMIT = 500_000
ORD_DATA_REPO_ID = "Open-Reaction-Database/ord-data"
UNKNOWN_CONDITIONS = {
    "solvents": [],
    "temperature_celsius": None,
    "catalysts": [],
}

OPTIONAL_DEPENDENCIES = {
    "datasets": "datasets",
    "huggingface_hub": "huggingface_hub",
    "ord_schema": "ord-schema",
    "google.protobuf": "protobuf",
    "tqdm": "tqdm",
}


def optional_dependency_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def require_optional_dependencies(
    sources: Iterable[str],
    ord_data_dir: Path | None,
) -> None:
    required_modules: set[str] = {"tqdm"}
    source_set = set(sources)

    if "uspto" in source_set:
        required_modules.add("datasets")
    if "ord" in source_set:
        required_modules.update({"ord_schema", "google.protobuf"})
        if ord_data_dir is None:
            required_modules.add("huggingface_hub")

    missing = sorted(
        module
        for module in required_modules
        if not optional_dependency_available(module)
    )
    if not missing:
        return

    packages = sorted({OPTIONAL_DEPENDENCIES[module] for module in missing})
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


def normalize_row(row: dict, split: str, idx: int) -> Optional[dict]:
    """
    Normalize a USPTO-50K row into the shared Qdrant payload schema.

    Supported source schemas include:
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

    reactants_canonical = canonicalize_reaction_side(reactants) or str(reactants)
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
        "reactants_smiles": reactants_canonical,
        "reactant_smiles": reactants_canonical,
        "product_smiles": product_canonical,
        "reaction_smiles": reaction_smiles or f"{reactants_canonical}>>{product_canonical}",
        "conditions": UNKNOWN_CONDITIONS.copy(),
        "source": "USPTO",
        "source_dataset": "pingzhili/uspto-50k",
        "solvent": None,
        "temperature_celsius": None,
        "pressure_atm": None,
        "reaction_time_hours": None,
        "yield_percent": None,
    }


def canonicalize_reaction_side(smiles: str | Iterable[str] | None) -> Optional[str]:
    if smiles is None:
        return None

    parts = smiles.split(".") if isinstance(smiles, str) else list(smiles)
    canonical_parts = []
    for part in parts:
        canonical = canonicalize_smiles(str(part).strip())
        if canonical:
            canonical_parts.append(canonical)

    if not canonical_parts:
        return None
    return ".".join(canonical_parts)


def recreate_collection(client, collection_name: str) -> None:
    from qdrant_client.models import Distance, VectorParams

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


def find_first_value(data: Any, keys: set[str]) -> Any:
    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys and value not in (None, "", []):
                return value
        for value in data.values():
            found = find_first_value(value, keys)
            if found not in (None, "", []):
                return found
    elif isinstance(data, list):
        for value in data:
            found = find_first_value(value, keys)
            if found not in (None, "", []):
                return found
    return None


def numeric_value(data: Any) -> Optional[float]:
    if isinstance(data, (int, float)):
        return float(data)
    if isinstance(data, str):
        try:
            return float(data)
        except ValueError:
            return None
    if isinstance(data, dict):
        for key in ("value", "amount", "mean", "setpoint", "lower", "upper"):
            value = numeric_value(data.get(key))
            if value is not None:
                return value
    return None


def measurement_unit(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    unit = data.get("units") or data.get("unit")
    if unit:
        return str(unit).lower()

    for value in data.values():
        nested_unit = measurement_unit(value)
        if nested_unit:
            return nested_unit
    return None


def temperature_to_celsius(temperature: Any) -> Optional[float]:
    value = numeric_value(temperature)
    if value is None:
        return None

    unit = measurement_unit(temperature)
    if unit and "kelvin" in unit:
        return round(value - 273.15, 2)
    if unit and "fahrenheit" in unit:
        return round((value - 32.0) * 5.0 / 9.0, 2)
    return value


def role_matches(role: str | None, needles: tuple[str, ...]) -> bool:
    if not role:
        return False
    normalized = role.upper()
    return any(needle in normalized for needle in needles)


def compound_smiles(compound: dict) -> Optional[str]:
    identifiers = compound.get("identifiers") or []
    if isinstance(identifiers, dict):
        identifiers = identifiers.values()
    for identifier in identifiers:
        if not isinstance(identifier, dict):
            continue
        identifier_type = str(identifier.get("type", "")).upper()
        value = identifier.get("value")
        if value and "SMILES" in identifier_type:
            return str(value)

    for key in ("smiles", "canonical_smiles"):
        value = compound.get(key)
        if value:
            return str(value)
    return None


def compound_name(compound: dict) -> Optional[str]:
    identifiers = compound.get("identifiers") or []
    if isinstance(identifiers, dict):
        identifiers = identifiers.values()
    for identifier in identifiers:
        if not isinstance(identifier, dict):
            continue
        value = identifier.get("value")
        if value:
            return str(value)
    return compound.get("name")


def iter_input_compounds(reaction: dict) -> Iterator[dict]:
    inputs = reaction.get("inputs") or {}
    values = inputs.values() if isinstance(inputs, dict) else inputs
    for reaction_input in values:
        if not isinstance(reaction_input, dict):
            continue
        for component in reaction_input.get("components", []):
            if isinstance(component, dict):
                yield component


def iter_product_compounds(reaction: dict) -> Iterator[dict]:
    for outcome in reaction.get("outcomes", []):
        if not isinstance(outcome, dict):
            continue
        for product in outcome.get("products", []):
            if isinstance(product, dict):
                yield product


def extract_yield_percent(product: dict) -> Optional[float]:
    for measurement in product.get("measurements", []):
        measurement_type = str(measurement.get("type", "")).upper()
        if "YIELD" not in measurement_type:
            continue
        percentage = measurement.get("percentage")
        if percentage is None:
            percentage = find_first_value(measurement, {"percentage"})
        return numeric_value(percentage)
    return None


def reaction_identifier(reaction: dict, fallback: str) -> str:
    direct_id = reaction.get("reaction_id") or reaction.get("id")
    if direct_id:
        return str(direct_id)

    identifiers = reaction.get("identifiers") or []
    if isinstance(identifiers, dict):
        identifiers = identifiers.values()

    for identifier in identifiers:
        if not isinstance(identifier, dict):
            continue
        value = identifier.get("value")
        if value:
            return str(value)

    return fallback


def extract_ord_conditions(reaction: dict) -> dict:
    solvents: list[str] = []
    catalysts: list[str] = []

    for compound in iter_input_compounds(reaction):
        role = compound.get("reaction_role")
        name = compound_name(compound)
        smiles = compound_smiles(compound)
        label = smiles or name
        if not label:
            continue
        if role_matches(role, ("SOLVENT",)):
            solvents.append(label)
        elif role_matches(role, ("CATALYST",)):
            catalysts.append(label)

    conditions = reaction.get("conditions") or {}
    temperature = find_first_value(
        conditions,
        {"temperature", "setpoint", "internal_temperature"},
    )

    return {
        "solvents": sorted(set(solvents)),
        "temperature_celsius": temperature_to_celsius(temperature),
        "catalysts": sorted(set(catalysts)),
    }


def normalize_ord_reaction(reaction: dict, dataset_name: str, idx: int) -> Optional[dict]:
    reactants: list[str] = []

    for compound in iter_input_compounds(reaction):
        role = compound.get("reaction_role")
        if role_matches(role, ("SOLVENT", "CATALYST")):
            continue
        smiles = compound_smiles(compound)
        canonical = canonicalize_smiles(smiles)
        if canonical:
            reactants.append(canonical)

    products: list[str] = []
    yield_percent = None
    for product in iter_product_compounds(reaction):
        smiles = compound_smiles(product)
        canonical = canonicalize_smiles(smiles)
        if canonical:
            products.append(canonical)
            yield_percent = yield_percent or extract_yield_percent(product)

    if not reactants or not products:
        return None

    product_smiles = products[0]
    reactants_smiles = ".".join(sorted(set(reactants)))
    reaction_id = reaction_identifier(
        reaction,
        fallback=f"ord_{Path(dataset_name).stem}_{idx}",
    )
    conditions = extract_ord_conditions(reaction)
    solvent = ", ".join(conditions["solvents"]) or None
    catalyst = ", ".join(conditions["catalysts"]) or None

    return {
        "reaction_id": str(reaction_id),
        "split": dataset_name,
        "reaction_class": None,
        "reaction_class_normalized": None,
        "reactants_smiles": reactants_smiles,
        "reactant_smiles": reactants_smiles,
        "product_smiles": product_smiles,
        "reaction_smiles": f"{reactants_smiles}>>{product_smiles}",
        "conditions": conditions,
        "source": "ORD",
        "source_dataset": dataset_name,
        "solvent": solvent,
        "temperature_celsius": conditions["temperature_celsius"],
        "pressure_atm": None,
        "reaction_time_hours": None,
        "yield_percent": yield_percent,
        "catalyst": catalyst,
    }


def iter_uspto_payloads(dataset_name: str) -> Iterator[dict]:
    from datasets import load_dataset
    from tqdm import tqdm

    LOGGER.info("Loading USPTO-50K dataset: %s", dataset_name)
    dataset = load_dataset(dataset_name)

    for split_name, split_data in dataset.items():
        LOGGER.info("Processing USPTO split=%s rows=%s", split_name, len(split_data))
        for idx, row in enumerate(tqdm(split_data, desc=f"USPTO {split_name}")):
            normalized = normalize_row(row, split_name, idx)
            if normalized is not None:
                yield normalized


def download_ord_data(repo_id: str, allow_patterns: list[str] | None) -> Path:
    from huggingface_hub import snapshot_download

    LOGGER.info("Downloading ORD data from Hugging Face repo: %s", repo_id)
    snapshot_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns or ["data/**/*.pb.gz", "data/*.pb.gz"],
    )
    return Path(snapshot_dir)


def iter_ord_files(ord_data_dir: Path) -> list[Path]:
    if ord_data_dir.is_file() and ord_data_dir.name.endswith(".pb.gz"):
        return [ord_data_dir]
    return sorted(ord_data_dir.glob("data/**/*.pb.gz")) or sorted(
        ord_data_dir.glob("**/*.pb.gz")
    )


def iter_ord_payloads(ord_data_dir: Path) -> Iterator[dict]:
    from google.protobuf.json_format import MessageToDict
    from ord_schema.message_helpers import load_message
    from ord_schema.proto import dataset_pb2
    from tqdm import tqdm

    files = iter_ord_files(ord_data_dir)
    LOGGER.info("Processing ORD protobuf files: %s", len(files))

    for file_path in files:
        LOGGER.info("Loading ORD file: %s", file_path)
        dataset = load_message(str(file_path), dataset_pb2.Dataset)
        for idx, reaction in enumerate(
            tqdm(dataset.reactions, desc=f"ORD {file_path.name}")
        ):
            reaction_dict = MessageToDict(
                reaction,
                preserving_proto_field_name=True,
                use_integers_for_enums=False,
            )
            normalized = normalize_ord_reaction(reaction_dict, file_path.name, idx)
            if normalized is not None:
                yield normalized


def index_payloads(
    client,
    payloads: Iterable[dict],
    product_collection: str,
    transform_collection: str,
    batch_size: int,
    limit: int | None,
    source_name: str,
) -> tuple[int, int]:
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

        point_id = str(uuid.uuid4())
        product_points.append(
            PointStruct(id=point_id, vector=product_vector, payload=payload)
        )
        transform_points.append(
            PointStruct(id=point_id, vector=transform_vector, payload=payload)
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


def remaining_index_limit(limit: int | None, indexed_total: int) -> int | None:
    if limit is None:
        return None
    return max(limit - indexed_total, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Qdrant RAG collections from USPTO-50K and ORD."
    )
    parser.add_argument("--dataset", default="pingzhili/uspto-50k")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--transform-collection", default=TRANSFORM_COLLECTION_NAME)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_INDEX_LIMIT,
        help=(
            "Total indexed reaction limit across selected sources. "
            "Use --limit 0 to process all available reactions."
        ),
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Accepted for backwards compatibility; collections are always recreated.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("uspto", "ord"),
        default=["uspto", "ord"],
        help="Data sources to index.",
    )
    parser.add_argument(
        "--ord-data-dir",
        type=Path,
        default=None,
        help="Local ORD data directory or single .pb.gz file. If omitted, download from Hugging Face.",
    )
    parser.add_argument("--ord-repo-id", default=ORD_DATA_REPO_ID)
    parser.add_argument(
        "--ord-allow-pattern",
        action="append",
        default=None,
        help="Hugging Face allow pattern for ORD files, e.g. data/4d/*.pb.gz. Repeatable.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        args.limit = None
    return args


def main() -> None:
    from qdrant_client import QdrantClient

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args()
    require_optional_dependencies(args.sources, args.ord_data_dir)
    suppress_rdkit_warnings()

    client = QdrantClient(host=args.host, port=args.port)

    recreate_collection(client, args.collection)
    recreate_collection(client, args.transform_collection)

    totals = {"indexed": 0, "skipped": 0}

    if "uspto" in args.sources:
        remaining_limit = remaining_index_limit(args.limit, totals["indexed"])
        indexed, skipped = index_payloads(
            client=client,
            payloads=iter_uspto_payloads(args.dataset),
            product_collection=args.collection,
            transform_collection=args.transform_collection,
            batch_size=args.batch_size,
            limit=remaining_limit,
            source_name="USPTO",
        )
        totals["indexed"] += indexed
        totals["skipped"] += skipped

    if (
        "ord" in args.sources
        and remaining_index_limit(args.limit, totals["indexed"]) != 0
    ):
        ord_data_dir = args.ord_data_dir or download_ord_data(
            args.ord_repo_id,
            args.ord_allow_pattern,
        )
        remaining_limit = remaining_index_limit(args.limit, totals["indexed"])
        indexed, skipped = index_payloads(
            client=client,
            payloads=iter_ord_payloads(ord_data_dir),
            product_collection=args.collection,
            transform_collection=args.transform_collection,
            batch_size=args.batch_size,
            limit=remaining_limit,
            source_name="ORD",
        )
        totals["indexed"] += indexed
        totals["skipped"] += skipped

    LOGGER.info("Done.")
    LOGGER.info("Indexed total: %s", totals["indexed"])
    LOGGER.info("Skipped total: %s", totals["skipped"])
    LOGGER.info("Product collection: %s", args.collection)
    LOGGER.info("Transform collection: %s", args.transform_collection)


if __name__ == "__main__":
    main()
