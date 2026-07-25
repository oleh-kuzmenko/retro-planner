#!/usr/bin/env python3
"""USPTO-50K parsing shared by `build_eval_targets_uspto.py` and `index_uspto_to_qdrant.py`."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Optional

from indexing_common import LOGGER, require_modules

from retro_eval.chemistry import canonicalize_smiles
from retro_eval.reaction_classes import normalize_reaction_class

DEFAULT_DATASET = "pingzhili/uspto-50k"

UNKNOWN_CONDITIONS = {
    "solvents": [],
    "temperature_celsius": None,
    "catalysts": [],
}

# Only the two fields a retrosynthesis test actually needs are stored in
# Qdrant and in eval-target files; USPTO conditions are unknown anyway (see
# UNKNOWN_CONDITIONS above), and reaction_class is dropped too.
USPTO_PAYLOAD_FIELDS = ("product_smiles", "reactants_smiles")


def require_uspto_dependencies() -> None:
    require_modules({"datasets", "tqdm"}, {"datasets": "datasets", "tqdm": "tqdm"})


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


def normalize_row(row: dict, split: str, idx: int) -> Optional[dict]:
    """Normalize a USPTO-50K row into the shared Qdrant payload schema.

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


def iter_uspto_payloads(dataset_name: str = DEFAULT_DATASET) -> Iterator[dict]:
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
