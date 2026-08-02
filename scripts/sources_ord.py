#!/usr/bin/env python3
"""ORD parsing shared by `build_eval_targets_ord.py` and `build_train_data_ord.py`."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

from indexing_common import LOGGER, require_modules

from retro_eval.chemistry import canonicalize_smiles

DEFAULT_ORD_REPO_ID = "Open-Reaction-Database/ord-data"

# Unlike USPTO-50K, ORD protobuf records genuinely carry reaction conditions
# and a native reaction id, so those are kept alongside product/reactants.
# Fields ORD never populates (reaction_class, pressure_atm, reaction_time_hours)
# and pure indexing bookkeeping (split, source, source_dataset, the legacy
# singular `reactant_smiles` duplicate, the derivable `reaction_smiles`) are
# left out.
ORD_PAYLOAD_FIELDS = (
    "reaction_id",
    "product_smiles",
    "reactants_smiles",
    "solvent",
    "temperature_celsius",
    "catalyst",
    "yield_percent",
)


def require_ord_dependencies(ord_data_dir: Path | None) -> None:
    required_modules = {"tqdm", "ord_schema", "google.protobuf"}
    if ord_data_dir is None:
        required_modules.add("huggingface_hub")
    require_modules(
        required_modules,
        {
            "tqdm": "tqdm",
            "ord_schema": "ord-schema",
            "google.protobuf": "protobuf",
            "huggingface_hub": "huggingface_hub",
        },
    )


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


NON_REACTANT_ROLES = ("SOLVENT", "CATALYST", "REAGENT", "WORKUP", "INTERNAL_STANDARD", "AUTHENTIC_STANDARD")


def normalize_ord_reaction(reaction: dict, dataset_name: str, idx: int) -> Optional[dict]:
    reactants: list[str] = []

    for compound in iter_input_compounds(reaction):
        role = compound.get("reaction_role")
        if role_matches(role, NON_REACTANT_ROLES):
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


def iter_payloads_from_ord_file(file_path: Path) -> Iterator[dict]:
    from google.protobuf.json_format import MessageToDict
    from ord_schema.message_helpers import load_message
    from ord_schema.proto import dataset_pb2
    from tqdm import tqdm

    LOGGER.info("Loading ORD file: %s", file_path)
    dataset = load_message(str(file_path), dataset_pb2.Dataset)
    for idx, reaction in enumerate(tqdm(dataset.reactions, desc=f"ORD {file_path.name}")):
        reaction_dict = MessageToDict(
            reaction,
            preserving_proto_field_name=True,
            use_integers_for_enums=False,
        )
        normalized = normalize_ord_reaction(reaction_dict, file_path.name, idx)
        if normalized is not None:
            yield normalized


def iter_ord_payloads(
    ord_data_dir: Path, max_per_source: int | None = None
) -> Iterator[dict]:
    """Yield normalized ORD reactions, one file at a time.

    `max_per_source`, if given, stops reading a file once it has yielded that
    many reactions from it and moves on to the next one. ORD source files
    vary hugely in size (a few reactions to 50k+ in the current snapshot), so
    an unbounded read -- or a plain global `--limit` applied afterwards --
    lets whichever large file(s) happen to sort first dominate the result.
    Capping per file keeps every contributed source represented. This is a
    cheap first-N-per-file cap, not a random sample of each file: it skips
    the expensive per-reaction normalization once a file's cap is hit
    instead of reading the file in full like `stratified_reservoir_sample`
    would need to for a uniform-random pick.
    """
    files = iter_ord_files(ord_data_dir)
    LOGGER.info("Processing ORD protobuf files: %s", len(files))

    for file_path in files:
        if max_per_source is None:
            yield from iter_payloads_from_ord_file(file_path)
            continue

        count = 0
        for payload in iter_payloads_from_ord_file(file_path):
            if count >= max_per_source:
                break
            count += 1
            yield payload


def resolve_ord_data_dir(
    ord_data_dir: Path | None, repo_id: str, allow_patterns: list[str] | None
) -> Path:
    return ord_data_dir or download_ord_data(repo_id, allow_patterns)
