#!/usr/bin/env python3
"""Remove atom maps and canonicalize an evaluation-target JSON file.

Example:
    python scripts/clean_atom_mapped_eval_json.py \
        --input uspto_eval_targets.json \
        --output data/uspto_eval_targets_clean.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem, rdBase

from retro_eval.chemistry import mol_from_smiles_without_atom_maps


SMILES_FIELDS = ("product_smiles", "reactants_smiles")


def canonicalize_without_atom_maps(smiles: str) -> str | None:
    """Return unmapped canonical SMILES, preserving dot-separated fragments."""
    fragments: list[str] = []
    for fragment in smiles.split("."):
        molecule = mol_from_smiles_without_atom_maps(fragment)
        if molecule is None:
            return None
        fragments.append(Chem.MolToSmiles(molecule, canonical=True))
    return ".".join(fragments)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove atom maps and canonicalize product/reactant SMILES in a JSON array."
    )
    parser.add_argument("--input", type=Path, required=True, help="Source evaluation-target JSON file.")
    parser.add_argument("--output", type=Path, required=True, help="Destination cleaned JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        records = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read JSON from {args.input}: {error}") from error

    if not isinstance(records, list):
        raise SystemExit(f"{args.input} must contain a JSON array.")

    cleaned_records: list[dict] = []
    with rdBase.BlockLogs():
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise SystemExit(f"Record {index} must be a JSON object.")

            cleaned = record.copy()
            for field in SMILES_FIELDS:
                value = record.get(field)
                if not isinstance(value, str) or not value:
                    raise SystemExit(f"Record {index} has no usable {field}.")
                canonical = canonicalize_without_atom_maps(value)
                if canonical is None:
                    raise SystemExit(f"Record {index} has invalid {field}: {value!r}")
                cleaned[field] = canonical
            cleaned_records.append(cleaned)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cleaned_records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cleaned_records)} cleaned record(s) to {args.output}")


if __name__ == "__main__":
    main()
