#!/usr/bin/env python3
"""Rewrite the canonical USPTO-50K train split into root-aligned form (R-SMILES,
Zhong et al., Chemical Science 2022), using the dataset's own atom maps.

The ORD sibling of this script (`build_root_aligned_data.py`) has to run RXNMapper
to recover an atom mapping first, because ORD SMILES are unmapped. USPTO-50K ships
atom-mapped `reactants`/`product` columns, so the mapping is exact and free here --
no neural mapper, no failure modes from mismapping.

Two modes, selected by `--copies`:

- `--copies 1` (default): one row per reaction. The reactants are rooted at the atom
  the *product's own canonical* SMILES already starts at, and the product is left in
  plain canonical form. The input format is therefore unchanged, so a checkpoint
  trained on this data is used at inference exactly like any other -- same as the ORD
  script's behaviour.
- `--copies N > 1`: the full R-SMILES recipe -- N renderings per reaction, each rooted
  at a different randomly chosen product atom, with *both* sides rooted consistently
  at the mapped counterpart of that atom. The first copy always uses the canonical
  root, so the `--copies 1` output is a subset of it. This changes the input
  distribution as well, which is exactly the point: the model sees many renderings of
  the same reaction and stops memorizing one canonical string. Train with
  `--no-augment`; the online SMILES randomization in `train_reactant_model_ord.py`
  would destroy the alignment this file encodes.

Rows whose mapping cannot be followed (product root atom unmapped, or no reactant
fragment carries that map number) fall back to plain canonical SMILES rather than
being dropped, so the pool size stays predictable.

Example:
    python scripts/build_root_aligned_uspto.py \\
        --output data/v2_uspto_train_rootaligned/reactants_train.jsonl
    python scripts/build_root_aligned_uspto.py --copies 5 --seed 42 \\
        --output data/v2_uspto_train_rsmiles5/reactants_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from indexing_common import load_excluded_product_smiles, suppress_rdkit_warnings
from rdkit import Chem

from retro_eval.chemistry import canonicalize_smiles_without_atom_maps

LOGGER = logging.getLogger("retro_eval.build_root_aligned_uspto")

DEFAULT_DATASET = "bisectgroup/USPTO_50K"


def canonical_root_atom_idx(mol) -> int:
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    return ranks.index(0)


def strip_maps(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def render_pair(reactants_mapped: str, product_mapped: str, root_idx: int | None) -> tuple[str, str] | None:
    """Render one (product, reactants) pair rooted at `root_idx` of the product.

    `root_idx=None` means "use the product's canonical root and leave the product
    string canonical" -- the `--copies 1` behaviour.
    """
    prod_mol = Chem.MolFromSmiles(product_mapped)
    if prod_mol is None:
        return None

    keep_product_canonical = root_idx is None
    if root_idx is None:
        root_idx = canonical_root_atom_idx(prod_mol)

    root_map_num = prod_mol.GetAtomWithIdx(root_idx).GetAtomMapNum()
    if root_map_num == 0:
        return None

    aligned_frag = None
    aligned_idx = None
    other_frags = []
    for frag in reactants_mapped.split("."):
        fmol = Chem.MolFromSmiles(frag)
        if fmol is None:
            other_frags.append(frag)
            continue
        found = next((a.GetIdx() for a in fmol.GetAtoms() if a.GetAtomMapNum() == root_map_num), None)
        if found is not None and aligned_frag is None:
            aligned_frag, aligned_idx = fmol, found
        else:
            other_frags.append(frag)

    if aligned_frag is None:
        return None

    rooted_reactant = Chem.MolToSmiles(strip_maps(aligned_frag), rootedAtAtom=aligned_idx, canonical=True)

    other_clean = []
    for frag in other_frags:
        fmol = Chem.MolFromSmiles(frag)
        other_clean.append(Chem.MolToSmiles(strip_maps(fmol)) if fmol is not None else frag)

    product_out = (
        Chem.MolToSmiles(strip_maps(prod_mol))
        if keep_product_canonical
        else Chem.MolToSmiles(strip_maps(prod_mol), rootedAtAtom=root_idx, canonical=True)
    )
    return product_out, ".".join([rooted_reactant] + other_clean)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--copies", type=int, default=1, help="Renderings per reaction; >1 enables random roots.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=Path("data/v2_uspto_test_holdout.json"),
        help="Products in this eval-targets file are dropped, as in build_train_data_uspto.py.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    suppress_rdkit_warnings()

    from datasets import load_dataset

    excluded = load_excluded_product_smiles(args.exclude_file)
    rows = load_dataset(args.dataset)[args.split]
    LOGGER.info("Loaded split=%s rows=%d", args.split, len(rows))

    rng = random.Random(args.seed)
    seen: set[str] = set()
    out_rows: list[dict] = []
    aligned = fallback = dropped = 0

    for row in rows:
        product_mapped = row["product"]
        reactants_mapped = row["reactants"]
        product_canonical = canonicalize_smiles_without_atom_maps(product_mapped)
        if not product_canonical or product_canonical in seen:
            continue
        if product_canonical in excluded:
            dropped += 1
            continue
        seen.add(product_canonical)

        prod_mol = Chem.MolFromSmiles(product_mapped)
        num_atoms = prod_mol.GetNumAtoms() if prod_mol is not None else 0
        roots: list[int | None] = [None]
        if args.copies > 1 and num_atoms:
            pool = list(range(num_atoms))
            rng.shuffle(pool)
            roots += pool[: args.copies - 1]

        for root_idx in roots:
            rendered = render_pair(reactants_mapped, product_mapped, root_idx)
            if rendered is None:
                if root_idx is None:
                    reactants_canonical = canonicalize_smiles_without_atom_maps(reactants_mapped)
                    out_rows.append(
                        {
                            "product_smiles": product_canonical,
                            "reactants_smiles": reactants_canonical or reactants_mapped,
                        }
                    )
                    fallback += 1
                continue
            aligned += 1
            out_rows.append({"product_smiles": rendered[0], "reactants_smiles": rendered[1]})

    LOGGER.info(
        "Reactions=%d dropped_as_test=%d rendered=%d fallback=%d",
        len(seen),
        dropped,
        aligned,
        fallback,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for out_row in out_rows:
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote %d row(s) to %s", len(out_rows), args.output)


if __name__ == "__main__":
    main()
