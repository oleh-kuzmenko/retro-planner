#!/usr/bin/env python3
"""Rewrite a reactants_train/val.jsonl pool into root-aligned form (Root-Aligned SMILES,
Zhong et al., Chemical Science 2022) for Model 1 fine-tuning.

Root-aligned SMILES: instead of both product and reactants using independently-canonical
SMILES, the reactant fragment sharing the reaction is rewritten to start ("rooted") at the
same atom the product's own canonical SMILES already starts at. This maximizes the shared
prefix between input and target, turning most of the seq2seq mapping into a "copy" the
model doesn't have to learn from scratch, and reportedly gives a large top-1 accuracy gain
(+9.23 points in one published ablation) purely from the representation change, with no
architecture change and no extra model capacity.

**Product SMILES is left untouched** (still plain canonical) -- only the target
(reactants_smiles) is rewritten. This sidesteps the usual "what root do I use at inference
time, when I don't have the reactants yet" problem entirely: the input format never changes,
so a checkpoint trained this way is used for inference exactly like any other -- the eval
script needs no changes on the input side. Only the model's raw *output* is now in this
rooted (but still 100% valid, RDKit-parseable) form, which downstream canonicalization
(already used everywhere in this project's evaluation code, e.g. `canonical_precursor_set`)
handles with no changes needed either.

Method: atom-map product<->reactants with RXNMapper (attention-guided; a pretrained
transformer atom mapper, not template-based, so it works directly on raw ORD SMILES with no
extra annotation needed). Find the product's own canonical root atom (RDKit
`CanonicalRankAtoms`, the atom ranked 0), look up its map number, and find the reactant
fragment containing an atom with that same map number. Reroot only that one fragment
(RDKit `MolToSmiles(..., rootedAtAtom=...)`), strip all atom-map numbers, and place it first
among the (canonicalized, unmodified) other fragments.

Falls back to the original (unmodified) reactants_smiles when: RXNMapper fails outright, the
product's canonical root atom has no map number, or no reactant fragment contains that map
number (all real, expected failure modes -- kept as plain canonical SMILES rather than
dropped, so the pool size is unchanged).

Example:
    python scripts/build_root_aligned_data.py \\
        --input data/v2_ord_train/reactants_train.jsonl \\
        --output data/v2_ord_train_rootaligned/reactants_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rdkit import Chem, RDLogger
from rxnmapper import RXNMapper

RDLogger.DisableLog("rdApp.*")

LOGGER = logging.getLogger("retro_eval.build_root_aligned_data")


def canonical_root_atom_idx(mol) -> int:
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    return ranks.index(0)


def root_reactants_on_product_root(mapped_rxn: str) -> str | None:
    reactants_mapped, product_mapped = mapped_rxn.split(">>")
    prod_mol = Chem.MolFromSmiles(product_mapped)
    if prod_mol is None:
        return None
    root_map_num = prod_mol.GetAtomWithIdx(canonical_root_atom_idx(prod_mol)).GetAtomMapNum()
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

    for atom in aligned_frag.GetAtoms():
        atom.SetAtomMapNum(0)
    rooted_smiles = Chem.MolToSmiles(aligned_frag, rootedAtAtom=aligned_idx, canonical=True)

    other_clean = []
    for frag in other_frags:
        fmol = Chem.MolFromSmiles(frag)
        if fmol is None:
            other_clean.append(frag)
            continue
        for atom in fmol.GetAtoms():
            atom.SetAtomMapNum(0)
        other_clean.append(Chem.MolToSmiles(fmol))

    return ".".join([rooted_smiles] + other_clean)


def same_molecule_set(a: str, b: str) -> bool:
    try:
        return {Chem.CanonSmiles(s) for s in a.split(".")} == {Chem.CanonSmiles(s) for s in b.split(".")}
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32, help="RXNMapper batch size.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    rows = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    LOGGER.info("Loaded %d row(s) from %s", len(rows), args.input)

    mapper = RXNMapper()
    aligned_count = 0
    verified_same_count = 0
    fallback_count = 0
    out_rows = []

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        rxn_smiles_list = [f"{r['reactants_smiles']}>>{r['product_smiles']}" for r in batch]
        try:
            results = mapper.get_attention_guided_atom_maps(rxn_smiles_list)
        except Exception as exc:
            LOGGER.warning("Batch at %d failed entirely (%s); falling back to canonical for all.", start, exc)
            results = [None] * len(batch)

        for row, result in zip(batch, results):
            new_reactants = None
            if result is not None:
                try:
                    new_reactants = root_reactants_on_product_root(result["mapped_rxn"])
                except Exception:
                    new_reactants = None
                if new_reactants is not None and not same_molecule_set(row["reactants_smiles"], new_reactants):
                    LOGGER.warning("Molecule-set mismatch after rooting, falling back: %s", row["product_smiles"][:80])
                    new_reactants = None
                elif new_reactants is not None:
                    verified_same_count += 1

            if new_reactants is not None:
                aligned_count += 1
                out_rows.append({"product_smiles": row["product_smiles"], "reactants_smiles": new_reactants})
            else:
                fallback_count += 1
                out_rows.append(row)

        if (start // args.batch_size + 1) % 10 == 0:
            LOGGER.info(
                "Processed %d/%d (aligned=%d, fallback=%d)",
                min(start + args.batch_size, len(rows)),
                len(rows),
                aligned_count,
                fallback_count,
            )

    LOGGER.info(
        "Done. aligned=%d (%.1f%%) verified_same_molecules=%d fallback=%d (%.1f%%)",
        aligned_count,
        100 * aligned_count / len(rows),
        verified_same_count,
        fallback_count,
        100 * fallback_count / len(rows),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote %d row(s) to %s", len(out_rows), args.output)


if __name__ == "__main__":
    main()
