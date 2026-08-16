#!/usr/bin/env python3
"""Separate "wrote the reaction down differently" from "proposed different chemistry".

A cross-source retrosynthesis number mixes two unrelated failures. USPTO-50K lists only
the molecules that donate atoms to the product -- verified here, not assumed: of 8,580
fragments in its test split, **zero** contribute no mapped atom. ORD lists the whole
vessel charge, dissociated: `[Na+]` occurs 1,502 times in the 5,687-record conditions
test, `O=C([O-])[O-]` 559 times. A model trained on one convention therefore scores near
zero against the other even when its disconnection is right.

So this script scores each run twice: once against the reference as written, and once
against both sides reduced to substrates. The gap between those two numbers is
bookkeeping; what survives the reduction is chemistry.

    substrate = the molecule donates at least one heavy atom to the product

That definition is the one USPTO-50K already follows, which is what makes the two
sources commensurable. It puts Boc2O among the substrates -- its atoms end up in the
product -- where a chemist would call it a reagent; the definition is kept because it is
objective and source-independent, and because USPTO's own atom maps agree with it (the
first test record maps exactly the half of Boc2O that is transferred).

**Applying it without atom maps.** ORD carries none (0 of 2,000 sampled rows), so the
rule is a proxy: ions, metal-containing species and carbon-free molecules are reagents
outright, and everything else is a substrate when its maximum common substructure with
the product covers at least 5 atoms and at least 70% of the fragment.

`--validate` measures that proxy where ground truth exists. USPTO-50K has no negatives
at all, so they are injected: two frequent ORD auxiliaries per record, which by
construction donate nothing. Hard negatives are included on purpose -- PPh3, DIPEA, EDC,
DCC, HOBt and DMAP all share substructure with ordinary products.

Usage:
    python scripts/models/substrate_role_analysis.py --validate
    python scripts/models/substrate_role_analysis.py \
        experiments/v2_model1_topk/baseline_ord_topk.json:"ReactionT5 base" \
        experiments/v2_model1_uspto50k/A5_rsmiles10_e2_ord300_topk.json:"A5"
"""

from __future__ import annotations

import argparse
import json
import random
import re
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

RDLogger.DisableLog("rdApp.*")

# Transition and alkali metals seen on ORD reactant sides. A fragment carrying one is a
# salt, a base or a catalyst; none of them survive into the product as themselves.
METALS = frozenset("Li Na K Mg Ca Zn Cu Pd Pt Ni Fe Al Ti Sn Cs Rb Ag Au Hg Mn Co Cr".split())

MIN_MCS_ATOMS = 5
MIN_MCS_FRACTION = 0.7

# Injected negatives for --validate: real reagents that donate nothing. The organic ones
# are the hard cases -- a triphenylphosphine shares two or three rings with many products.
INJECTED_REAGENTS = (
    "c1ccc(P(c2ccccc2)c2ccccc2)cc1",      # PPh3
    "CCN(C(C)C)C(C)C",                    # DIPEA
    "CCN(CC)CC",                          # NEt3
    "CCN=C=NCCCN(C)C",                    # EDC
    "C1CCC(N=C=NC2CCCCC2)CC1",            # DCC
    "On1nnc2ccccc21",                     # HOBt
    "CN(C)c1ccncc1",                      # DMAP
    "c1c[nH]cn1",                         # imidazole
    "Cc1ccc(S(=O)(=O)O)cc1",              # TsOH
    "[Na+]", "[K+]", "O=C([O-])[O-]", "[OH-]", "[H-]", "O", "CO", "C1CCOC1",
)

ATOM_MAP = re.compile(r":(\d+)\]")


@lru_cache(maxsize=None)
def _mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


@lru_cache(maxsize=None)
def canonical(smiles: str) -> str | None:
    mol = _mol(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


@lru_cache(maxsize=None)
def is_substrate(fragment: str, product: str) -> bool:
    frag, prod = _mol(fragment), _mol(product)
    if frag is None or prod is None:
        return False
    symbols = {atom.GetSymbol() for atom in frag.GetAtoms()}
    if symbols & METALS:
        return False
    if not any(atom.GetSymbol() == "C" for atom in frag.GetAtoms()):
        return False
    heavy = frag.GetNumHeavyAtoms()
    if heavy == 1:
        return False
    shared = rdFMCS.FindMCS(
        [frag, prod], timeout=2, ringMatchesRingOnly=True, completeRingsOnly=False
    ).numAtoms
    return shared >= MIN_MCS_ATOMS and shared >= MIN_MCS_FRACTION * heavy


def molecule_set(smiles: str, product: str, keep_all: bool) -> set[str]:
    out: set[str] = set()
    for fragment in smiles.split("."):
        if not fragment:
            continue
        canon = canonical(fragment)
        if canon is None:
            continue
        if keep_all or is_substrate(canon, product):
            out.add(canon)
    return out


def score_run(path: Path, ks=(1, 3, 5)) -> dict:
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    out = {}
    for mode in ("full", "substrates"):
        keep_all = mode == "full"
        hits = {k: 0 for k in ks}
        for record in records:
            product = canonical(record["product_smiles"])
            reference = molecule_set(record["reactants_smiles"], product, keep_all)
            if not reference:
                continue
            rank = None
            for index, candidate in enumerate(record["candidates"][: max(ks)], start=1):
                if molecule_set(candidate, product, keep_all) == reference:
                    rank = index
                    break
            for k in ks:
                if rank is not None and rank <= k:
                    hits[k] += 1
        out[mode] = {k: hits[k] / len(records) for k in ks}
    return out


def validate(sample_size: int, seed: int) -> None:
    from datasets import load_dataset

    random.seed(seed)
    dataset = load_dataset("bisectgroup/USPTO_50K", split="test")

    unmapped = 0
    fragments = 0
    for row in dataset:
        product_maps = set(ATOM_MAP.findall(row["product"]))
        for fragment in row["reactants"].split("."):
            fragments += 1
            if not set(ATOM_MAP.findall(fragment)) & product_maps:
                unmapped += 1
    print(f"USPTO-50K test: {fragments} fragments, {unmapped} donate no atom to the product")

    def strip_maps(smiles: str) -> str | None:
        return canonical(ATOM_MAP.sub("]", smiles))

    pairs = []
    for index in random.sample(range(len(dataset)), sample_size):
        row = dataset[index]
        product = strip_maps(row["product"])
        if product is None:
            continue
        for fragment in row["reactants"].split("."):
            clean = strip_maps(fragment)
            if clean:
                pairs.append((clean, product, True))
        for reagent in random.sample(INJECTED_REAGENTS, 2):
            pairs.append((canonical(reagent), product, False))

    tp = fp = fn = tn = 0
    for fragment, product, truly_substrate in pairs:
        predicted = is_substrate(fragment, product)
        if truly_substrate and predicted:
            tp += 1
        elif truly_substrate:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1

    print(f"validation pairs: {len(pairs)} ({tp + fn} substrates, {tn + fp} injected reagents)")
    print(f"  substrate recall  {tp / (tp + fn):.1%}")
    print(f"  reagent recall    {tn / (tn + fp):.1%}")
    print(f"  reagent precision {tn / (tn + fn):.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="top-k result files as path[:label]")
    parser.add_argument("--validate", action="store_true", help="measure the proxy against USPTO atom maps")
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.validate:
        validate(args.sample_size, args.seed)
        if args.runs:
            print()

    if not args.runs:
        return

    print(f"{'run':32s} {'reference':16s} {'top-1':>7s} {'top-3':>7s} {'top-5':>7s}")
    for entry in args.runs:
        path, _, label = entry.partition(":")
        label = label or Path(path).stem
        scored = score_run(Path(path))
        for mode, name in (("full", "as written"), ("substrates", "substrates only")):
            row = scored[mode]
            print(f"{label:32s} {name:16s} {row[1]:6.1%} {row[3]:6.1%} {row[5]:6.1%}")


if __name__ == "__main__":
    main()
